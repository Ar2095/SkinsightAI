"""Build a temporal hold-out test set from the ISIC Archive.

The Skinsight model was trained in May 2025 on images downloaded from the ISIC
Archive (the checkpoint was committed 2025-05-20). Any image the archive created
after CUTOFF cannot have been in that training set, so it is a clean held-out test.

The ISIC API pages images in creation order, so we start the cursor at CUTOFF and
read everything after it. Two test splits are built:

  dermoscopic : every post-cutoff dermoscopic image with a Benign/Malignant
                diagnosis, one image per lesion. Benign images from the MEL-SELF
                trial are very numerous, so a seeded random subset is kept.
                This matches the image type the model was trained on.
  rcm         : reflectance confocal microscopy tiles. A different imaging
                modality the model never saw: an out-of-distribution stress test.
  clinical    : macroscopic (non-dermoscopic) close-up photos of the lesions,
                the closest thing in the archive to a phone photo. Very small.

Usage:
  python eval/fetch_holdout.py              # metadata + selection + download (~420 MB)
  python eval/fetch_holdout.py --refresh    # re-query the API instead of the cached CSV
Outputs eval/data/metadata_all.csv, eval/data/manifest.csv, eval/data/images/.
"""

import argparse
import base64
import csv
import random
import time
import urllib.parse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

API = "https://api.isic-archive.com/api/v2/images/"
CUTOFF = "2025-06-01 00:00:00+00:00"
DATA = Path(__file__).parent / "data"

# ISIC (diagnosis_1, diagnosis_2 prefix) -> the model's 11 subtype classes.
# The archive appends qualifiers to some diagnosis_2 values
# (e.g. "... - Follicular"), so we match on a prefix.
CLASS_MAP = [
    ("Malignant", "Malignant soft tissue proliferations", "malignant_soft-tissue-proliferations"),
    ("Malignant", "Malignant adnexal epithelial proliferations", "malignant_adnexal-epithelial-proliferations"),
    ("Malignant", "Malignant melanocytic proliferations", "malignant_melanocytic-proliferation"),
    ("Malignant", "Malignant epidermal proliferations", "malignant_epidermal-proliferations"),
    ("Benign", "Flat melanotic pigmentations", "benign_flat-melanotic-pigmentations"),
    ("Benign", "Benign soft tissue proliferations", "benign_soft-tissue-proliferations"),
    ("Benign", "Inflammatory or infectious diseases", "benign_inflammatory-or-infectious-diseases"),
    ("Benign", "Benign adnexal epithelial proliferations", "benign_adnexal-epithelial-proliferations"),
    ("Benign", "Benign melanocytic proliferations", "benign_melanocytic-proliferations"),
    ("Benign", "Benign epidermal proliferations", "benign_epidermal-proliferations"),
    ("Benign", "Benign - Other", "benign_other"),
]

# image_type -> test split. "RCM: mosaic" images (stitched gigapixel scans, ~200 MB
# each) are left out: they are meaningless once shrunk to the model's 224 px input.
SPLITS = {"dermoscopic": "dermoscopic", "RCM: tile": "rcm", "RCM: macroscopic": "clinical"}

FIELDS = ["isic_id", "split", "source", "license", "image_type", "diagnosis_1", "diagnosis_2",
          "diagnosis_3", "diagnosis_confirm_type", "binary", "subtype", "lesion_id", "patient_id",
          "anatom_site", "age_approx", "sex", "fitzpatrick_skin_type", "url"]


def map_subtype(d1, d2):
    for want1, prefix, label in CLASS_MAP:
        if d1 == want1 and d2 and d2.startswith(prefix):
            return label
    return ""


def cursor_for(ts):
    return base64.b64encode(("p=" + urllib.parse.quote_plus(ts)).encode()).decode()


def get(url):
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r
        except requests.RequestException:
            time.sleep(2 ** attempt)
    raise SystemExit(f"ISIC API request failed: {url}")


def fetch_metadata():
    rows, url = [], f"{API}?limit=100&cursor={urllib.parse.quote(cursor_for(CUTOFF))}"
    while url:
        page = get(url).json()
        for im in page["results"]:
            c, a = im["metadata"].get("clinical", {}), im["metadata"].get("acquisition", {})
            d1, d2 = c.get("diagnosis_1", ""), c.get("diagnosis_2", "")
            itype = a.get("image_type", "")
            rows.append({
                "isic_id": im["isic_id"],
                "split": SPLITS.get(itype, "other"),
                "source": im.get("attribution", ""),
                "license": im.get("copyright_license", ""),
                "image_type": itype,
                "diagnosis_1": d1,
                "diagnosis_2": d2,
                "diagnosis_3": c.get("diagnosis_3", ""),
                "diagnosis_confirm_type": c.get("diagnosis_confirm_type", ""),
                "binary": {"Benign": 0, "Malignant": 1}.get(d1, ""),
                "subtype": map_subtype(d1, d2),
                "lesion_id": c.get("lesion_id", ""),
                "patient_id": c.get("patient_id", ""),
                "anatom_site": c.get("anatom_site_1", ""),
                "age_approx": c.get("age_approx", ""),
                "sex": c.get("sex", ""),
                "fitzpatrick_skin_type": c.get("fitzpatrick_skin_type", ""),
                "url": im["files"]["full"]["url"],
            })
        url = page.get("next")
        if len(rows) % 1000 == 0:
            print(f"  fetched {len(rows)} image records", flush=True)
    return rows


def one_per_lesion(rows, rng):
    """Keep one random image per lesion_id (images without a lesion_id are kept)."""
    by_lesion, out = {}, []
    for r in rows:
        if r["lesion_id"]:
            by_lesion.setdefault(r["lesion_id"], []).append(r)
        else:
            out.append(r)
    for lesion in sorted(by_lesion):
        out.append(rng.choice(sorted(by_lesion[lesion], key=lambda r: r["isic_id"])))
    return out


def select(rows, benign_cap, seed):
    rng = random.Random(seed)
    labelled = [r for r in rows if r["binary"] != "" and r["split"] != "other"]
    derm = one_per_lesion([r for r in labelled if r["split"] == "dermoscopic"], rng)
    rest = [r for r in labelled if r["split"] != "dermoscopic"]  # each RCM image is distinct

    # MEL-SELF contributes ~3000 benign images; cap it so the download stays small.
    melself_benign = sorted((r for r in derm if r["source"].startswith("MEL-SELF") and str(r["binary"]) == "0"),
                            key=lambda r: r["isic_id"])
    rng.shuffle(melself_benign)
    keep = set(r["isic_id"] for r in melself_benign[:benign_cap])
    derm = [r for r in derm if not (r["source"].startswith("MEL-SELF") and str(r["binary"]) == "0")
            or r["isic_id"] in keep]
    return sorted(derm + rest, key=lambda r: (r["split"], r["isic_id"]))


def download(row, outdir):
    path = outdir / f"{row['isic_id']}.jpg"
    if path.exists() and path.stat().st_size > 0:
        return
    r = get(row["url"])
    path.write_bytes(r.content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benign-cap", type=int, default=400,
                    help="max MEL-SELF benign lesions kept in the dermoscopic split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refresh", action="store_true", help="re-query the ISIC API")
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    meta_path = DATA / "metadata_all.csv"
    if args.refresh or not meta_path.exists():
        print(f"Querying ISIC for images created after {CUTOFF} ...")
        rows = fetch_metadata()
        with meta_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    with meta_path.open() as f:
        rows = list(csv.DictReader(f))
    print(f"{len(rows)} images created after {CUTOFF}")

    sample = select(rows, args.benign_cap, args.seed)
    with (DATA / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sample)
    for split in ("dermoscopic", "rcm", "clinical"):
        s = [r for r in sample if r["split"] == split]
        print(f"  {split:12s} {len(s):4d} images | malignant {sum(r['binary'] == '1' for r in s)}"
              f" | benign {sum(r['binary'] == '0' for r in s)}")
    print("  subtype labels:", dict(Counter(r["subtype"] for r in sample if r["subtype"])))

    if args.no_download:
        return
    outdir = DATA / "images"
    outdir.mkdir(exist_ok=True)
    with ThreadPoolExecutor(16) as pool:
        for i, _ in enumerate(pool.map(lambda r: download(r, outdir), sample), 1):
            if i % 100 == 0 or i == len(sample):
                print(f"  downloaded {i}/{len(sample)}", flush=True)
    print("Done.")


if __name__ == "__main__":
    main()
