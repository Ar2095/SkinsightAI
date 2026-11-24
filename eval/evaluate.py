"""Evaluate the Skinsight model on the temporal hold-out set built by fetch_holdout.py.

Usage:
  python eval/fetch_holdout.py   # once, downloads the test images
  python eval/evaluate.py        # writes eval/results/{predictions.csv,metrics.json,results.md,*.png}

Everything is deterministic (fixed seeds); a CPU run takes a few minutes.
"""

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             roc_auc_score, roc_curve)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skinsight_model import MEAN, STD, SUBTYPES, load_model  # noqa: E402

HERE = Path(__file__).parent
DATA, OUT = HERE / "data", HERE / "results"
THRESHOLD = 0.5      # the app calls a lesion "Malignant" when P(malignant) > 0.5
N_BOOT = 2000
rng = np.random.default_rng(0)
torch.manual_seed(0)

# Published reference point (different images, so context, not a head-to-head):
# Haenssle et al., Annals of Oncology 2018, 58 dermatologists reading dermoscopy
# images only (level I): mean sensitivity 86.6%, specificity 71.3% for melanoma.
DERMATOLOGIST_REF = {"sensitivity": 0.866, "specificity": 0.713,
                     "source": "Haenssle et al., Ann Oncol 2018 (dermoscopy only, 58 dermatologists)"}

_norm_mean = torch.tensor(MEAN).view(3, 1, 1)
_norm_std = torch.tensor(STD).view(3, 1, 1)


def to_tensor(img224: Image.Image) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(img224.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (x - _norm_mean) / _norm_std


def base_224(path):
    """App preprocessing step 1: squash the (user-cropped) image to 224x224."""
    return Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)


# ---------------------------------------------------------------- perturbations
def jpeg(img, q):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    return Image.open(buf).convert("RGB")


def zoom_out(img, frac):
    """Lesion occupies `frac` of the frame width: shrink and pad with the border colour."""
    side = max(8, int(224 * frac))
    small = img.resize((side, side), Image.BILINEAR)
    arr = np.asarray(img)
    border = tuple(int(v) for v in np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]]).mean(0))
    canvas = Image.new("RGB", (224, 224), border)
    canvas.paste(small, ((224 - side) // 2, (224 - side) // 2))
    return canvas


def zoom_in(img, frac):
    """Crop the central `frac` of the frame (user zoomed in too far) and resize back."""
    side = int(224 * frac)
    o = (224 - side) // 2
    return img.crop((o, o, o + side, o + side)).resize((224, 224), Image.BILINEAR)


PERTURBATIONS = {
    "none": [("original", lambda im: im)],
    "blur": [(f"gaussian r={r}", (lambda r: lambda im: im.filter(ImageFilter.GaussianBlur(r)))(r))
             for r in (1, 2, 4, 8)],
    "jpeg": [(f"quality {q}", (lambda q: lambda im: jpeg(im, q))(q)) for q in (50, 20, 10, 5)],
    "brightness": [(f"x{f}", (lambda f: lambda im: ImageEnhance.Brightness(im).enhance(f))(f))
                   for f in (0.5, 0.75, 1.25, 1.5)],
    "resolution": [(f"{s}px", (lambda s: lambda im: im.resize((s, s), Image.BILINEAR)
                               .resize((224, 224), Image.BILINEAR))(s)) for s in (112, 56, 28)],
    "zoom_out": [(f"lesion fills {int(f * 100)}%", (lambda f: lambda im: zoom_out(im, f))(f))
                 for f in (0.75, 0.5, 0.3)],
    "zoom_in": [(f"central {int(f * 100)}%", (lambda f: lambda im: zoom_in(im, f))(f))
                for f in (0.75, 0.5, 0.3)],
    "orientation": [("h-flip", lambda im: im.transpose(Image.FLIP_LEFT_RIGHT)),
                    ("rotate 90", lambda im: im.transpose(Image.ROTATE_90))],
}


@torch.no_grad()
def predict(model, images, batch=64):
    p_mal, p_sub = [], []
    for i in range(0, len(images), batch):
        x = torch.stack([to_tensor(im) for im in images[i:i + batch]])
        b, s = model(x)
        p_mal.append(torch.sigmoid(b).squeeze(1).numpy())
        p_sub.append(torch.softmax(s, 1).numpy())
    return np.concatenate(p_mal), np.concatenate(p_sub)


# ---------------------------------------------------------------- metrics
def binary_metrics(y, p, thr=THRESHOLD):
    y, p = np.asarray(y, int), np.asarray(p, float)
    pred = (p > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = {"n": int(len(y)), "n_malignant": int(y.sum()), "n_benign": int((1 - y).sum()),
         "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
         "sensitivity": tp / (tp + fn) if tp + fn else None,
         "specificity": tn / (tn + fp) if tn + fp else None,
         "ppv": tp / (tp + fp) if tp + fp else None,
         "npv": tn / (tn + fn) if tn + fn else None,
         "accuracy": (tp + tn) / len(y)}
    if 0 < y.sum() < len(y):
        m["auroc"] = roc_auc_score(y, p)
        m["auprc"] = average_precision_score(y, p)
        m["balanced_accuracy"] = (m["sensitivity"] + m["specificity"]) / 2
        m["brier"] = brier_score_loss(y, p)
    return m


def bootstrap_ci(y, p, fn, n=N_BOOT):
    """Stratified bootstrap (resample malignant and benign separately) 95% CI."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(fn(y[idx], p[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def sens(y, p):
    return ((p > THRESHOLD) & (y == 1)).sum() / (y == 1).sum()


def spec(y, p):
    return ((p <= THRESHOLD) & (y == 0)).sum() / (y == 0).sum()


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if mask.any():
            e += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return e


def spec_at_sens(y, p, target):
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target)[0][0]
    return {"target_sensitivity": target, "specificity": float(1 - fpr[ok]),
            "threshold": float(thr[ok])}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if o is None or np.isnan(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA / "manifest.csv", dtype={"age_approx": "float"})
    df["binary"] = df["binary"].astype(int)
    missing = [i for i in df.isic_id if not (DATA / "images" / f"{i}.jpg").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images missing; run eval/fetch_holdout.py first")

    model = load_model()
    print(f"Loading {len(df)} images ...")
    imgs = [base_224(DATA / "images" / f"{i}.jpg") for i in df.isic_id]

    t0 = time.perf_counter()
    p_mal, p_sub = predict(model, imgs)
    df["p_malignant"] = p_mal
    df["pred_subtype"] = [SUBTYPES[k] for k in p_sub.argmax(1)]
    df["p_pred_subtype"] = p_sub.max(1)
    df["subtype_mal_mass"] = p_sub[:, [i for i, s in enumerate(SUBTYPES) if s.startswith("malignant")]].sum(1)
    for k, s in enumerate(SUBTYPES):
        df[f"p_{s}"] = p_sub[:, k]
    print(f"Inference: {len(df)} images in {time.perf_counter() - t0:.1f}s")

    # single-image latency (what one app user waits for, CPU, batch of 1)
    x = to_tensor(imgs[0]).unsqueeze(0)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        t0 = time.perf_counter()
        for _ in range(20):
            model(x)
    latency_ms = (time.perf_counter() - t0) / 20 * 1000

    results = {"threshold": THRESHOLD, "cutoff_created_after": "2025-06-01",
               "cpu_latency_ms_batch1": latency_ms, "torch_threads": torch.get_num_threads(),
               "dermatologist_reference": DERMATOLOGIST_REF, "splits": {}}

    # ---- primary: binary on each split
    for split, g in df.groupby("split"):
        y, p = g.binary.values, g.p_malignant.values
        m = binary_metrics(y, p)
        if "auroc" in m and min(m["n_malignant"], m["n_benign"]) >= 5:
            m["auroc_ci95"] = bootstrap_ci(y, p, roc_auc_score)
            m["sensitivity_ci95"] = bootstrap_ci(y, p, sens)
            m["specificity_ci95"] = bootstrap_ci(y, p, spec)
            m["ece"] = ece(y, p)
            m["mean_p_malignant_benign"] = float(p[y == 0].mean())
            m["mean_p_malignant_malignant"] = float(p[y == 1].mean())
            m["spec_at_sens"] = [spec_at_sens(y, p, t) for t in (0.8, 0.9, 0.95)]
            # Same task scored from the subtype head (sum of the 4 malignant subtype probs).
            m["auroc_from_subtype_head"] = roc_auc_score(y, g.subtype_mal_mass.values)
        results["splits"][split] = m

    derm = df[df.split == "dermoscopic"].copy()
    y, p = derm.binary.values, derm.p_malignant.values

    # ---- baselines on the primary split
    prev = y.mean()
    results["baselines"] = {
        "always_benign": {"accuracy": 1 - prev, "sensitivity": 0.0, "specificity": 1.0, "auroc": 0.5},
        "refer_everyone": {"accuracy": prev, "sensitivity": 1.0, "specificity": 0.0, "auroc": 0.5},
        "random_coin": {"sensitivity": 0.5, "specificity": 0.5, "auroc": 0.5},
    }

    # ---- how often a given score was actually malignant (shown next to each result in the app)
    edges = [0.0, 0.1, 0.5, 0.9, 1.0]
    results["score_bins_dermoscopic"] = [
        {"low": lo, "high": hi, "n": int(m.sum()), "n_malignant": int(y[m].sum()),
         "malignant_rate": float(y[m].mean()) if m.any() else None}
        for lo, hi in zip(edges[:-1], edges[1:])
        for m in [(p >= lo) & ((p < hi) if hi < 1 else (p <= hi))]
    ]

    # ---- subgroups on the primary split (only where both classes have >= 5 cases)
    derm["age_band"] = pd.cut(derm.age_approx, [0, 40, 60, 120], labels=["<40", "40-59", "60+"])
    derm["histopath"] = np.where(derm.diagnosis_confirm_type == "histopathology", "histopathology", "other")
    derm["source_short"] = derm.source.str.split(",").str[0].str.replace("Dr. ", "", regex=False)
    sub = {}
    for col in ["source_short", "histopath", "sex", "age_band", "fitzpatrick_skin_type", "anatom_site"]:
        for val, g in derm.groupby(col, observed=True):
            if not isinstance(val, str) or not val:
                continue
            gy, gp = g.binary.values, g.p_malignant.values
            entry = {"n": len(g), "n_malignant": int(gy.sum()), "n_benign": int((1 - gy).sum()),
                     "sensitivity": sens(gy, gp) if gy.sum() else None,
                     "specificity": spec(gy, gp) if (1 - gy).sum() else None}
            if min(gy.sum(), (1 - gy).sum()) >= 5:
                entry["auroc"] = roc_auc_score(gy, gp)
            sub[f"{col}={val}"] = entry
    results["subgroups_dermoscopic"] = sub

    # ---- subtype head (only 5 of 11 classes are present in the hold-out data)
    s = df[df.subtype.notna() & (df.split == "dermoscopic")]
    s_all = df[df.subtype.notna()]
    def subtype_block(t):
        cm = pd.crosstab(t.subtype, t.pred_subtype)
        top3 = np.mean([lab in sorted(SUBTYPES, key=lambda c: -row[f"p_{c}"])[:3]
                        for lab, (_, row) in zip(t.subtype, t.iterrows())])
        per_class = {lab: float((g.pred_subtype == lab).mean()) for lab, g in t.groupby("subtype")}
        return {"n": len(t), "top1_accuracy": float((t.subtype == t.pred_subtype).mean()),
                "top3_accuracy": float(top3), "per_class_recall": per_class,
                "majority_class_baseline": float(t.subtype.value_counts(normalize=True).iloc[0]),
                "confusion": {r: {c: int(v) for c, v in cm.loc[r].items() if v} for r in cm.index}}
    results["subtype_dermoscopic"] = subtype_block(s)
    results["subtype_all_splits"] = subtype_block(s_all)

    # ---- robustness: perturb the primary split and re-score
    print("Robustness sweep ...")
    derm_imgs = [imgs[i] for i in derm.index]
    rob = {}
    for family, variants in PERTURBATIONS.items():
        for name, fn in variants:
            pm, _ = predict(model, [fn(im) for im in derm_imgs])
            rob[f"{family}: {name}"] = {
                "auroc": roc_auc_score(y, pm), "sensitivity": sens(y, pm), "specificity": spec(y, pm),
                "flip_rate": float(((pm > THRESHOLD) != (p > THRESHOLD)).mean())}
            print(f"  {family:12s} {name:18s} AUROC {rob[f'{family}: {name}']['auroc']:.3f}")
    results["robustness_dermoscopic"] = rob

    df.drop(columns=["url"]).to_csv(OUT / "predictions.csv", index=False)
    results = clean(results)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    write_markdown(results)
    make_figures(df, results)
    print(f"Wrote {OUT}/metrics.json, results.md, predictions.csv and figures")


def fmt(v, pct=True):
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%" if pct else f"{v:.3f}"


def write_markdown(r):
    L = ["# Skinsight evaluation results", "",
         "Generated by `eval/evaluate.py`. Test images were added to the ISIC Archive after "
         "2025-06-01, after the model was trained, so none can be in its training set.", "",
         "## Benign vs. malignant, per split", "",
         "| Split | n (mal / ben) | AUROC [95% CI] | Sensitivity [95% CI] | Specificity [95% CI] | PPV | NPV | Brier | ECE |",
         "|---|---|---|---|---|---|---|---|---|"]
    for split in ("dermoscopic", "rcm", "clinical"):
        m = r["splits"].get(split)
        if not m:
            continue
        ci = lambda k: f" [{fmt(m[k][0], k != 'auroc_ci95')}–{fmt(m[k][1], k != 'auroc_ci95')}]" if k in m else ""
        L.append(f"| {split} | {m['n']} ({m['n_malignant']} / {m['n_benign']}) | {fmt(m.get('auroc'), False)}{ci('auroc_ci95')} "
                 f"| {fmt(m['sensitivity'])}{ci('sensitivity_ci95')} | {fmt(m['specificity'])}{ci('specificity_ci95')} "
                 f"| {fmt(m['ppv'])} | {fmt(m['npv'])} | {fmt(m.get('brier'), False)} | {fmt(m.get('ece'), False)} |")
    d = r["splits"]["dermoscopic"]
    L += ["", f"Threshold: P(malignant) > {r['threshold']} (the app's rule). "
          f"Single-image CPU latency: {r['cpu_latency_ms_batch1']:.0f} ms.", "",
          "### Other operating points (dermoscopic; thresholds picked on the test set, so optimistic)", "",
          "| Target sensitivity | Specificity | Threshold |", "|---|---|---|"]
    for s in d["spec_at_sens"]:
        L.append(f"| {fmt(s['target_sensitivity'])} | {fmt(s['specificity'])} | {s['threshold']:.4f} |")
    ref = r["dermatologist_reference"]
    L += ["", "### Observed malignant rate by score (dermoscopic)", "",
          "| P(malignant) | Lesions | Malignant | Rate |", "|---|---|---|---|"]
    for b in r["score_bins_dermoscopic"]:
        L.append(f"| {b['low']:.1f}–{b['high']:.1f} | {b['n']} | {b['n_malignant']} | {fmt(b['malignant_rate'])} |")
    L += ["", "## Baselines and reference (dermoscopic split)", "",
          "| Method | Sensitivity | Specificity | AUROC |", "|---|---|---|---|",
          f"| Skinsight (threshold 0.5) | {fmt(d['sensitivity'])} | {fmt(d['specificity'])} | {fmt(d['auroc'], False)} |",
          f"| Skinsight subtype head (sum of malignant classes) | | | {fmt(d['auroc_from_subtype_head'], False)} |",
          "| Always say benign | 0.0% | 100.0% | 0.500 |",
          "| Refer everyone | 100.0% | 0.0% | 0.500 |",
          f"| Dermatologists, dermoscopy only (published, different images) | {fmt(ref['sensitivity'])} | {fmt(ref['specificity'])} | |",
          "", f"Reference: {ref['source']}.", "",
          "## Subgroups (dermoscopic split)", "",
          "| Subgroup | n (mal / ben) | Sensitivity | Specificity | AUROC |", "|---|---|---|---|---|"]
    for k, v in r["subgroups_dermoscopic"].items():
        L.append(f"| {k} | {v['n']} ({v['n_malignant']} / {v['n_benign']}) | {fmt(v['sensitivity'])} "
                 f"| {fmt(v['specificity'])} | {fmt(v.get('auroc'), False)} |")
    for key, title in (("subtype_dermoscopic", "dermoscopic"), ("subtype_all_splits", "all splits")):
        st = r[key]
        L += ["", f"## Subtype head ({title}, n = {st['n']}; only 5 of 11 classes present)", "",
              f"Top-1 accuracy {fmt(st['top1_accuracy'])}, top-3 accuracy {fmt(st['top3_accuracy'])}, "
              f"majority-class baseline {fmt(st['majority_class_baseline'])}.", "",
              "| True class | Recall |", "|---|---|"]
        for c, v in st["per_class_recall"].items():
            L.append(f"| {c} | {fmt(v)} |")
    L += ["", "## Robustness (dermoscopic split, perturbation applied after the 224 px resize)", "",
          "| Perturbation | AUROC | Sensitivity | Specificity | Decisions flipped |", "|---|---|---|---|---|"]
    for k, v in r["robustness_dermoscopic"].items():
        L.append(f"| {k} | {fmt(v['auroc'], False)} | {fmt(v['sensitivity'])} | {fmt(v['specificity'])} | {fmt(v['flip_rate'])} |")
    (OUT / "results.md").write_text("\n".join(L) + "\n")


def make_figures(df, r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plum, rose, pink, ink, grid = "#7A1240", "#E0648F", "#F4B6CB", "#2B1622", "#EBD5DE"
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#B98DA0",
                         "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink,
                         "axes.spines.top": False, "axes.spines.right": False})

    # ROC
    fig, ax = plt.subplots(figsize=(6, 5.2), dpi=200)
    for split, color, label in (("dermoscopic", plum, "Dermoscopic (primary)"),
                                ("rcm", rose, "Confocal microscopy (out of distribution)"),
                                ("clinical", "#C7A0B0", "Clinical close-up (n=18)")):
        g = df[df.split == split]
        if g.binary.nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(g.binary, g.p_malignant)
        ax.plot(fpr, tpr, color=color, lw=2.4 if split == "dermoscopic" else 1.6,
                label=f"{label}  AUROC {r['splits'][split]['auroc']:.2f}")
    d = r["splits"]["dermoscopic"]
    ax.scatter([1 - d["specificity"]], [d["sensitivity"]], color=plum, zorder=5, s=40)
    ax.annotate("app threshold 0.5", (1 - d["specificity"], d["sensitivity"]),
                xytext=(8, -14), textcoords="offset points", color=plum, fontsize=8)
    ref = r["dermatologist_reference"]
    ax.scatter([1 - ref["specificity"]], [ref["sensitivity"]], marker="D", color=ink, s=30, zorder=5,
               label="Dermatologists, dermoscopy (published)")
    ax.plot([0, 1], [0, 1], ls="--", color="#C9AEB9", lw=1)
    ax.set(xlabel="False positive rate (1 − specificity)", ylabel="Sensitivity",
           xlim=(0, 1), ylim=(0, 1.01))
    ax.grid(color=grid, lw=0.6)
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "roc.png")
    plt.close(fig)

    # Calibration
    g = df[df.split == "dermoscopic"]
    fig, ax = plt.subplots(figsize=(5, 4.6), dpi=200)
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(g.p_malignant, bins) - 1, 0, 9)
    xs, ys, ns = [], [], []
    for b in range(10):
        m = idx == b
        if m.sum():
            xs.append(g.p_malignant[m].mean()); ys.append(g.binary[m].mean()); ns.append(m.sum())
    ax.plot([0, 1], [0, 1], ls="--", color="#C9AEB9", lw=1)
    ax.plot(xs, ys, color=plum, lw=2)
    ax.scatter(xs, ys, s=[12 + 2.5 * n ** 0.8 for n in ns], color=rose, edgecolor=plum, zorder=3)
    ax.set(xlabel="Predicted P(malignant)", ylabel="Observed fraction malignant", xlim=(0, 1), ylim=(0, 1))
    ax.grid(color=grid, lw=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "calibration.png")
    plt.close(fig)

    # Robustness
    rob = r["robustness_dermoscopic"]
    names = list(rob)
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=200)
    vals = [rob[n]["auroc"] for n in names]
    colors = [plum if n.startswith("none") else (rose if v >= rob[names[0]]["auroc"] - 0.05 else pink)
              for n, v in zip(names, vals)]
    ax.barh(range(len(names)), vals, color=colors)
    ax.axvline(rob[names[0]]["auroc"], color=plum, lw=1, ls="--")
    ax.axvline(0.5, color="#C9AEB9", lw=1)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.invert_yaxis()
    ax.set(xlim=(0.4, 1.0), xlabel="AUROC (dermoscopic split)")
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=7, color=ink)
    fig.tight_layout()
    fig.savefig(OUT / "robustness.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
