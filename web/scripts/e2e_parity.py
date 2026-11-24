"""End-to-end check: the deployed web app gives the same answer as PyTorch.

Opens the site in headless Chrome, runs each image through the real UI
(upload -> default crop -> Analyze), reads the crop box and P(malignant) the
page computed, then runs the identical crop through PyTorch.

Usage:
  cd web && npm run build && npx vite preview --port 4173 &
  python web/scripts/e2e_parity.py [--url http://localhost:4173] [images...]
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from skinsight_model import load_model, preprocess  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:4173")
    ap.add_argument("images", nargs="*", default=[str(p) for p in sorted((ROOT / "web/public/samples").glob("*.jpg"))])
    args = ap.parse_args()
    model = load_model()
    worst = 0.0
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(args.url)
        page.wait_for_selector("#model-status.ready", timeout=120_000)
        for img in args.images:
            page.click("#back-2") if page.is_visible("#back-2") else None
            page.evaluate("document.querySelector('#stage-1').classList.contains('hidden') && document.querySelector('#restart').click()")
            page.set_input_files("#file", img)
            page.wait_for_selector(".cropper-container", timeout=20_000)
            page.wait_for_timeout(600)
            page.click("#analyze")
            page.wait_for_function("window.__skinsightLastRun && !document.querySelector('#stage-3').classList.contains('hidden')")
            run = page.evaluate("window.__skinsightLastRun")
            page.evaluate("window.__skinsightLastRun = null")
            c = run["crop"]
            im = ImageOps.exif_transpose(Image.open(img)).convert("RGB")
            crop = im.crop((c["x"], c["y"], c["x"] + c["w"], c["y"] + c["h"]))
            with torch.no_grad():
                p = torch.sigmoid(model(preprocess(crop).unsqueeze(0))[0]).item()
            d = abs(p - run["pMalignant"])
            worst = max(worst, d)
            print(f"{Path(img).name:28s} crop {c['w']}x{c['h']}  browser {run['pMalignant']:.6f}  pytorch {p:.6f}  |diff| {d:.1e}")
            page.click("#restart")
        browser.close()
    print(f"max |diff| = {worst:.1e}")


if __name__ == "__main__":
    main()
