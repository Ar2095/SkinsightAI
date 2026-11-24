"""Render the README / Devpost banners from banner.html.

Usage (from the repo root, with the Vite dev server running in web/):
  cd web && npx vite --port 5179 &
  python web/devpost-src/render_banners.py
Writes PNG + WebP files to web/public/devpost/, which the deployed site serves at /devpost/.
"""

import json
import urllib.parse
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "web" / "public" / "devpost"
BASE = "http://localhost:5179/devpost-src/banner.html"

SECTIONS = {
    "problem": ("The Problem", "and who has it"),
    "what-it-does": ("What It Does", ""),
    "how-we-built-it": ("How We Built It", "method"),
    "evaluation": ("Evaluation", "what we measured and how"),
    "benefits-harms": ("Who Benefits", "and who could be harmed"),
    "limitations": ("Limitations", ""),
    "challenges": ("Challenges", ""),
    "what-we-learned": ("What We Learned", ""),
    "another-month": ("Another Month", "what we'd do next"),
    "credits": ("Credits", "models, data, code and AI tools"),
}


def results_params():
    m = json.loads((ROOT / "eval/results/metrics.json").read_text())
    d = m["splits"]["dermoscopic"]
    pct = lambda v: f"{v * 100:.0f}%"
    return {"n": d["n"], "nm": d["n_malignant"], "nb": d["n_benign"], "tn": d["tn"],
            "auroc": f"{d['auroc']:.2f}", "ci": f"{d['auroc_ci95'][0]:.2f}–{d['auroc_ci95'][1]:.2f}",
            "sens": pct(d["sensitivity"]), "spec": pct(d["specificity"]), "npv": pct(d["npv"]),
            "prev": f"{d['n_malignant'] / d['n'] * 100:.1f}%"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [("skinsight-banner", "kind=hero", 600), ("stats", "kind=stats", 560),
            ("results", "kind=results&m=" + urllib.parse.quote(json.dumps(results_params())), 480)]
    for slug, (title, sub) in SECTIONS.items():
        jobs.append((f"section-{slug}", "kind=section&" + urllib.parse.urlencode({"title": title, "sub": sub}), 300))
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1600, "height": 600}, device_scale_factor=1)
        for name, query, height in jobs:
            page.set_viewport_size({"width": 1600, "height": height})
            page.goto(f"{BASE}?{query}")
            page.wait_for_selector("body[data-ready='1']", timeout=30_000)
            page.wait_for_timeout(200)
            png = OUT / f"{name}.png"
            page.screenshot(path=str(png), clip={"x": 0, "y": 0, "width": 1600, "height": height})
            Image.open(png).save(OUT / f"{name}.webp", quality=90, method=6)
            print("wrote", png.name)
        browser.close()


if __name__ == "__main__":
    main()
