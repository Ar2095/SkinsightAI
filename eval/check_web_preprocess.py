"""Check that the web app's JavaScript resize (web/src/preprocess.js) matches PIL.

The browser must feed the model the same pixels the evaluation did, or the
evaluation numbers would not describe the deployed app. This dumps test images
as raw RGB, resizes them with PIL, and asks Node to do the same with the JS port.

Usage: python eval/check_web_preprocess.py [--n 60]   (needs Node 18+)
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    images = sorted((ROOT / "eval" / "data" / "images").glob("*.jpg"))[: args.n]
    images += sorted((ROOT / "images").glob("crop*.png"))
    with tempfile.TemporaryDirectory() as tmp:
        index = []
        for p in images:
            im = Image.open(p).convert("RGB")
            (Path(tmp) / f"{p.stem}.rgb").write_bytes(im.tobytes())
            (Path(tmp) / f"{p.stem}.224.rgb").write_bytes(im.resize((224, 224), Image.BILINEAR).tobytes())
            index.append({"id": p.stem, "w": im.width, "h": im.height})
        (Path(tmp) / "index.json").write_text(json.dumps(index))
        out = subprocess.run(["node", str(ROOT / "web" / "scripts" / "check-preprocess.mjs"), tmp],
                             capture_output=True, text=True, check=True).stdout
    print(out.strip())


if __name__ == "__main__":
    main()
