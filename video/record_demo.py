"""Record the live demo with Chrome's screencast, paced to the narration.

Usage: python video/record_demo.py [--url http://localhost:4173]
Writes build/demo/frames/*.jpg, build/demo/frames.txt (ffmpeg concat list)
and build/demo/markers.json (when each narration beat starts, in seconds).
"""
import argparse
import base64
import json
import math
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
OUT = HERE / "build" / "demo"
DUR = json.loads((HERE / "build" / "durations.json").read_text())
PHOTOS = {k: str(Path.home() / "Desktop" / f"{k}.jpg") for k in ("benign", "malignant", "real-life-benign")}

CURSOR_JS = """
(() => {
  const add = () => {
    const c = document.createElement('div');
    c.id = '__cursor';
    c.innerHTML = '<svg width="30" height="30" viewBox="0 0 24 24"><path d="M4 2l15 9.5-6.6 1.3 3.9 7.4-2.9 1.5-3.9-7.5L4 19z" fill="#1d0c15" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/></svg>';
    Object.assign(c.style, {position:'fixed', left:'-50px', top:'-50px', zIndex: 2147483647, pointerEvents:'none', filter:'drop-shadow(0 2px 3px rgba(0,0,0,.35))'});
    document.body.appendChild(c);
    document.addEventListener('mousemove', e => { c.style.left = (e.clientX - 4) + 'px'; c.style.top = (e.clientY - 2) + 'px'; }, true);
    document.addEventListener('mousedown', e => {
      const r = document.createElement('div');
      Object.assign(r.style, {position:'fixed', left:(e.clientX-22)+'px', top:(e.clientY-22)+'px', width:'44px', height:'44px', borderRadius:'50%',
        border:'3px solid rgba(224,100,143,.9)', zIndex:2147483646, pointerEvents:'none', transition:'transform .45s ease-out, opacity .45s ease-out'});
      document.body.appendChild(r);
      requestAnimationFrame(() => { r.style.transform = 'scale(1.8)'; r.style.opacity = '0'; });
      setTimeout(() => r.remove(), 500);
    }, true);
  };
  if (document.body) add(); else document.addEventListener('DOMContentLoaded', add);
})();
"""


class Recorder:
    def __init__(self, page):
        self.page = page
        self.cdp = page.context.new_cdp_session(page)
        self.frames = []  # (timestamp, path)
        self.mouse = (720, 400)
        (OUT / "frames").mkdir(parents=True, exist_ok=True)
        self.cdp.on("Page.screencastFrame", self._on_frame)

    def _on_frame(self, ev):
        p = OUT / "frames" / f"{len(self.frames):05d}.jpg"
        p.write_bytes(base64.b64decode(ev["data"]))
        self.frames.append((ev["metadata"]["timestamp"], p))
        self.cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]})

    def start(self):
        self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92, "maxWidth": 1920,
                                               "maxHeight": 1080, "everyNthFrame": 1})
        self.t0 = time.time()

    def stop(self):
        self.cdp.send("Page.stopScreencast")

    def wait(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            # Tiny invisible mouse jiggle keeps the screencast producing frames.
            self.page.wait_for_timeout(min(100, max(1, (end - time.time()) * 1000)))

    def move(self, x, y, dur=0.7):
        x0, y0 = self.mouse
        n = max(2, int(dur * 60))
        for i in range(1, n + 1):
            t = i / n
            e = 0.5 - 0.5 * math.cos(math.pi * t)  # ease in-out
            self.page.mouse.move(x0 + (x - x0) * e, y0 + (y - y0) * e)
            self.page.wait_for_timeout(1000 / 60)
        self.mouse = (x, y)

    def center(self, selector):
        b = self.page.locator(selector).first.bounding_box()
        return b["x"] + b["width"] / 2, b["y"] + b["height"] / 2

    def click(self, selector, dur=0.7):
        self.move(*self.center(selector), dur)
        self.page.wait_for_timeout(120)
        self.page.mouse.down()
        self.page.wait_for_timeout(90)
        self.page.mouse.up()

    def upload(self, path):
        x, y = self.center("#dropzone strong")
        self.move(x, y, 0.8)
        with self.page.expect_file_chooser() as fc:
            self.page.mouse.down(); self.page.wait_for_timeout(90); self.page.mouse.up()
        fc.value.set_files(path)
        self.page.wait_for_selector(".cropper-container", timeout=20_000)

    def scroll(self, dy, dur=1.2):
        n = int(dur * 60)
        for i in range(n):
            self.page.mouse.wheel(0, dy / n)
            self.page.wait_for_timeout(1000 / 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:4173")
    args = ap.parse_args()
    for f in (OUT / "frames").glob("*.jpg"):
        f.unlink()
    markers, results = {}, {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 810}, device_scale_factor=4 / 3)
        page.add_init_script(CURSOR_JS)
        page.goto(args.url)
        page.wait_for_selector("#model-status.ready", timeout=120_000)
        page.evaluate("window.scrollTo(0, 0)")
        r = Recorder(page)
        page.mouse.move(*r.mouse)
        r.start()

        def beat(name, actions, pad=0.35):
            start = time.time()
            markers[name] = start - r.t0
            actions()
            r.wait(max(0.0, start + DUR[name] + pad - time.time()))

        def result_tour():
            page.wait_for_selector("#stage-3:not(.hidden)")
            r.wait(0.9)
            r.move(*r.center("#verdict-label"), 0.8)
            r.wait(1.6)
            r.move(*r.center("#trust"), 0.9)
            r.wait(1.4)
            r.scroll(330, 1.4)
            r.move(*r.center(".subtypes tbody tr"), 0.8)

        # --- open
        def open_site():
            r.wait(0.8)
            r.move(*r.center(".hero h1"), 1.0)
            r.wait(1.2)
            r.move(*r.center(".privacy"), 1.0)
            r.wait(1.4)
            r.move(*r.center("#model-status"), 1.0)
        beat("open", open_site)

        # --- benign
        beat("benign_upload", lambda: (r.scroll(260, 0.8), r.upload(PHOTOS["benign"])), pad=0.1)

        def crop_tour():
            r.wait(0.6)
            r.move(*r.center(".cropper-crop-box"), 0.8)
            r.wait(1.0)
            r.move(*r.center("#preview"), 0.9)
        beat("benign_crop", crop_tour, pad=0.0)
        r.click("#analyze", 0.6)
        page.wait_for_function("window.__skinsightLastRun")
        results["benign"] = page.evaluate("window.__skinsightLastRun")
        beat("benign_result", result_tour, pad=0.6)

        # --- malignant
        def malig_upload():
            page.evaluate("window.__skinsightLastRun = null")
            r.click("#restart", 0.7)
            r.wait(1.0)
            r.upload(PHOTOS["malignant"])
            r.wait(0.8)
            r.move(*r.center(".cropper-crop-box"), 0.7)
            r.wait(0.6)
            r.click("#analyze", 0.7)
        beat("malig_upload", malig_upload, pad=0.0)
        page.wait_for_function("window.__skinsightLastRun")
        results["malignant"] = page.evaluate("window.__skinsightLastRun")
        beat("malig_result", result_tour, pad=0.6)

        # --- friend's mole
        def friend_upload():
            page.evaluate("window.__skinsightLastRun = null")
            r.click("#restart", 0.7)
            r.wait(1.0)
            r.upload(PHOTOS["real-life-benign"])
            r.wait(1.5)
            r.move(*r.center(".cropper-crop-box"), 0.8)
            r.wait(2.0)
            r.move(*r.center("#preview"), 0.8)
        beat("friend_upload", friend_upload, pad=0.0)
        r.click("#analyze", 0.6)
        page.wait_for_function("window.__skinsightLastRun")
        results["friend"] = page.evaluate("window.__skinsightLastRun")

        def friend_result():
            page.wait_for_selector("#stage-3:not(.hidden)")
            r.wait(0.9)
            r.move(*r.center("#verdict-label"), 0.8)
            r.wait(1.5)
            r.move(*r.center("#trust"), 0.9)
        beat("friend_result", friend_result, pad=1.0)
        markers["_end"] = time.time() - r.t0
        r.wait(0.3)
        r.stop()
        browser.close()

    # ffmpeg concat list with per-frame durations
    fr = r.frames
    lines = []
    for i, (ts, p) in enumerate(fr):
        nxt = fr[i + 1][0] if i + 1 < len(fr) else ts + 0.5
        lines += [f"file '{p.resolve()}'", f"duration {max(0.001, nxt - ts):.4f}"]
    lines.append(f"file '{fr[-1][1].resolve()}'")
    (OUT / "frames.txt").write_text("\n".join(lines) + "\n")
    # marker times relative to first frame
    first = fr[0][0]
    offset = first - r.t0
    markers = {k: v - offset for k, v in markers.items()}
    (OUT / "markers.json").write_text(json.dumps({"markers": markers, "results": results}, indent=2))
    print(f"{len(fr)} frames, {fr[-1][0] - first:.1f}s")
    print(json.dumps(markers, indent=1))
    print({k: round(v["pMalignant"], 4) for k, v in results.items()})


if __name__ == "__main__":
    main()
