"""Assemble the Skinsight demo video.

Prereqs: python video/tts.py ; python video/record_demo.py ; demo frames in build/demo/f30
Usage:   python video/build_video.py [--only scene1,scene2] [--preview]
Output:  video/build/skinsight-demo.mp4
"""
import argparse
import json
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
B = HERE / "build"
FPS = 30
ROOT = HERE.parent
LOOPS = Path("/Library/Audio/Apple Loops/Apple/07 Chillwave")

dur = json.loads((B / "durations.json").read_text())
words = json.loads((B / "words.json").read_text())
demo = json.loads((B / "demo" / "markers.json").read_text())
met = json.loads((ROOT / "eval/results/metrics.json").read_text())
d = met["splits"]["dermoscopic"]
rob = met["robustness_dermoscopic"]
M = {"n": d["n"], "nm": d["n_malignant"], "nb": d["n_benign"], "auroc": d["auroc"],
     "ci": f"{d['auroc_ci95'][0]:.2f}–{d['auroc_ci95'][1]:.2f}", "sens": round(d["sensitivity"] * 100),
     "acc": round(d["accuracy"] * 100), "blur8": f"{rob['blur: gaussian r=8']['auroc']:.2f}",
     "dark": f"{min(rob['brightness: x0.5']['auroc'], rob['brightness: x1.5']['auroc']):.2f}",
     "small": f"{rob['zoom_out: lesion fills 30%']['auroc']:.2f}",
     "flip": f"{round(min(v['flip_rate'] for k, v in rob.items() if k.split(':')[0] in ('zoom_in', 'orientation')) * 100)}–"
             f"{round(max(v['flip_rate'] for k, v in rob.items() if k.split(':')[0] in ('zoom_in', 'orientation')) * 100)}"}


def words_at(beat, offset):
    return [{"t": w["t"] + offset, "w": w["w"]} for w in words[beat]]


def seg(scene, beat=None, lead=0.3, tail=0.6, dur_override=None, **params):
    """One scene; narration `beat` starts `lead` seconds in."""
    D = dur_override or (lead + (dur[beat] if beat else 0) + tail)
    audio = [(beat, lead)] if beat else []
    if beat:
        params.setdefault("words", words_at(beat, lead))
    return {"scene": scene, "dur": D, "audio": audio, "params": params}


nframes = len(list((B / "demo" / "f30").glob("*.jpg")))
mk = demo["markers"]
demo_dur = min(nframes / FPS, mk["_end"] + 0.8)

SEGMENTS = [
    seg("title", dur_override=3.6),
    seg("hook", "hook", lead=0.3, tail=0.7),
    seg("survival", "survival", lead=0.2, tail=0.7),
    seg("access", "access", lead=0.2, tail=0.8),
    seg("intro", "intro", lead=0.2, tail=1.0),
    {"scene": "demo", "dur": demo_dur, "params": {"markers": mk, "nframes": nframes},
     "audio": [(k, v) for k, v in mk.items() if not k.startswith("_")]},
    seg("slide", "data", lead=0.4, tail=0.5, src="build/slides/slide6.png"),
    # model: slide 8, then the pipeline graphic from "For this version" onward
]
model_split = next(w["t"] for w in words["model"] if w["w"] == "For") + 0.2 - 0.25
SEGMENTS += [
    {"scene": "slide", "dur": model_split, "audio": [("model", 0.2)],
     "params": {"src": "build/slides/slide8.png"}},
    {"scene": "pipeline", "dur": 0.2 + dur["model"] + 0.7 - model_split, "audio": [],
     "params": {"words": words_at("model", 0.2 - model_split)}},
    seg("slide", "eval_old", lead=0.3, tail=0.5, src="build/slides/slide9.png",
        overlay={"at": 0.3 + next(w["t"] for w in words["eval_old"] if w["w"] == "But"),
                 "text": "That split was random: images of the same lesion could be in both training and testing."}),
    seg("eval_new", "eval_new", lead=0.3, tail=1.0, m=M),
    seg("limits", "eval_limits", lead=0.3, tail=1.2, m=M),
    seg("outro", "outro", lead=0.4, tail=3.2),
]
for s in SEGMENTS:
    if s["scene"] == "slide":
        s["params"]["dur"] = s["dur"]


def render(only=None, scale=1.0):
    (B / "segments").mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=scale)
        page.goto((HERE / "scenes.html").as_uri())
        page.evaluate("document.fonts.ready")
        page.evaluate("preload(['../web/public/img/logo.png','build/slides/slide6.png','build/slides/slide8.png','build/slides/slide9.png'])")
        for i, s in enumerate(SEGMENTS):
            out = B / "segments" / f"{i:02d}-{s['scene']}.mp4"
            if only and s["scene"] not in only:
                continue
            n = round(s["dur"] * FPS)
            ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "image2pipe", "-framerate", str(FPS),
                                   "-c:v", "mjpeg", "-i", "-", "-vf", "scale=1920:1080:flags=lanczos,format=yuv420p",
                                   "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-r", str(FPS), str(out)],
                                  stdin=subprocess.PIPE)
            params = json.dumps(s["params"])
            for f in range(n):
                page.evaluate(f"render({json.dumps(s['scene'])}, {f / FPS}, {params})")
                page.evaluate("Promise.all([...document.images].map(i => i.decode().catch(() => {})))")
                ff.stdin.write(page.screenshot(type="jpeg", quality=93))
            ff.stdin.close()
            ff.wait()
            print(f"rendered {out.name}: {n} frames")
        browser.close()


def assemble():
    segs = sorted((B / "segments").glob("*.mp4"))
    assert len(segs) == len(SEGMENTS), "render all segments first"
    (B / "concat.txt").write_text("".join(f"file '{p.resolve()}'\n" for p in segs))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(B / "concat.txt"),
                    "-c", "copy", str(B / "video_only.mp4")], check=True)

    # narration placement
    t, clips = 0.0, []
    starts = {}
    for s in SEGMENTS:
        starts.setdefault(s["scene"], t)
        for beat, off in s["audio"]:
            clips.append((beat, t + off))
        t += s["dur"]
    total = t
    demo_start = next(st for st, s in zip(_cum(), SEGMENTS) if s["scene"] == "demo")
    outro_start = total - SEGMENTS[-1]["dur"]

    inputs, filters = [], []
    for i, (beat, at) in enumerate(clips):
        inputs += ["-i", str(B / "audio" / f"{beat}.mp3")]
        ms = int(at * 1000)
        filters.append(f"[{i}:a]aresample=48000,adelay={ms}|{ms},apad[n{i}]")
    nn = len(clips)
    filters.append("".join(f"[n{i}]" for i in range(nn)) + f"amix=inputs={nn}:normalize=0:duration=longest,atrim=0:{total:.3f}[voice]")

    # music bed: Forward Progress synth/bass/guitar (9.6 s loops) + Pastel Colors beat (4.8 s), all 100 BPM
    layers = [("Forward Progress Synth.caf", 0.0), ("Forward Progress Bass.caf", 3.6),
              ("Pastel Colors Beat 01.caf", demo_start), ("Forward Progress Guitar.caf", demo_start + 19.2)]
    for j, (f, enter) in enumerate(layers):
        k = nn + j
        inputs += ["-stream_loop", "-1", "-i", str(LOOPS / f)]
        filters.append(f"[{k}:a]aresample=48000,atrim=0:{total:.3f},asetpts=PTS-STARTPTS,"
                       f"volume='if(lt(t,{enter}),0,min(1,(t-{enter})/1.5))':eval=frame[m{j}]")
    filters.append("".join(f"[m{j}]" for j in range(len(layers))) + f"amix=inputs={len(layers)}:normalize=0[mus0]")
    # louder on title + outro, ducked under narration, fade out at the end
    filters.append(f"[mus0]volume='if(lt(t,3.4),0.55,if(gt(t,{outro_start + 4.6:.2f}),0.5,0.14))':eval=frame,"
                   f"afade=t=in:st=0:d=1.2,afade=t=out:st={total - 2.6:.2f}:d=2.6[music]")
    filters.append("[voice][music]amix=inputs=2:normalize=0,loudnorm=I=-15:TP=-1.5:LRA=9[aout]")
    subprocess.run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(filters),
                    "-map", "[aout]", "-c:a", "pcm_s16le", str(B / "audio_mix.wav")], check=True)
    out = B / "skinsight-demo.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(B / "video_only.mp4"), "-i", str(B / "audio_mix.wav"),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                    "-movflags", "+faststart", str(out)], check=True)
    print(f"wrote {out} ({total:.1f}s)")
    (B / "timeline.json").write_text(json.dumps({"total": total, "clips": clips}, indent=1))


def _cum():
    t = 0.0
    for s in SEGMENTS:
        yield t
        t += s["dur"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    if not a.no_render:
        render(set(a.only.split(",")) if a.only else None)
    if not a.only:
        assemble()
