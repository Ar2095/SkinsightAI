"""Split a single recorded voiceover into per-beat clips (replaces tts.py output).

Prereq: whisper.cpp word timestamps for the voiceover:
  ffmpeg -i voiceover.mp3 -ar 16000 -ac 1 build/vo/vo16k.wav
  whisper-cli -ng -m ggml-base.en.bin -f build/vo/vo16k.wav -ml 1 -sow -ojf -of build/vo/vo
Usage: python video/vo_split.py path/to/voiceover.mp3
Writes build/audio/<beat>.mp3, build/durations.json, build/words.json (same format as tts.py).
"""
import difflib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
B = HERE / "build"
SRC = Path(sys.argv[1])

beats = json.loads((HERE / "script.json").read_text())["beats"]
NUM = {"70": "seventy", "99": "ninetynine", "36": "thirtysix", "18": "eighteen", "6": "six", "97": "ninetyseven",
       "1": "one", "11": "eleven", "90": "ninety", "598": "fivehundredninetyeight", "87": "eightyseven",
       "78000": "seventyeightthousand", "072": "zeropointseventwo", "5": "five"}


def norm(w):
    w = re.sub(r"[^a-z0-9]", "", w.lower().replace("%", ""))
    return NUM.get(w, w)


# whisper words (ms)
segs = json.loads((B / "vo" / "vo.json").read_text())["transcription"]
hw = [(norm(s["text"]), s["offsets"]["from"] / 1000, s["offsets"]["to"] / 1000) for s in segs if norm(s["text"])]

# script tokens, tagged with beat
st = []
for b in beats:
    for tok in b["text"].split():
        if norm(tok):
            st.append((b["id"], tok.strip(".,:;!?"), norm(tok)))

sm = difflib.SequenceMatcher(a=[x[2] for x in st], b=[x[0] for x in hw], autojunk=False)
t0 = [None] * len(st)
t1 = [None] * len(st)
for a, bb, n in sm.get_matching_blocks():
    for k in range(n):
        t0[a + k], t1[a + k] = hw[bb + k][1], hw[bb + k][2]
matched = sum(x is not None for x in t0)
print(f"matched {matched}/{len(st)} script words")
# interpolate gaps
known = [i for i, x in enumerate(t0) if x is not None]
for i in range(len(st)):
    if t0[i] is None:
        lo = max([k for k in known if k < i], default=None)
        hi = min([k for k in known if k > i], default=None)
        if lo is None:
            t0[i] = t1[i] = t0[hi]
        elif hi is None:
            t0[i] = t1[i] = t1[lo]
        else:
            f = (i - lo) / (hi - lo)
            t0[i] = t1[i] = t1[lo] + f * (t0[hi] - t1[lo])

# silences for precise cut points
sil = subprocess.run(["ffmpeg", "-v", "info", "-i", str(SRC), "-af", "silencedetect=noise=-40dB:d=0.12", "-f", "null", "-"],
                     capture_output=True, text=True).stderr
ss = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", sil)]
se = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", sil)]
silences = list(zip(ss, se))


import wave
import numpy as np

with wave.open(str(B / "vo" / "vo16k.wav")) as wf:
    pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)
RMS = np.sqrt(np.convolve(pcm ** 2, np.ones(160) / 160, mode="same"))  # 10 ms window @16k


def boundary(t_end, t_next):
    """(end of clip k, start of clip k+1) around the gap between two beats."""
    c = [(s, e) for s, e in silences if s < t_next + 0.3 and e > t_end - 0.3]
    if c:
        s, e = max(c, key=lambda x: x[1] - x[0])
        return s + min(0.12, (e - s) / 2), e - min(0.06, (e - s) / 2)
    lo, hi = int((min(t_end, t_next) - 0.12) * 16000), int((max(t_end, t_next) + 0.12) * 16000)
    cut = (lo + int(np.argmin(RMS[lo:hi]))) / 16000
    return cut, cut


cuts = {}
ids = [b["id"] for b in beats]
for k, b in enumerate(ids):
    idx = [i for i, x in enumerate(st) if x[0] == b]
    cuts[b] = [t0[idx[0]], t1[idx[-1]]]
first_start = max(0.0, cuts[ids[0]][0] - 0.1)
for k in range(len(ids) - 1):
    e, s2 = boundary(cuts[ids[k]][1], cuts[ids[k + 1]][0])
    cuts[ids[k]][1], cuts[ids[k + 1]][0] = e, s2
cuts[ids[0]][0] = first_start
FILE_END = len(pcm) / 16000
last = [s for s, e in silences if s > cuts[ids[-1]][1] - 0.6]
cuts[ids[-1]][1] = min(FILE_END, (last[0] + 0.15) if last else cuts[ids[-1]][1] + 0.2)

(B / "audio").mkdir(parents=True, exist_ok=True)
durations, words = {}, {}
for b in beats:
    idx = [i for i, x in enumerate(st) if x[0] == b["id"]]
    start, end = cuts[b["id"]]
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(SRC),
                    "-af", "afade=t=in:d=0.02,areverse,afade=t=in:d=0.04,areverse", "-c:a", "libmp3lame", "-b:a", "192k",
                    str(B / "audio" / f"{b['id']}.mp3")], check=True)
    durations[b["id"]] = round(end - start, 3)
    words[b["id"]] = [{"t": round(max(0.0, t0[i] - start), 3), "w": st[i][1]} for i in idx]
    print(f"{b['id']:15s} {start:7.2f}-{end:7.2f}  {end - start:5.2f}s")
(B / "durations.json").write_text(json.dumps(durations, indent=2))
(B / "words.json").write_text(json.dumps(words, indent=1))
print(f"total speech {sum(durations.values()):.1f}s")
