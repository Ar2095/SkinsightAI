"""Synthesize one narration clip per beat in script.json.

Writes build/audio/<id>.mp3, build/durations.json and build/words.json
(word start times in seconds per beat, used to sync animations to speech).
"""
import asyncio
import json
import subprocess
from pathlib import Path

import edge_tts

HERE = Path(__file__).parent
OUT = HERE / "build" / "audio"


async def synth(text, voice, rate, path):
    words = []
    with open(path, "wb") as f:
        comm = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
        async for msg in comm.stream():
            if msg["type"] == "audio":
                f.write(msg["data"])
            elif msg["type"] == "WordBoundary":
                words.append({"t": msg["offset"] / 1e7, "w": msg["text"]})
    return words


async def main():
    s = json.loads((HERE / "script.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    durations, words = {}, {}
    for b in s["beats"]:
        path = OUT / f"{b['id']}.mp3"
        words[b["id"]] = await synth(b["text"], s["voice"], s.get("rate", "+0%"), path)
        d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                  str(path)], capture_output=True, text=True).stdout)
        durations[b["id"]] = d
        print(f"{b['id']:15s} {d:5.2f}s  {len(words[b['id']])} words")
    (HERE / "build" / "durations.json").write_text(json.dumps(durations, indent=2))
    (HERE / "build" / "words.json").write_text(json.dumps(words, indent=1))
    print(f"total narration {sum(durations.values()):.1f}s")


asyncio.run(main())
