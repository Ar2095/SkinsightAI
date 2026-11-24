// Called by eval/check_web_preprocess.py: reads raw RGB images + PIL's 224x224 output
// and reports how many bytes the JS resampler gets different.
import { readFileSync } from "node:fs";
import { resizePIL, SIZE } from "../src/preprocess.js";

const dir = process.argv[2];
const index = JSON.parse(readFileSync(`${dir}/index.json`, "utf8"));
let diffBytes = 0, maxDiff = 0, total = 0;
for (const { id, w, h } of index) {
  const raw = new Uint8Array(readFileSync(`${dir}/${id}.rgb`));
  const want = new Uint8Array(readFileSync(`${dir}/${id}.224.rgb`));
  const got = resizePIL(raw, w, h, SIZE, SIZE, 3);
  for (let i = 0; i < want.length; i++) {
    const d = Math.abs(got[i] - want[i]);
    if (d) diffBytes++;
    if (d > maxDiff) maxDiff = d;
  }
  total += want.length;
}
console.log(JSON.stringify({ images: index.length, bytes: total, differing_bytes: diffBytes, max_abs_diff: maxDiff }));
