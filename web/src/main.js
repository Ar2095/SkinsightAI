import "cropperjs/dist/cropper.css";
import "./style.css";
import Cropper from "cropperjs";
import * as ort from "onnxruntime-web/wasm";
import { resizePIL, toTensor, SIZE } from "./preprocess.js";
import metrics from "../../eval/results/metrics.json";
import rocUrl from "../../eval/results/roc.png";
import robUrl from "../../eval/results/robustness.png";

const SUBTYPES = [
  ["malignant_soft-tissue-proliferations", "Soft tissue proliferation", true],
  ["malignant_adnexal-epithelial-proliferations", "Adnexal epithelial proliferation (e.g. basal cell carcinoma)", true],
  ["malignant_melanocytic-proliferation", "Melanocytic proliferation (melanoma)", true],
  ["malignant_epidermal-proliferations", "Epidermal proliferation (e.g. squamous cell carcinoma)", true],
  ["benign_flat-melanotic-pigmentations", "Flat melanotic pigmentation (e.g. lentigo)", false],
  ["benign_soft-tissue-proliferations", "Soft tissue proliferation (e.g. dermatofibroma)", false],
  ["benign_inflammatory-or-infectious-diseases", "Inflammatory or infectious condition", false],
  ["benign_adnexal-epithelial-proliferations", "Adnexal epithelial proliferation", false],
  ["benign_melanocytic-proliferations", "Melanocytic proliferation (mole / nevus)", false],
  ["benign_epidermal-proliferations", "Epidermal proliferation (e.g. seborrheic keratosis)", false],
  ["benign_other", "Other", false],
];
const THRESHOLD = metrics.threshold; // 0.5, the rule the evaluation scored
const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ model
ort.env.wasm.numThreads = 1;
let session = null;

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const buf = new Uint8Array(total);
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf.set(value, got);
    got += value.length;
    onProgress(got / total);
  }
  return buf;
}

const modelReady = (async () => {
  const status = $("model-status");
  try {
    const bytes = await fetchWithProgress("/model/skinsight.onnx", (f) => {
      status.textContent = `Loading model… ${Math.round(f * 100)}%`;
    });
    session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    status.textContent = "";
    status.classList.add("ready");
  } catch (err) {
    status.textContent = "The model couldn't load. Please refresh the page.";
    status.classList.add("error");
    throw err;
  }
})();

// ------------------------------------------------------------------ stages
function show(stage) {
  for (const n of [1, 2, 3]) {
    $(`stage-${n}`).classList.toggle("hidden", n !== stage);
    const li = document.querySelector(`.stepper li[data-step="${n}"]`);
    li.classList.toggle("active", n === stage);
    li.classList.toggle("done", n < stage);
  }
  document.querySelector(".stepper").scrollIntoView({ behavior: "smooth", block: "start" });
}

let cropper = null;
let sourceBitmap = null; // full-resolution, EXIF-oriented
let truthLabel = null;

async function loadImage(blob, truth = null) {
  truthLabel = truth;
  sourceBitmap = await createImageBitmap(blob, { imageOrientation: "from-image" });
  const img = $("crop-img");
  if (cropper) cropper.destroy();
  img.src = URL.createObjectURL(blob);
  show(2);
  await img.decode().catch(() => {});
  cropper = new Cropper(img, {
    aspectRatio: 1,
    viewMode: 1,
    autoCropArea: 0.9,
    dragMode: "move",
    background: false,
    responsive: true,
    checkOrientation: true,
    crop: drawPreview,
  });
}

$("file").addEventListener("change", (e) => {
  const f = e.target.files?.[0];
  if (f) loadImage(f);
  e.target.value = "";
});
const dz = $("dropzone");
["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
dz.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files?.[0];
  if (f && f.type.startsWith("image/")) loadImage(f);
});
document.querySelectorAll(".sample").forEach((b) =>
  b.addEventListener("click", async () => {
    const blob = await (await fetch(b.dataset.src)).blob();
    loadImage(blob, b.dataset.truth);
  }),
);
$("back-1").addEventListener("click", () => show(1));
$("back-2").addEventListener("click", () => show(2));
$("restart").addEventListener("click", () => show(1));

// ------------------------------------------------------------------ crop + preprocess
function cropRect() {
  const d = cropper.getData(true); // natural-image pixels (after EXIF rotation)
  const x = Math.max(0, d.x), y = Math.max(0, d.y);
  const w = Math.min(d.width, sourceBitmap.width - x), h = Math.min(d.height, sourceBitmap.height - y);
  return { x, y, w: Math.max(1, w), h: Math.max(1, h) };
}

function croppedPixels() {
  const { x, y, w, h } = cropRect();
  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(sourceBitmap, x, y, w, h, 0, 0, w, h);
  return { data: ctx.getImageData(0, 0, w, h).data, w, h };
}

let previewQueued = false;
function drawPreview() {
  if (previewQueued || !sourceBitmap) return;
  previewQueued = true;
  requestAnimationFrame(() => {
    previewQueued = false;
    const { x, y, w, h } = cropRect();
    const ctx = $("preview").getContext("2d");
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(sourceBitmap, x, y, w, h, 0, 0, SIZE, SIZE);
    ctx.strokeStyle = "rgba(255,255,255,0.75)";
    ctx.lineWidth = 1;
    for (const t of [SIZE / 3, (2 * SIZE) / 3]) {
      ctx.beginPath(); ctx.moveTo(t, 0); ctx.lineTo(t, SIZE); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, t); ctx.lineTo(SIZE, t); ctx.stroke();
    }
  });
}

// ------------------------------------------------------------------ inference
$("analyze").addEventListener("click", async () => {
  const btn = $("analyze");
  btn.disabled = true;
  btn.textContent = session ? "Analyzing…" : "Waiting for model…";
  try {
    await modelReady;
    const { data, w, h } = croppedPixels();
    const rgb224 = resizePIL(data, w, h, SIZE, SIZE, 4); // matches PIL exactly
    const input = new ort.Tensor("float32", toTensor(rgb224), [1, 3, SIZE, SIZE]);
    const out = await session.run({ image: input });
    renderResult(out.p_malignant.data[0], Array.from(out.p_subtype.data), rgb224);
    // Read by web/scripts/e2e_parity.py to check the browser against PyTorch.
    window.__skinsightLastRun = { crop: cropRect(), pMalignant: out.p_malignant.data[0] };
    show(3);
  } catch (err) {
    console.error(err);
    $("model-status").textContent = "Something went wrong running the model. Please try again.";
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze";
  }
});

const pct = (v, d = 0) => `${(v * 100).toFixed(d)}%`;
const derm = metrics.splits.dermoscopic;

function renderResult(pMal, pSub, rgb224) {
  // the exact 224x224 image the model saw
  const img = new ImageData(SIZE, SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    img.data.set([rgb224[i * 3], rgb224[i * 3 + 1], rgb224[i * 3 + 2], 255], i * 4);
  }
  $("result-crop").getContext("2d").putImageData(img, 0, 0);
  $("truth").textContent = truthLabel ? `Biopsy-confirmed diagnosis: ${truthLabel}` : "";
  $("truth").classList.toggle("hidden", !truthLabel);

  const malignant = pMal > THRESHOLD;
  const v = $("verdict");
  v.classList.toggle("malignant", malignant);
  v.classList.toggle("benign", !malignant);
  $("verdict-label").textContent = malignant ? "Looks malignant" : "Looks benign";
  $("p-mal").textContent = pMal.toFixed(3);
  $("meter-fill").style.width = `${Math.max(1.5, pMal * 100)}%`;

  $("advice").textContent = malignant
    ? "This spot shares features with skin cancers. It may be cancerous. Book a dermatologist visit for a skin exam or dermoscopy as soon as you can. Treatment works best when cancer is caught early."
    : "This spot looks most like non-cancerous lesions. Keep watching it for changes. See a dermatologist right away if it changes in size, shape or color, or starts to hurt or bleed. This tool does not replace a professional exam.";

  const bin = metrics.score_bins_dermoscopic.find((b) => pMal >= b.low && (pMal < b.high || b.high === 1));
  const binText = `Of the ${bin.n} test lesions that scored between ${bin.low} and ${bin.high}, <b>${bin.n_malignant} turned out to be malignant</b> (${pct(bin.malignant_rate)}).`;
  $("trust").innerHTML = malignant
    ? `<b>How much to trust this:</b> ${binText} Most flags in our tests were false alarms, but spots scoring this high were far more likely to be cancer than low-scoring ones (${pct(metrics.score_bins_dermoscopic[0].malignant_rate, 1)} under 0.1). A doctor visit is the safe next step.`
    : `<b>How much to trust this:</b> ${binText} Skinsight still <b>missed ${pct(1 - derm.sensitivity)} of cancers</b> in our tests, so a benign result does not rule cancer out.`;

  const rows = SUBTYPES.map(([key, name, mal], i) => ({ key, name, mal, p: pSub[i] }))
    .filter((r) => r.p > 0.01)
    .sort((a, b) => b.p - a.p);
  $("subtype-rows").innerHTML = rows
    .map((r) => {
      const level = r.p > 0.9 ? "strong" : r.p > 0.5 ? "mid" : "";
      return `<tr class="${r.mal ? "mal" : "ben"} ${level}">
        <td><span class="tag">${r.mal ? "Malignant" : "Benign"}</span>${r.name}</td>
        <td><div class="bar"><i style="width:${(r.p * 100).toFixed(1)}%"></i></div><span class="num">${pct(r.p, 1)}</span></td>
      </tr>`;
    })
    .join("");
}

// ------------------------------------------------------------------ evaluation section
function renderEvaluation() {
  $("n-test").textContent = derm.n;
  const [lo, hi] = derm.auroc_ci95;
  const tiles = [
    [derm.auroc.toFixed(2), "AUROC", `95% CI ${lo.toFixed(2)}–${hi.toFixed(2)}. 0.5 is a coin flip, 1.0 is perfect.`],
    [pct(derm.sensitivity), "of cancers caught", `${derm.tp} of ${derm.n_malignant} malignant lesions. Dermatologists using dermoscopy catch ~${pct(metrics.dermatologist_reference.sensitivity)}.`],
    [pct(derm.specificity), "of benign spots cleared", `${derm.tn} of ${derm.n_benign} benign lesions correctly called benign.`],
    [pct(derm.npv), "of “benign” calls correct", `At this test set's ${pct(derm.n_malignant / derm.n, 1)} cancer rate. Varies with who is using it.`],
  ];
  $("stats").innerHTML = tiles
    .map(([n, l, s]) => `<div class="stat"><div class="stat-n">${n}</div><div class="stat-l">${l}</div><div class="stat-s">${s}</div></div>`)
    .join("");
  $("fig-roc").src = rocUrl;
  $("fig-rob").src = robUrl;
}
renderEvaluation();
