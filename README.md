![Skinsight](https://skinsight-weld.vercel.app/devpost/skinsight-banner.webp)

# Skinsight: AI-Powered Dermatology Insights

### **Try it (no install):** https://skinsight-weld.vercel.app
### **Demo video:** https://youtu.be/WciqO6wmbLM
### **Code:** https://github.com/Ar2095/SkinsightAI

[![Watch the Skinsight demo video](https://img.youtube.com/vi/WciqO6wmbLM/maxresdefault.jpg)](https://youtu.be/WciqO6wmbLM)

Skin cancer is the most common cancer in the United States, and **when it is caught early, it is almost always curable**: melanoma found while still localized has a five-year survival rate above 99%. The hard part is getting a suspicious spot looked at in time.

**Skinsight is a deep-learning model that gives anyone a first look at a skin spot in under a minute.** Upload a photo, line up the spot, and Skinsight estimates whether it looks **benign or malignant** and which of **11 lesion types** it most resembles, then tells you what to do next and how much to trust the answer.

- **The model:** a ResNet-18 convolutional neural network, pretrained on ImageNet and fine-tuned with **two output heads** (benign vs. malignant, and lesion type) on about **78,000 labeled dermoscopic images** from the ISIC Archive.
- **Private by design:** the model is exported to ONNX and runs **entirely in your browser**. Your photo is never uploaded anywhere, and there's no server to pay for.
- **Tested honestly:** it was evaluated on **598 lesions added to the ISIC Archive after the model was trained**, with confidence intervals, baselines, subgroups and stress tests. Every result in the app quotes how often a similar score was actually cancer in that test.

### For judges: where to find each criterion

| Criterion | Where |
|---|---|
| **Technical Depth** | **How It Was Built**: dual-head fine-tuned CNN, ONNX export, byte-exact browser preprocessing, three parity checks |
| **Rigor & Reproducibility** | **Evaluation**: temporal hold-out, metric in clinical units (per 1,000 lesions), bootstrap CIs, baselines, seeds and config; **Run It Yourself**: one command per step |
| **Clinical Relevance & Honesty** | **The Problem**, **Who Benefits, and Who Could Be Harmed**, **Limitations** |
| **Demo Quality** | [Demo video](https://youtu.be/WciqO6wmbLM) (working app on screen in the first minute), [live site](https://skinsight-weld.vercel.app) with built-in sample images, nothing to install |

---

![The Problem](https://skinsight-weld.vercel.app/devpost/section-problem.webp)

## The Problem

At least **1 in 5 Americans** will develop skin cancer by age 70. Catching it early is what saves lives: melanoma caught while still localized has a five-year survival rate above **99%**. Once it has spread to distant organs, survival drops to **36%**. Early detection is hard in practice, though:

- **Long waits.** A new dermatology appointment takes **36.5 days** on average in the US, and some cities wait months.
- **No dermatologist nearby.** People in rural areas often have none within reach.
- **Time and money.** Many people skip a visit for a spot they aren't sure about.

![1 in 5 Americans develop skin cancer by 70; 36.5-day average dermatology wait; >99% melanoma survival when caught early vs 36% once it spreads](https://skinsight-weld.vercel.app/devpost/stats.webp)

**Who has this problem:** anyone looking at a mole and wondering whether it's worth an appointment, especially people who can't easily get to a dermatologist. Today their options are a printed ABCDE checklist (subjective), teledermatology (costly and slow), or consumer apps that often charge per scan.

---

![What It Does](https://skinsight-weld.vercel.app/devpost/section-what-it-does.webp)

## What It Does

**Try it in 30 seconds:** open https://skinsight-weld.vercel.app and click one of the sample images, or upload your own photo. Nothing to install or sign up for.

1. **Upload** a close-up photo of a skin spot, or pick one of the built-in ISIC samples (including one cancer that Skinsight misses, so you can see a failure too).
2. **Crop** it with a square box. A live preview shows exactly the 224×224 image the model will see, next to example crops.
3. **Get a result:** benign or malignant with the probability, the most likely lesion types, next steps in plain language, and a **"how much to trust this"** note quoting how often lesions with a similar score were actually cancer in our evaluation.

| Upload | Crop | Result |
|---|---|---|
| ![Upload step](https://skinsight-weld.vercel.app/devpost/screens/1-home.webp) | ![Crop step](https://skinsight-weld.vercel.app/devpost/screens/2-benign-crop.webp) | ![Benign result](https://skinsight-weld.vercel.app/devpost/screens/2-benign-result.webp) |

| Melanoma flagged | Real phone photo | Accuracy, built into the site |
|---|---|---|
| ![Malignant result](https://skinsight-weld.vercel.app/devpost/screens/3-melanoma-result.webp) | ![Phone photo result](https://skinsight-weld.vercel.app/devpost/screens/4-phone-photo-result.webp) | ![Accuracy section](https://skinsight-weld.vercel.app/devpost/screens/5-accuracy.webp) |

The page also has a **How accurate is it?** section. Its numbers are imported directly from `eval/results/metrics.json` at build time, so the site can never drift from the evaluation.

---

![How It Was Built](https://skinsight-weld.vercel.app/devpost/section-how-it-was-built.webp)

## How It Was Built

```
ISIC Archive (78k labeled images)
   └─> fine-tune ResNet-18, two heads (binary + 11 subtypes)          skin-imaging.py
         └─> checkpoint dual_head_skin_model.pth
               ├─> temporal hold-out evaluation (598 new lesions)      eval/fetch_holdout.py, eval/evaluate.py
               └─> ONNX export (sigmoid + softmax in the graph)        export_onnx.py
                     └─> browser: PIL-exact resize -> onnxruntime-web (WebAssembly)   web/
                           └─> result + calibrated "how much to trust this" from the evaluation
```

### Data
The model was trained on images from the [ISIC Archive](https://www.isic-archive.com) (downloaded May 2025), about 78,000 labeled dermoscopic images, grouped into binary (benign/malignant) and 11 subtype folders taken from ISIC's diagnosis hierarchy (`diagnosis_1` / `diagnosis_2`), for example `malignant_melanocytic-proliferation` (melanoma) and `benign_epidermal-proliferations` (e.g. seborrheic keratosis).

### Model
[`skin-imaging.py`](skin-imaging.py) holds the training code; [`skinsight_model.py`](skinsight_model.py) is the shared model definition used by evaluation and export.

- **Backbone:** torchvision ResNet-18, pretrained on ImageNet, with its classifier removed
- **Binary head:** Linear(512→128) → ReLU → Dropout(0.3) → Linear(128→1), trained with `BCEWithLogitsLoss`
- **Subtype head:** the same shape ending in 11 outputs, trained with `CrossEntropyLoss`. The two losses are summed, so the lesion-type task acts as extra supervision for the binary decision.
- **Input:** 224×224, ImageNet normalization, random horizontal flip and ±10° rotation during training
- **Optimizer:** Adam (lr 1e-4), batch size 64, 10 epochs, random 80/20 train/validation split

The trained checkpoint is [`dual_head_skin_model.pth`](dual_head_skin_model.pth).

### App
A static Vite site in [`web/`](web/), hosted on Vercel, with no backend:

- [`export_onnx.py`](export_onnx.py) converts the checkpoint to ONNX, with the sigmoid and softmax built into the graph.
- The browser runs it with **onnxruntime-web** (WebAssembly, self-hosted).
- Cropping uses Cropper.js at the photo's full resolution.
- An earlier Streamlit interface ([`skin_classifier.py`](skin_classifier.py)) is also included.

**The deployed app is checked against the evaluated model.** The browser's canvas resize uses a different filter from the Python pipeline, which would make the app's numbers differ from the evaluation. So PIL's antialiased bilinear resampler was ported to JavaScript ([`web/src/preprocess.js`](web/src/preprocess.js)), and it reproduces PIL byte-for-byte. Each layer is verified:

| Check | Script | Result |
|---|---|---|
| ONNX vs. PyTorch, all 749 test images | `export_onnx.py` | max \|ΔP\| 5e-6, 0 decisions changed |
| JS resize vs. PIL, 43 images | `eval/check_web_preprocess.py` | 0 differing bytes |
| Live site vs. PyTorch, same crop, through the real UI | `web/scripts/e2e_parity.py` | max \|ΔP\| 3e-7 |

So the evaluation below describes exactly what a user of the website gets.

---

![Evaluation](https://skinsight-weld.vercel.app/devpost/section-evaluation.webp)

## Evaluation

### What was measured
**The task:** given a dermoscopic image of a lesion, say whether it is malignant.

**Primary metric:** AUROC, plus **sensitivity** (share of cancers caught) and **specificity** (share of benign lesions correctly cleared) at the app's own decision rule, P(malignant) > 0.5. They are reported in clinical units below: **cancers caught and missed, and unnecessary referrals, per 1,000 lesions checked.** PPV/NPV, Brier score and calibration are also reported. 95% confidence intervals come from 2,000 stratified bootstrap resamples. The subtype head is scored with top-1 and top-3 accuracy.

### How it was measured: a temporal hold-out
A random split can put images of the same lesion or patient on both sides, which inflates results. The training-time validation split was a random 80/20 split with no fixed seed, so it isn't relied on here. Instead, [`eval/fetch_holdout.py`](eval/fetch_holdout.py) builds a test set the model **cannot** have seen:

- **New images only.** It queries the public ISIC API for every image *created after 2025-06-01*. The model checkpoint was committed on 2025-05-20, so none of these can be in its training data.
- **Labeled lesions only.** It keeps images with a Benign or Malignant diagnosis.
- **One image per lesion.** It keeps one image per `lesion_id`, since some lesions are photographed repeatedly.
- **A capped benign sample.** The MEL-SELF trial contributes about 3,000 benign images, so it keeps a seeded random sample of 400.

That produces three test splits:

| Split | What it is | Malignant | Benign |
|---|---|---|---|
| **dermoscopic** (primary) | same imaging type as training | 41 | 557 |
| **rcm** | reflectance confocal microscopy tiles: a modality the model never saw | 107 | 26 |
| **clinical** | non-dermoscopic close-up photos | 11 | 7 |

The image IDs and labels are committed ([`eval/data/manifest.csv`](eval/data/manifest.csv)), so anyone can rebuild the exact set.

### Results

![Tested on 598 lesions it had never seen: AUROC 0.72, 49% of cancers caught, 73% of benign cleared, 95% of benign calls correct](https://skinsight-weld.vercel.app/devpost/results.webp)

| Split | n (mal / ben) | AUROC [95% CI] | Sensitivity [95% CI] | Specificity [95% CI] | PPV | NPV |
|---|---|---|---|---|---|---|
| **dermoscopic** | 598 (41 / 557) | **0.721** [0.643–0.799] | **48.8%** [34.1–63.4] | **73.4%** [69.5–77.0] | 11.9% | 95.1% |
| rcm (out of distribution) | 133 (107 / 26) | 0.514 [0.486–0.542] | 100% | 0% | 80.5% | – |
| clinical close-ups | 18 (11 / 7) | 0.688 [0.390–0.922] | 54.5% | 71.4% | 75.0% | 50.0% |

### In clinical units: per 1,000 lesions checked
Scaled from the dermoscopic hold-out (41 cancers in 598 lesions, 6.9% prevalence; raw counts TP 20, FN 21, FP 148, TN 409):

| Per 1,000 lesions | Skinsight (threshold 0.5) | Dermatologists, dermoscopy (reference*) | Refer everyone |
|---|---|---|---|
| Cancers present | 68.6 | 68.6 | 68.6 |
| **Cancers caught** | **33.4** | 59.4 | 68.6 |
| **Cancers missed** | **35.1** | 9.2 | 0 |
| **Benign lesions sent to a doctor unnecessarily** | **247** | 267 | 931 |
| Benign lesions correctly reassured | 684 | 664 | 0 |
| **Referrals per cancer found** | **8.4** | 5.5 | 14.6 |

\* Sensitivity 86.6% / specificity 71.3% for 58 dermatologists from [Haenssle et al., *Ann Oncol* 2018](https://doi.org/10.1093/annonc/mdy166), applied to this test set's prevalence. Different images, so this is context, not a head-to-head.

**The cost that matters most is the missed cancer.** At its default setting, Skinsight would miss about 35 of every 69 cancers, while sending about as many benign lesions to a doctor as a dermatologist would. That's why it is framed as an educational first look, not a screening test, and why every benign result says so.

### Baselines
(dermoscopic split)

| Method | Sensitivity | Specificity | AUROC |
|---|---|---|---|
| Skinsight, binary head, threshold 0.5 | 48.8% | 73.4% | 0.721 |
| Skinsight, subtype head (sum of the 4 malignant classes) | – | – | 0.722 |
| Always say "benign" | 0% | 100% | 0.5 |
| Refer everyone | 100% | 0% | 0.5 |
| 58 dermatologists reading dermoscopy images (Haenssle 2018; different images) | 86.6% | 71.3% | – |
| Training-time random validation split (no seed; possible same-lesion leakage) | ~78% | ~94% | – |

### How often a score was actually cancer
The app shows this next to each result, instead of the raw probability:

| P(malignant) | Lesions | Actually malignant |
|---|---|---|
| 0.0–0.1 | 242 | 5 (2.1%) |
| 0.1–0.5 | 188 | 16 (8.5%) |
| 0.5–0.9 | 109 | 8 (7.3%) |
| 0.9–1.0 | 59 | 12 (20.3%) |

**Subtype head** (dermoscopic, n = 198 with a fine-grained label; only 5 of the 11 classes appear in the new data): top-1 accuracy is 46.5%, below the 75.8% majority-class baseline, while top-3 accuracy is 85.9%. In the confusion matrix, every class's most common prediction is itself.

<p align="center"><img src="eval/results/roc.png" width="48%" alt="ROC curves" /> <img src="eval/results/robustness.png" width="48%" alt="Robustness to image degradations" /></p>

### What the results mean
- **Skinsight ranks lesions clearly better than chance (AUROC 0.72), but it is not a screening tool.** At its default cut-off it catches about half of cancers (48.8% sensitivity). Dermatologists using dermoscopy catch about 87%.
- **A low score is meaningful.** Lesions scoring under 0.1 were cancer 2.1% of the time, versus 20.3% for scores above 0.9, a tenfold difference. But a "benign" result does not rule cancer out, and the app says so.
- **The 95% NPV mostly reflects how rare cancer is.** At 6.9% prevalence, even "always benign" would be right 93% of the time.
- **Moving the cut-off doesn't rescue it.** Thresholds chosen on the test set itself (so these are optimistic) give 57.8% specificity at 80% sensitivity, and 39.0% specificity at 90% sensitivity.
- **The probabilities are not calibrated** (ECE 0.25, Brier 0.21). Benign lesions average P(malignant) = 0.30. That's why the app reports the observed rate for each score range.
- **Individual answers are unstable near the cut-off.** Mirroring the image changes 15.0% of benign/malignant decisions, rotating it 90° changes 23.4%, and cropping to the central 75% changes 21.6%, even though AUROC barely moves. The melanoma photo in the demo video is called malignant (0.97) with the app's default crop, but benign (0.01) when cropped tightly around the lesion.
- **Random splits flatter models.** The training-time random split reported 90.4% binary accuracy. On genuinely new lesions, accuracy is 71.7%, below the 93% you'd get from always saying "benign". The temporal hold-out is the number to trust.

### Where it stops being useful
Measured with [`eval/evaluate.py`](eval/evaluate.py) by degrading the dermoscopic split:

- **Not a skin-surface photo.** On confocal microscopy tiles the model outputs about 0.58 for nearly every image and calls all of them malignant (AUROC 0.51).
- **Blur:** AUROC holds up to a Gaussian radius of 2 px (at 224 px), drops to 0.65 at 4 px, and reaches chance (0.50) at 8 px.
- **Lighting:** ±50% brightness drops AUROC to 0.63–0.64.
- **Framing:** when the lesion fills only 50% or 30% of the crop, AUROC falls to 0.62 and 0.57. Cropping in tighter *helps*: the central 75% scores 0.77. That's why the app makes you crop.
- **Heavy compression:** JPEG quality 5 drops AUROC to 0.58. Quality 10–50 has no effect.
- **Darker skin is untested.** Only 234 test images carry a Fitzpatrick label: 220 are type I, and none are type IV–VI.

**Subgroups** (dermoscopic; small groups have wide intervals; full table in [`eval/results/results.md`](eval/results/results.md)):

| Subgroup | Malignant / Benign | AUROC |
|---|---|---|
| Biopsy-confirmed labels | 30 / 151 | 0.75 |
| Not biopsy-confirmed | 11 / 406 | 0.68 |
| Male | 17 / 257 | 0.83 |
| Female | 24 / 300 | 0.65 |
| Head and neck | 24 / 94 | 0.60 |
| Trunk | 5 / 196 | 0.78 |

**Compute cost:** 10 ms per image on a laptop CPU (PyTorch, batch of 1). The web app has no server, so serving costs nothing beyond static hosting.

### Seeds and configuration

| Setting | Value | Where |
|---|---|---|
| Hold-out cutoff | images created after 2025-06-01 | `eval/fetch_holdout.py` |
| One image per lesion; MEL-SELF benign cap | 400, `--seed 0` | `eval/fetch_holdout.py` |
| Decision threshold | P(malignant) > 0.5 (the app's rule) | `eval/evaluate.py` |
| Bootstrap | 2,000 stratified resamples, `numpy` seed 0, `torch.manual_seed(0)` | `eval/evaluate.py` |
| Preprocessing | resize to 224×224 (PIL bilinear), ImageNet mean/std | `skinsight_model.py` |
| Score bins | 0, 0.1, 0.5, 0.9, 1.0 | `eval/evaluate.py` |
| Dependencies | pinned versions | `requirements.txt` |

**If your numbers differ from these,** it will be for one of these reasons:
- **You passed `--refresh`.** That re-queries the ISIC API, and the archive keeps growing, so the test set changes. Without it, the committed metadata reproduces the exact 749 images used here.
- **Different hardware or BLAS.** Probabilities can move by about 1e-6. In the parity checks no decision changed.
- **You retrained.** The training-time split had no seed, so retraining gives a different checkpoint. Every number here is for the committed `dual_head_skin_model.pth`.

---

![Who Benefits](https://skinsight-weld.vercel.app/devpost/section-benefits-harms.webp)

## Who Benefits, and Who Could Be Harmed

**Who benefits**
- **People who can't easily see a dermatologist.** They get a free, private, immediate first look, a reminder of the warning signs (ABCDE), and a clear prompt to book a visit when a spot is flagged or changing.
- **People unsure whether a spot is worth an appointment.** A high score is a concrete nudge to act: in testing, spots scoring above 0.9 were ten times more likely to be cancer than spots scoring under 0.1.
- **Students and the public,** who can see a clear example of what a dermatology model can and can't do, with its evaluation published alongside it.

**Who could be harmed, and how the app addresses it**
- **People falsely reassured.** This is the biggest risk: Skinsight misses about half of cancers. A benign result never says "safe". The app shows the miss rate on every benign result and tells people to see a doctor about changing spots.
- **People sent to the doctor unnecessarily.** About 88% of malignant flags in the test set were false alarms. That causes anxiety and uses appointments. The result card quotes the real rate.
- **People with darker skin.** Performance could not be measured for Fitzpatrick IV–VI at all, and dermatology datasets are known to under-represent them. The app lists this under "Don't trust it when".
- **Privacy.** Skin photos are sensitive. Inference runs on-device, and no image is uploaded or stored.
- **Misuse on the wrong images.** On non-dermoscopic or microscopy images, the output is meaningless. The app gives photo guidance and states the known failure modes.

---

![Limitations](https://skinsight-weld.vercel.app/devpost/section-limitations.webp)

## Limitations

- **Few positives.** Only 41 malignant lesions in the primary split, so sensitivity's 95% CI spans 34–63%.
- **Label quality varies.** 406 of the 557 benign labels are not biopsy-confirmed. Most are expert consensus on a single image, from the MEL-SELF trial.
- **Dermoscopic test, phone-photo use.** The test images are dermoscopic, but people will upload phone photos. The only clinical-photo split has 18 images.
- **Same archive.** The temporal split rules out duplicate *images*. But it is still the ISIC Archive, and a patient photographed before June 2025 could reappear with a new image of a different lesion.
- **Subtype coverage.** Only 5 of the 11 subtype classes appear in the post-cutoff data, so 6 classes are untested.
- **Crop and orientation sensitivity.** A single result can flip with how the photo is framed or mirrored (15–34% of decisions in testing).
- **Untuned threshold.** The 0.5 cut-off was never tuned for sensitivity, and the scores are uncalibrated.
- **Indirect dermatologist comparison.** The dermatologist numbers come from a published study on different images.
- **Unrecoverable training split.** The training-time validation split used no seed, so its 90% figure can't be reproduced or audited. The temporal hold-out is reported instead.

---

![Challenges](https://skinsight-weld.vercel.app/devpost/section-challenges.webp)

## Challenges

- **Finding a dataset.** Many available skin datasets were incomplete, synthetic, or too small. The ISIC Archive's diagnosis hierarchy provided 11 consistent classes.
- **Choosing a model.** Several architectures and ensembles were tried (`ensemble_predictions.py` in the history). ResNet-18 was kept because the alternatives were less accurate.
- **Finding a fair test set.** The training split had no seed, so it couldn't be recovered. The fix came from ISIC's API, which pages images in creation order, so it can be queried for only the images added after the model existed.
- **Matching the browser to Python exactly.** Canvas resizing gave slightly different pixels, so PIL's fixed-point resampler was ported to JavaScript and tested byte-for-byte.

---

![Lessons Learned](https://skinsight-weld.vercel.app/devpost/section-lessons-learned.webp)

## Lessons Learned

- **A random split can flatter a model badly.** This one did: 90% accuracy on a random split, versus 72% accuracy and 49% sensitivity on new lesions.
- **Accuracy is the wrong headline when 93% of lesions are benign.** Sensitivity, specificity and AUROC, in cancers caught and missed, tell the real story.
- **A probability isn't a risk until it's calibrated.** A score of 0.97 here meant "about 1 in 5 chance of cancer".
- **Shipping the model is part of the evaluation.** If the deployed preprocessing differs from the evaluated one, the numbers no longer describe the product.

---

![Improvements](https://skinsight-weld.vercel.app/devpost/section-improvements.webp)

## Improvements

1. **Retrain properly:**
   - patient-level train/validation/test splits with a fixed seed
   - class weighting for the rare malignant classes
   - phone-photo-style augmentation (blur, lighting, framing)
2. **Stabilize single answers.** Average predictions over flips, rotations and several crops (test-time augmentation), and re-measure the decision-flip rate.
3. **Calibrate and choose the threshold for sensitivity.** Apply temperature scaling on a validation set, then pick the cut-off for about 90% sensitivity on validation, not test.
4. **Test on phone photos and diverse skin.** Evaluate externally on PAD-UFES-20 (smartphone images, excluded from training) and on skin-tone-labelled sets such as DDI and Fitzpatrick17k, reporting results by Fitzpatrick type.
5. **Add a reject option.** Detect out-of-distribution or low-quality photos and decline to answer instead of guessing.
6. **Run a small reader study.** Compare Skinsight against non-expert users with the ABCDE checklist on the same images, since that is the real alternative for Skinsight's users.

---

![Credits](https://skinsight-weld.vercel.app/devpost/section-credits.webp)

## Credits

**Built by:** [vlww](https://github.com/vlww), a student.

**Data**
- [ISIC Archive](https://www.isic-archive.com), International Skin Imaging Collaboration. Training images (May 2025) and test images (created after 2025-06-01) are each under their own CC-0 or CC-BY license. All test images are CC-BY, contributed by:
  - Memorial Sloan Kettering Cancer Center
  - the [MEL-SELF trial](https://www.sydney.edu.au/medicine-health/our-research/research-centres/melself-project.html), University of Sydney
  - Dr. Cristian Navarrete et al.
- The in-app sample images are ISIC_6537724, ISIC_3099700, ISIC_5423259 and ISIC_7168023 (CC-BY, from MSKCC and Navarrete et al.).
- Demo video: the benign mole and melanoma photos come from public skin-lesion datasets outside the ISIC Archive, so the model never trained on anything from those sources. The third photo is a friend's mole, taken on a phone.

**Third-party models and code**
- [ResNet-18](https://arxiv.org/abs/1512.03385) (He et al., 2016), with ImageNet weights from [torchvision](https://pytorch.org/vision/)
- [PyTorch](https://pytorch.org), [scikit-learn](https://scikit-learn.org), [pandas](https://pandas.pydata.org), [Matplotlib](https://matplotlib.org), [Pillow](https://python-pillow.org)
- Pillow's bilinear resampler (`libImaging/Resample.c`, HPND license), ported to JavaScript in `web/src/preprocess.js`
- [ONNX Runtime Web](https://onnxruntime.ai) (MIT), [Cropper.js](https://fengyuanchen.github.io/cropperjs/) (MIT), [Vite](https://vite.dev) (MIT)
- [Streamlit](https://streamlit.io) and [streamlit-cropper](https://github.com/turner-anderson/streamlit-cropper) for the Streamlit interface
- Fonts: Jost, Inter, Unbounded and Space Grotesk (Google Fonts, SIL OFL)

**Demo video**
- Narration: AI voice generated with [ElevenLabs](https://elevenlabs.io) ("Chris"), script in [`video/SCRIPT.md`](video/SCRIPT.md)
- Music: GarageBand Apple Loops (royalty-free)
- Slides from the Skinsight project presentation

**Statistics cited**
- [Skin Cancer Foundation](https://www.skincancer.org/skin-cancer-information/skin-cancer-facts/) (1 in 5 Americans by age 70)
- [AMN Healthcare 2025 Survey of Physician Appointment Wait Times](https://www.amnhealthcare.com/siteassets/amn-insights/physician/ps-2025-physician-appt-wait-times---wp-v6.pdf) (36.5 days)
- [American Cancer Society](https://www.cancer.org/cancer/types/melanoma-skin-cancer/detection-diagnosis-staging/survival-rates-for-melanoma-skin-cancer-by-stage.html) (melanoma survival by stage)
- [Haenssle et al. 2018](https://doi.org/10.1093/annonc/mdy166) (dermatologist reference)

**AI tools:** Claude Code (Anthropic) helped write the hold-out evaluation scripts, the ONNX export, the web app, the banners and this README. Every result and number above was checked against the scripts' output. The demo video's narration is an ElevenLabs AI voice reading the project's script.

---

## Run It Yourself

**Requirements:** Python 3.12 and Node 20+. Google Chrome is needed only for the optional browser checks. No GPU needed.

```bash
git clone https://github.com/Ar2095/SkinsightAI.git
cd SkinsightAI
python3.12 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Reproduce the evaluation
```bash
python eval/fetch_holdout.py        # downloads the 749 test images listed in eval/data/manifest.csv (~420 MB)
python eval/evaluate.py             # ~2 min on a laptop CPU; writes eval/results/
```
This writes to `eval/results/`:
- `metrics.json` (every number in this README; e.g. `splits.dermoscopic.auroc` = 0.7205)
- `results.md` (all tables, including every subgroup)
- `predictions.csv` (per-image scores)
- `roc.png`, `calibration.png` and `robustness.png`

The run is deterministic: see **Seeds and configuration** (in Evaluation) for every setting and the only ways a rerun can differ.

### Check the deployed model against the evaluated one
```bash
python export_onnx.py                       # rebuilds web/public/model/skinsight.onnx and checks it against PyTorch
python eval/check_web_preprocess.py         # JS resize vs. PIL (needs Node)
cd web && npm ci && npm run build && npx vite preview --port 4173 &
cd .. && python web/scripts/e2e_parity.py   # drives the real UI in headless Chrome
```

### Run the web app
```bash
cd web
npm ci
npm run dev        # http://localhost:5173
```
To deploy, run `vercel deploy --prod` from the repo root. [`vercel.json`](vercel.json) builds `web/` and serves `web/dist`.

### Retrain (optional)
Put the ISIC images in `dataset/<benign|malignant>/<subtype>/*.jpg`, then run `python skin-imaging.py` (a GPU is strongly recommended). It saves `skin_model.pth`; the evaluation uses the committed `dual_head_skin_model.pth`.

### Run the Streamlit interface
```bash
streamlit run skin_classifier.py
```

### Repository layout
```
skinsight_model.py        model definition, class names, preprocessing (shared by eval + export)
dual_head_skin_model.pth  trained checkpoint
skin-imaging.py           training script
skin_classifier.py        Streamlit interface
export_onnx.py            PyTorch -> ONNX + parity check
eval/
  fetch_holdout.py        builds the post-cutoff ISIC test set
  evaluate.py             all metrics, bootstrap CIs, subgroups, robustness, figures
  check_web_preprocess.py JS vs PIL resize check
  data/                   metadata + manifest (images are downloaded, not committed)
  results/                metrics.json, results.md, predictions.csv, figures
web/                      Vite site deployed to Vercel (src/, public/model/, public/devpost/ banners)
video/                    demo video pipeline (script, voiceover split, scene renderer)
```

---

*Skinsight is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. If you have any concerns, please consult a certified dermatologist.*
