# Skinsight demo video: narration script

The video uses a neural text-to-speech voice (Microsoft Edge TTS, en-US-AndrewNeural). To use your own voice, record each line below as its own clip. Clip lengths feed the demo pacing (`record_demo.py`), but the scene animations are cued to the TTS word timings (`build/words.json`), so recorded clips also need those cues re-timed before running `build_video.py`.

| Beat | TTS length | Line |
|---|---|---|
| `hook` | 4.8s | At least one in five Americans will develop skin cancer by the time they turn seventy. |
| `survival` | 9.6s | And catching it early changes everything. Melanoma found early has a five-year survival rate above ninety-nine percent. Once it spreads, that falls to thirty-six. |
| `access` | 12.0s | But getting checked isn't easy. The average wait for a new dermatology appointment is over a month. Rural areas may have no dermatologist at all, and most skin-check apps are generic, or locked behind a paywall. |
| `intro` | 5.3s | So we built Skinsight. AI-powered dermatology insights, running right in your browser. |
| `open` | 7.2s | Here's how it works. When you open the site, the model loads straight onto your device, so your photo is never uploaded anywhere. |
| `benign_upload` | 3.6s | Let's start with a mole that's benign. I upload the photo, |
| `benign_crop` | 5.3s | and line up the square on the spot. The preview on the right is exactly what the model will see. |
| `benign_result` | 12.4s | Skinsight says it looks benign, with about a six percent chance of being malignant. Its top lesion type is a melanocytic proliferation: a common mole. And it tells you how much to trust that answer, based on how it did in our testing. |
| `malig_upload` | 6.1s | Next, a spot with the classic warning signs of melanoma: an irregular border, and uneven color. |
| `malig_result` | 14.8s | This one is flagged as malignant, at ninety-seven percent, with melanoma as the top lesion type. Skinsight's advice: see a dermatologist as soon as you can. It's also honest that, at scores this high, about one in five lesions in our tests really were cancer. |
| `friend_upload` | 11.1s | And finally, just for fun, I took a picture of my friend's mole, on a regular phone camera. This photo obviously wasn't in any dataset, so it's a great test of what real-life data looks like. |
| `friend_result` | 5.8s | Skinsight calls it benign, at under one percent, which matches what we know. It's a harmless mole. |
| `data` | 11.0s | Under the hood, Skinsight was trained on over seventy-eight thousand labeled images from the International Skin Imaging Collaboration archive, sorted into benign and malignant, and eleven lesion types. |
| `model` | 17.6s | The model is a ResNet-18 convolutional neural network, pretrained on ImageNet, with two output heads. One predicts benign versus malignant. The other predicts the specific lesion type. For this version, we exported it to ONNX, so it runs entirely in the browser, with no server. |
| `eval_old` | 10.0s | When we first built Skinsight, our test split showed ninety percent accuracy. But that split was random, so images of the same lesion could end up in both training and testing. |
| `eval_new` | 16.2s | So this time, we tested it properly, on five hundred ninety-eight lesions that were added to the ISIC archive after the model was trained. The AUROC was zero point seven two. At its default setting, it caught about half of the cancers. Dermatologists catch about eighty-seven percent. |
| `eval_limits` | 19.4s | We also measured where it breaks: blurry or badly lit photos, spots that are small in the frame, and even re-cropping, which flips about one in five answers. Darker skin tones are barely covered by the test data at all. That's why Skinsight is an educational tool, not a diagnosis, and why every result tells you exactly how much to trust it. |
| `outro` | 4.3s | Skinsight. Check a spot in under a minute, and know when to see a doctor. |
