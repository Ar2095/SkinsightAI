# Skinsight demo video: narration script

The video uses a neural text-to-speech voice (Microsoft Edge TTS, en-US-BrianNeural). To use your own voice, record each line below as its own clip. Clip lengths feed the demo pacing (`record_demo.py`), but the scene animations are cued to the TTS word timings (`build/words.json`), so recorded clips also need those cues re-timed before running `build_video.py`.

| Beat | TTS length | Line |
|---|---|---|
| `hook` | 4.6s | At least one in five Americans will get skin cancer by the time they're seventy. |
| `survival` | 10.1s | And catching it early changes everything. When melanoma's found early, the five-year survival rate is above ninety-nine percent. Once it spreads, that drops to thirty-six. |
| `access` | 13.3s | But getting checked isn't easy. The average wait for a new dermatology appointment is over a month. A lot of rural areas don't have a dermatologist at all. And most skin-check apps are either generic, or stuck behind a paywall. |
| `intro` | 5.4s | So, we built Skinsight. AI-powered dermatology insights, right in your browser. |
| `open` | 7.4s | Here's how it works. When you open the site, the model loads right onto your device, so your photo never gets uploaded anywhere. |
| `benign_upload` | 8.1s | Let's start with a benign mole. This photo comes from a different dataset than the one we trained on, so the model's never seen it. I'll upload it, |
| `benign_crop` | 5.4s | and line the square up on the spot. The preview on the right is exactly what the model sees. |
| `benign_result` | 13.6s | Skinsight says it looks benign, with about a six percent chance of being malignant. Its top guess for the lesion type is a melanocytic proliferation, so, a common mole. And it tells you how much to trust that answer, based on how it did in our testing. |
| `malig_upload` | 6.7s | Next up, a melanoma, also from outside our training data. You can see the irregular border, and the uneven color. |
| `malig_result` | 14.7s | This one gets flagged as malignant, at ninety-seven percent, and melanoma is the top lesion type. Its advice: see a dermatologist as soon as you can. And it's honest that, at scores this high, about one in five lesions in our tests actually were cancer. |
| `friend_upload` | 11.1s | And finally, just for fun, I took a picture of my friend's mole, on a regular phone camera. This photo obviously wasn't in any dataset, so it's a really good test of what real-life data looks like. |
| `friend_result` | 5.9s | Skinsight calls it benign, at under one percent. Which matches what we know. It's a harmless mole. |
| `data` | 11.3s | Under the hood, Skinsight was trained on over seventy-eight thousand labeled images from the International Skin Imaging Collaboration archive, sorted into benign and malignant, and eleven lesion types. |
| `model` | 17.1s | The model's a ResNet-18 convolutional neural network, pretrained on ImageNet, with two output heads. One predicts benign versus malignant. The other predicts the specific lesion type. For this version, we exported it to ONNX, so it runs entirely in the browser, with no server. |
| `eval_old` | 9.7s | When we first built Skinsight, our test split showed ninety percent accuracy. But that split was random, so pictures of the same lesion could end up in both training and testing. |
| `eval_new` | 17.7s | So this time, we tested it properly, on five hundred ninety-eight lesions that were added to the ISIC archive after the model was trained. The AUROC was zero point seven two. At its default setting, it caught about half of the cancers. For comparison, dermatologists catch about eighty-seven percent. |
| `eval_limits` | 17.6s | We also tested where it breaks. Blurry or badly lit photos, spots that are small in the frame, and even re-cropping, which flips about one in five answers. So it's not going to be right every time. That's why Skinsight is an educational tool, not a diagnosis, and why every result tells you how much to trust it. |
| `outro` | 4.3s | Skinsight. Check a spot in under a minute, and know when to see a doctor. |
