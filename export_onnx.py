"""Export the PyTorch checkpoint to ONNX for in-browser inference, then verify parity.

Usage: python export_onnx.py
Writes web/public/model/skinsight.onnx. The graph takes a 1x3x224x224 normalised
image and returns P(malignant) (sigmoid applied) and the 11 subtype probabilities
(softmax applied), so the browser does no post-processing beyond display.
Parity is checked against PyTorch on every image in eval/data (if downloaded).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from skinsight_model import ROOT, load_model, preprocess

OUT = ROOT / "web" / "public" / "model" / "skinsight.onnx"


class WithProbabilities(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        b, s = self.model(x)
        return torch.sigmoid(b).squeeze(1), torch.softmax(s, dim=1)


def main():
    model = WithProbabilities(load_model()).eval()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.zeros(1, 3, 224, 224), str(OUT), dynamo=False,
                      input_names=["image"], output_names=["p_malignant", "p_subtype"],
                      dynamic_axes={"image": {0: "batch"}, "p_malignant": {0: "batch"},
                                    "p_subtype": {0: "batch"}},
                      opset_version=17)
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")

    import onnxruntime as ort
    from PIL import Image
    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    images = sorted((ROOT / "eval" / "data" / "images").glob("*.jpg"))
    if not images:
        images = sorted((ROOT / "images").glob("crop*.png"))
    worst_p, worst_s, flips = 0.0, 0.0, 0
    for i in range(0, len(images), 64):
        x = torch.stack([preprocess(Image.open(p)) for p in images[i:i + 64]])
        with torch.no_grad():
            tp, ts = model(x)
        op, os_ = sess.run(None, {"image": x.numpy()})
        worst_p = max(worst_p, float(np.abs(op - tp.numpy()).max()))
        worst_s = max(worst_s, float(np.abs(os_ - ts.numpy()).max()))
        flips += int(((op > 0.5) != (tp.numpy() > 0.5)).sum())
    print(f"Parity on {len(images)} images: max |dP(malignant)| = {worst_p:.2e}, "
          f"max |dP(subtype)| = {worst_s:.2e}, benign/malignant decisions changed = {flips}")


if __name__ == "__main__":
    main()
