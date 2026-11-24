"""Skinsight model definition, class names and preprocessing.

Shared by the evaluation scripts and the ONNX export, so the numbers in the
README and the model running in the browser come from identical code.
"""

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).parent
CHECKPOINT = ROOT / "dual_head_skin_model.pth"

# Output order of the subtype head, as used by the app.
SUBTYPES = [
    "malignant_soft-tissue-proliferations",
    "malignant_adnexal-epithelial-proliferations",
    "malignant_melanocytic-proliferation",
    "malignant_epidermal-proliferations",
    "benign_flat-melanotic-pigmentations",
    "benign_soft-tissue-proliferations",
    "benign_inflammatory-or-infectious-diseases",
    "benign_adnexal-epithelial-proliferations",
    "benign_melanocytic-proliferations",
    "benign_epidermal-proliferations",
    "benign_other",
]

PRETTY = {
    "malignant_soft-tissue-proliferations": "Soft Tissue Proliferations (Malignant)",
    "malignant_adnexal-epithelial-proliferations": "Adnexal Epithelial Proliferations (Malignant)",
    "malignant_melanocytic-proliferation": "Melanocytic Proliferations (Malignant)",
    "malignant_epidermal-proliferations": "Epidermal Proliferations (Malignant)",
    "benign_flat-melanotic-pigmentations": "Flat Melanotic Pigmentations (Benign)",
    "benign_soft-tissue-proliferations": "Soft Tissue Proliferations (Benign)",
    "benign_inflammatory-or-infectious-diseases": "Inflammatory or Infectious Diseases (Benign)",
    "benign_adnexal-epithelial-proliferations": "Adnexal Epithelial Proliferations (Benign)",
    "benign_melanocytic-proliferations": "Melanocytic Proliferations (Benign)",
    "benign_epidermal-proliferations": "Epidermal Proliferations (Benign)",
    "benign_other": "Other (Benign)",
}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Same preprocessing as the app: squash to 224x224 (no aspect-ratio crop), ImageNet norm.
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class DualHeadResNet(nn.Module):
    """ResNet-18 backbone with two heads: benign/malignant (1 logit) and subtype (11 logits)."""

    def __init__(self, num_subtypes=len(SUBTYPES)):
        super().__init__()
        self.base = models.resnet18(weights=None)
        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Identity()
        self.binary_head = nn.Sequential(
            nn.Linear(num_ftrs, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
        self.subtype_head = nn.Sequential(
            nn.Linear(num_ftrs, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_subtypes))

    def forward(self, x):
        features = self.base(x)
        return self.binary_head(features), self.subtype_head(features)


def load_model(path=CHECKPOINT):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model = DualHeadResNet()
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess(image: Image.Image) -> torch.Tensor:
    return PREPROCESS(image.convert("RGB"))
