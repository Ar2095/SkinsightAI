import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import torch.nn.functional as F  # for softmax

# === Model Definition ===
class DualHeadResNet(nn.Module):
    def __init__(self, num_subtypes):
        super().__init__()
        from torchvision.models import ResNet18_Weights
        self.base = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Identity()

        self.binary_head = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        self.subtype_head = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_subtypes)
        )

    def forward(self, x):
        features = self.base(x)
        binary_out = self.binary_head(features)
        subtype_out = self.subtype_head(features)
        return binary_out, subtype_out

# === Hardcoded subtype list ===
HARDCODED_SUBTYPES = [
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
    "benign_other"
]

PRETTY_SUBTYPE_NAMES = {
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
    "benign_other": "Other (Benign)"
}

@st.cache_resource
def load_model_and_subtypes(model_path="dual_head_skin_model.pth"):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    subtype_to_idx = {name: idx for idx, name in enumerate(HARDCODED_SUBTYPES)}

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    else:
        model_state = checkpoint

    num_subtypes = len(subtype_to_idx)
    if num_subtypes == 0:
        raise ValueError("No subtypes found. Check your hardcoded list.")

    model = DualHeadResNet(num_subtypes=num_subtypes)
    model.load_state_dict(model_state)
    model.eval()

    idx_to_subtype = {v: k for k, v in subtype_to_idx.items()}
    return model, idx_to_subtype


# === Image preprocessing ===
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# === Main Streamlit app ===
st.title("Skin Lesion Classifier")
st.write("Upload an image of a skin lesion to classify it as benign or malignant and identify the subtype.")

model, idx_to_subtype = load_model_and_subtypes()

uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = preprocess_image(image)

    with torch.no_grad():
        bin_out, subtype_out = model(input_tensor)

        prob_malignant = torch.sigmoid(bin_out).item()
        prob_benign = 1 - prob_malignant

        subtype_probs = F.softmax(subtype_out, dim=1).squeeze(0)
        top3_probs, top3_indices = torch.topk(subtype_probs, k=3)

    st.subheader("Prediction:")
    st.write(f"**Predicted Class:** {'Malignant' if prob_malignant > prob_benign else 'Benign'}")
    st.write(f"- Benign Probability: {prob_benign:.4f}")
    st.write(f"- Malignant Probability: {prob_malignant:.4f}")
    

    st.write("**Subtype Predictions:**")
    top3 = torch.topk(subtype_probs, 3)
    for i in range(3):
        idx = top3.indices[i].item()
        prob = top3.values[i].item()
        raw_name = idx_to_subtype.get(idx, "Unknown")
        pretty_name = PRETTY_SUBTYPE_NAMES.get(raw_name, raw_name)
        st.write(f"{pretty_name}: {prob:.4f}")
