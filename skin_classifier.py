import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

# === Dataset class to extract subtypes from folders ===
class SkinDataset:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.subtype_to_idx = {}
        # Scan folders to build subtype_to_idx map
        for label_dir in self.root_dir.glob("*"):
            if not label_dir.is_dir():
                continue
            for subfolder in label_dir.iterdir():
                if subfolder.is_dir():
                    subtype_name = subfolder.name.lower()
                    if subtype_name not in self.subtype_to_idx:
                        self.subtype_to_idx[subtype_name] = len(self.subtype_to_idx)

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

# === Load model and subtypes from dataset folder ===
@st.cache_resource
def load_model_and_subtypes(dataset_path="dataset", model_path="dual_head_skin_model.pth"):
    dataset = SkinDataset(dataset_path)
    subtype_to_idx = dataset.subtype_to_idx
    idx_to_subtype = {v: k for k, v in subtype_to_idx.items()}

    model = DualHeadResNet(num_subtypes=len(subtype_to_idx))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
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

# Load model and subtypes once
model, idx_to_subtype = load_model_and_subtypes()

st.write(f"Detected Subtypes ({len(idx_to_subtype)}):")
st.write(list(idx_to_subtype.values()))

uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = preprocess_image(image)

    with torch.no_grad():
        bin_out, subtype_out = model(input_tensor)
        bin_pred = torch.sigmoid(bin_out).item()
        subtype_pred = torch.argmax(subtype_out, dim=1).item()

    st.subheader("Prediction:")
    st.write(f"**Binary Classification:** {'Malignant' if bin_pred > 0.5 else 'Benign'} (Confidence: {bin_pred:.2f})")
    st.write(f"**Subtype:** {idx_to_subtype[subtype_pred]}")
