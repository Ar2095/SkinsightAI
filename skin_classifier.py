import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ===== Model Definition =====
class DualHeadResNet(nn.Module):
    def __init__(self, num_subtypes):
        super().__init__()
        self.base = models.resnet18(weights=None)  # weights loaded separately
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

# ===== Load Subtype Dictionary =====
subtype_to_idx = {
    "benign_adnexal-epithelial-proliferations": 0,
    "benign_epidermal-proliferations": 1,
    "benign_flat-melanotic-pigmentations": 2,
    "benign_inflammatory-or-infectious-diseases": 3,
    "benign_melanocytic-proliferations": 4,
    "benign_other": 5,
    "benign_soft-tissue-proliferations": 6,
    "malignant_adnexal-epithelial-proliferations": 7,
    "malignant_epidermal-proliferations": 8,
    "malignant_melanocytic-proliferation": 9,
    "malignant_soft-tissue-proliferations": 10
}
idx_to_subtype = {v: k for k, v in subtype_to_idx.items()}

benign_idxs = [v for k, v in subtype_to_idx.items() if "benign" in k]
malignant_idxs = [v for k, v in subtype_to_idx.items() if "malignant" in k]

# ===== Load Model =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DualHeadResNet(num_subtypes=len(subtype_to_idx))
model.load_state_dict(torch.load("dual_head_skin_model.pth", map_location=device))
model.to(device)
model.eval()

# ===== Image Transform =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===== Streamlit UI =====
st.title("Skin Image Classifier")
st.write("Upload an image of skin to determine if it's benign or malignant and predict the subtype.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        binary_out, subtype_out = model(input_tensor)
        binary_prob = torch.sigmoid(binary_out).item()
        if binary_prob > 0.5:
            binary_class = "Malignant"
            binary_confidence = binary_prob
        else:
            binary_class = "Benign"
            binary_confidence = 1 - binary_prob

        subtype_probs = torch.softmax(subtype_out, dim=1).cpu().numpy()[0]

        if binary_class == "Benign":
            subtype_mask = benign_idxs
        else:
            subtype_mask = malignant_idxs

        masked_probs = [(i, subtype_probs[i]) for i in subtype_mask]
        subtype_idx = max(masked_probs, key=lambda x: x[1])[0]
        subtype_name = idx_to_subtype[subtype_idx]
        subtype_confidence = subtype_probs[subtype_idx]

    st.markdown("### Prediction Results")
    st.write(f"**Binary Classification:** {binary_class} (Confidence: {binary_prob:.4f})")
    st.write(f"**Predicted Subtype:** {subtype_name} (Confidence: {subtype_confidence:.4f})")
