import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw
import torch.nn.functional as F
from streamlit_cropper import st_cropper
import pandas as pd

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

# === Create 3x3 grid overlay ===
def create_rule_of_thirds_overlay(size=224, line_color=(255, 0, 0, 100), line_width=2):
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    one_third = size // 3
    two_third = 2 * one_third

    # Vertical lines
    draw.line([(one_third, 0), (one_third, size)], fill=line_color, width=line_width)
    draw.line([(two_third, 0), (two_third, size)], fill=line_color, width=line_width)

    # Horizontal lines
    draw.line([(0, one_third), (size, one_third)], fill=line_color, width=line_width)
    draw.line([(0, two_third), (size, two_third)], fill=line_color, width=line_width)

    return overlay

# === Highlight function for coloring full rows with transparency ===
def highlight_probability_row(row):
    prob = float(row['Probability'])
    if prob > 0.95:
        color = 'rgba(46, 125, 50, 0.2)'  # dark green, 50% transparent
    elif prob > 0.80:
        color = 'rgba(129, 199, 132, 0.2)'  # light green
    elif prob > 0.50:
        color = 'rgba(255, 241, 118, 0.2)'  # yellow
    elif prob > 0.20:
        color = 'rgba(255, 183, 77, 0.2)'  # orange
    else:
        color = 'rgba(229, 115, 115, 0.2)'  # red

    # Apply to all columns in the row
    return [f'background-color: {color}'] * len(row)

# === Main Streamlit app ===
st.title("Skin Lesion Classifier")
st.write("Upload an image of a skin lesion to classify it as benign or malignant and identify the subtype.")

with st.expander("📘 Learn about skin cancer: The ABCDEs of melanoma"):
    st.markdown("### How to Recognize Signs of Melanoma")
    st.markdown("If you notice any of these signs, it's important to consult a healthcare provider.")
    image = Image.open("images/abcde_chart.png")
    st.image(image, use_container_width=True)

model, idx_to_subtype = load_model_and_subtypes()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.markdown("### Step 1: Crop the image")
    st.write(
        """
        **Guidelines for cropping:**

        - Adjust the crop box so the skin lesion is centered within the square.
        - Make sure the lesion is clearly visible and occupies a good portion of the box,
          but avoid zooming in too closely.
        - The crop area should be square to prevent distortion.
        - Try to exclude unnecessary background to improve model accuracy.
        """
    )

    # Display good crop examples
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("images/crop1.png", use_container_width=True)
    with col2:
        st.image("images/crop2.png", use_container_width=True)
    with col3:
        st.image("images/crop3.png", use_container_width=True)

    # Side-by-side cropping and preview with 2:1 ratio columns
    col_crop, col_preview = st.columns([2, 1])

    with col_crop:
        st.markdown("#### Crop the Lesion")
        # Resize uploaded image to fixed width for cropper
        fixed_width = 450
        aspect_ratio = image.width / image.height
        new_height = int(fixed_width / aspect_ratio)
        resized_for_cropper = image.resize((fixed_width, new_height))

        cropped_img = st_cropper(
            resized_for_cropper,
            aspect_ratio=(1, 1),
            box_color='#FF4B4B',
            return_type='image',
            realtime_update=True
        )

    with col_preview:
        st.markdown("#### Preview with 3x3 Grid")
        cropped_img_resized = cropped_img.resize((224, 224))
        overlay = create_rule_of_thirds_overlay(size=224)
        img_with_grid = Image.alpha_composite(cropped_img_resized.convert("RGBA"), overlay)
        st.image(img_with_grid, caption="Cropped Image with 3x3 Grid", use_container_width=True)

    # === Prediction ===
    input_tensor = preprocess_image(cropped_img_resized)

    with torch.no_grad():
        bin_out, subtype_out = model(input_tensor)

        prob_malignant = torch.sigmoid(bin_out).item()
        prob_benign = 1 - prob_malignant

        subtype_probs = F.softmax(subtype_out, dim=1).squeeze(0)

    # Display main prediction
    main_pred_class = "Malignant" if prob_malignant > prob_benign else "Benign"
    st.subheader("Prediction")
    st.write(f"**Prediction:** {main_pred_class}")

    # Probability table for benign/malignant with row coloring
    # Probability table for benign/malignant with row coloring and sorted by probability descending
    class_df = pd.DataFrame([
        {"Prediction": "Benign", "Probability": prob_benign},
        {"Prediction": "Malignant", "Probability": prob_malignant}
    ])
    class_df = class_df.sort_values(by="Probability", ascending=False).reset_index(drop=True)

    styled_class_df = class_df.style.format({"Probability": "{:.4f}"}).apply(
        highlight_probability_row, axis=1
    )
    st.dataframe(styled_class_df, hide_index=True, use_container_width=True)


    # === Top Subtype Predictions ===
    st.subheader("Subtype Predictions")

    # Create list of predictions over 1% probability
    visible_preds = []
    for idx, prob in enumerate(subtype_probs):
        if prob.item() > 0.01:
            raw_name = idx_to_subtype.get(idx, "Unknown")
            pretty_name = PRETTY_SUBTYPE_NAMES.get(raw_name, raw_name)
            visible_preds.append({"Subtype": pretty_name, "Probability": prob.item()})

    # Sort visible_preds by probability descending
    visible_preds = sorted(visible_preds, key=lambda x: x["Probability"], reverse=True)

    if visible_preds:
        subtype_df = pd.DataFrame(visible_preds)
        styled_subtype_df = subtype_df.style.format({"Probability": "{:.4f}"}).apply(
            highlight_probability_row, axis=1
        )
        st.dataframe(styled_subtype_df, hide_index=True, use_container_width=True)
    else:
        st.write("No subtype predictions exceeded the 0.01 probability threshold.")
    st.caption("Only predictions with probability greater than 0.01 are shown.")


    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.9em; color: gray; padding-top: 1em;">
        Disclaimer: This tool is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.  
        If you have any concerns, please consult a certified dermatologist.
        </div>
        """,
        unsafe_allow_html=True
    )
