import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw
import torch.nn.functional as F
from streamlit_cropper import st_cropper
import pandas as pd
import streamlit.components.v1 as components




#background css :)
st.markdown(
    """
    <style>
    /* Change the entire app background */
    .stApp {
        background-color: #2e0118;  
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown('<div id="top"></div>', unsafe_allow_html=True)

def scroll_to_anchor(anchor_id="top"):
    components.html(f"""
        <script>
        const el = document.getElementById('{anchor_id}');
        if (el) {{
            el.scrollIntoView({{behavior: 'smooth'}});
        }}
        </script>
    """, height=0)


# model
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

# subtypes and printable names
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

    model = DualHeadResNet(num_subtypes=len(subtype_to_idx))
    model.load_state_dict(model_state)
    model.eval()

    idx_to_subtype = {v: k for k, v in subtype_to_idx.items()}
    return model, idx_to_subtype

# image preprocessing for model
def preprocess_image(image: Image.Image) -> torch.Tensor:
    # Ensure the image is RGB (3 channels)
    if image.mode != "RGB":
        image = image.convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


# crop preview overlay
def create_rule_of_thirds_overlay(size=224, line_color=(255, 0, 0, 100), line_width=2):
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    one_third = size // 3
    two_third = 2 * one_third

    draw.line([(one_third, 0), (one_third, size)], fill=line_color, width=line_width)
    draw.line([(two_third, 0), (two_third, size)], fill=line_color, width=line_width)
    draw.line([(0, one_third), (size, one_third)], fill=line_color, width=line_width)
    draw.line([(0, two_third), (size, two_third)], fill=line_color, width=line_width)

    return overlay

#highlight
def highlight_multiclass_prediction(row):
    subtype = row['Subtype'].lower()
    prob = float(row['Probability'])

    if 'malignant' in subtype:
        if prob > 0.9:
            color = 'rgba(198, 40, 40, 0.3)' #dark red
        elif prob > 0.3:
            color = 'rgba(239, 83, 80, 0.3)'  #light red
        elif prob > 0.1:
            color = 'rgba(255, 160, 0, 0.3)' #orange
        else:
            color = ''
    elif 'benign' in subtype:
        if prob > 0.9:
            color = 'rgba(27, 94, 32, 0.3)'  #dark green
        elif prob > 0.3:
            color = 'rgba(102, 187, 106, 0.3)'  #light green
        elif prob > 0.1:
            color = 'rgba(255, 241, 118, 0.3)' #yellow
        else:
            color = ''
    else:
        color = ''

    return [f'background-color: {color}' if color else '' for _ in row]


def highlight_binary_prediction(row):
    pred = row['Prediction']
    prob = float(row['Probability'])

    if pred == 'Malignant':
        if prob > 0.9:
            color = 'rgba(198, 40, 40, 0.3)'
        elif prob > 0.75:
            color = 'rgba(239, 83, 80, 0.3)'
        elif prob > 0.5:
            color = 'rgba(255, 160, 0, 0.3)'
        else:
            color = ''
    elif pred == 'Benign':
        if prob > 0.9:
            color = 'rgba(27, 94, 32, 0.3)'
        elif prob > 0.75:
            color = 'rgba(102, 187, 106, 0.3)'
        elif prob > 0.5:
            color = 'rgba(255, 241, 118, 0.3)'
        else:
            color = ''
    else:
        color = ''
    return [f'background-color: {color}' if color else '' for _ in row]





# MAIN APP STARTS HERE

if "model" not in st.session_state or "idx_to_subtype" not in st.session_state:
    model, idx_to_subtype = load_model_and_subtypes()
    st.session_state.model = model
    st.session_state.idx_to_subtype = idx_to_subtype


if "stage" not in st.session_state:
    st.session_state.stage = 1
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "cropped_img" not in st.session_state:
    st.session_state.cropped_img = None

if st.session_state.stage == 1:
    logo_img = Image.open("images/skinsight_logo.png")
    st.image(logo_img, width=700)
    st.markdown("---")
    st.write("Skin cancer is the most common cancer in the United States. Early detection can save your life.")
    with st.expander("📕 Learn more about skin cancer: The ABCDEs of melanoma"):
        st.markdown("### How to Recognize Signs of Melanoma")
        st.image("images/abcde_chart.png", use_container_width=True)
    st.markdown("---")
    st.markdown("## Image Upload")
    st.write("Upload an image of a skin mark to classify it as benign or malignant and identify the subtype.")



    model, idx_to_subtype = load_model_and_subtypes()

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="hidden")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.session_state.uploaded_file = uploaded_file
        if st.button("Next: Crop Image"):
            st.session_state.stage = 2
            st.session_state.scroll_to_top = True
            st.rerun()
            scroll_to_anchor()


#CROP
if st.session_state.stage == 2:
    logo_img = Image.open("images/skinsight_logo.png")
    st.image(logo_img, width=700)
    st.markdown("---")
    st.markdown("## Crop the Image")
    if st.session_state.uploaded_file is not None:
        image = Image.open(st.session_state.uploaded_file)
        #st.image(image, caption="Original Image", use_container_width=True)
        st.write(
            """
            **Guidelines for cropping:**

            - Adjust the crop box so the skin lesion is centered within the square.
            - Make sure the lesion is clearly visible, but avoid zooming in too closely.
            - Try to exclude unnecessary background to improve model accuracy.
            """
        )
        st.write("""##### **Examples of good crops:**""")

        col1, col2, col3 = st.columns(3)
        with col1: st.image("images/crop1.png", use_container_width=True)
        with col2: st.image("images/crop2.png", use_container_width=True)
        with col3: st.image("images/crop3.png", use_container_width=True)

        st.markdown("---")
        col_crop, col_preview = st.columns([2, 1])
        with col_crop:
            st.markdown("#### Crop Using the Bounding Box")
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
            st.markdown("#### Preview with Grid")
            cropped_img_resized = cropped_img.resize((224, 224))
            overlay = create_rule_of_thirds_overlay(size=224)
            img_with_grid = Image.alpha_composite(cropped_img_resized.convert("RGBA"), overlay)
            st.image(img_with_grid, use_container_width=True)

        if st.button("Fake Crop → Next"):
            st.session_state.cropped_img = cropped_img_resized  # just pass through
            st.session_state.stage = 3
            st.rerun()
            scroll_to_anchor()


        
    else:
        st.warning("Please upload an image first.")
        st.button("Back to Upload", on_click=lambda: setattr(st.session_state, "stage", 1))
    

#PREDICT
if st.session_state.stage == 3:
    logo_img = Image.open("images/skinsight_logo.png")
    st.image(logo_img, width=700)
    st.markdown("---")
    model = st.session_state.model
    idx_to_subtype = st.session_state.idx_to_subtype

    if st.session_state.cropped_img is not None:
        input_tensor = preprocess_image(st.session_state.cropped_img)

        with torch.no_grad():
            bin_out, subtype_out = model(input_tensor)

            prob_malignant = torch.sigmoid(bin_out).item()
            prob_benign = 1 - prob_malignant

            subtype_probs = F.softmax(subtype_out, dim=1).squeeze(0)

            

        main_pred_class = "Malignant" if prob_malignant > prob_benign else "Benign"
        st.markdown("## Prediction")

        color = "rgba(27, 94, 32, 0.7)" if main_pred_class == "Benign" else "rgba(198, 40, 40, 0.7)"
        prob_value = prob_benign if main_pred_class == "Benign" else prob_malignant

        st.markdown(
            f"""
            <div style="
                background-color: {color};
                color: white;
                padding: 10px;
                border-radius: 10px;
                font-size: 24px;
                font-weight: bold;
                text-align: center;
                width: 60%;
                margin: 10px auto;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            ">
                Prediction: {main_pred_class}
                <div style="font-size: 18px; font-weight: normal; margin-top: 10px;">
                    Probability: {prob_value:.4f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if main_pred_class == "Benign":
            advice = "This means it is likely to be not cancerous and completely normal! However, make sure to monitor changes over time. If you notice changes in size, shape, or color; or if growth becomes painful or starts bleeding, schedule a consultation with a dermatologist immediately. Please remember that this tool does not replace professional medical evaluation."
        else:
            advice = "This means it may be cancerous and harmful. It's recommended that you consult a dermatolagist for a clinical skin examination or dermoscopy as soon as possible. Early treatment is often very effective, and the sooner it is caught, the better the outcome."
        st.write(advice)

        st.markdown("### Subtype Predictions")
        visible_preds = []
        for idx, prob in enumerate(subtype_probs):
            if prob.item() > 0.01:
                raw_name = idx_to_subtype.get(idx, "Unknown")
                pretty_name = PRETTY_SUBTYPE_NAMES.get(raw_name, raw_name)
                visible_preds.append({"Subtype": pretty_name, "Probability": prob.item()})

        visible_preds = sorted(visible_preds, key=lambda x: x["Probability"], reverse=True)

        if visible_preds:
            subtype_df = pd.DataFrame(visible_preds)
            styled_subtype_df = subtype_df.style.format({"Probability": "{:.4f}"}).apply(
                highlight_multiclass_prediction, axis=1
            )

            st.dataframe(styled_subtype_df, hide_index=True, use_container_width=True)
        else:
            st.write("No subtype predictions exceeded the 0.01 probability threshold.")
        st.caption("Only predictions with probability greater than 0.01 are shown.")

    else:
        st.warning("No cropped image found.")

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