import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import os

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="AIMonk Multilabel Image Attribute Classification",
    layout="centered"
)

# --------------------------------
# CONSTANTS
# --------------------------------
NUM_ATTRS = 4
ATTR_NAMES = ["Attr1", "Attr2", "Attr3", "Attr4"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "model.pth"
IMAGE_SIZE = 300   # fixed display size

# --------------------------------
# TRANSFORMS
# --------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------
# LOAD MODEL
# --------------------------------
@st.cache_resource
def load_model():
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, NUM_ATTRS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# --------------------------------
# HEADER
# --------------------------------
from PIL import Image

logo = Image.open("logo.jpg")

st.markdown(
    """
    <style>
    .header-container {
        display: flex;
        align-items: center;   
        gap: 15px;
    }
    .header-title {
        font-size: 38px;
        font-weight: 700;
        margin: 0;
        padding-top: 6px;      
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="header-container">
        <img src="data:image/jpg;base64,{img_data}" width="80"/>
        <div class="header-title">
            AIMonk Multilabel Image Attribute Classification
        </div>
    </div>
    """.format(
        img_data=__import__("base64").b64encode(
            open("logo.jpg", "rb").read()
        ).decode()
    ),
    unsafe_allow_html=True
)
st.markdown(
    "Upload an image and the model will predict which attributes are present."
)

st.divider()

# FILE UPLOAD
uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)

threshold = st.slider(
    "🎯 Prediction Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

st.divider()

# INFERENCE
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1.2])

    # IMAGE COLUMN 
    with col1:
        st.subheader("🖼 Uploaded Image")
        st.image(
            image,
            width=IMAGE_SIZE
        )
        st.caption(f"Image Name: {uploaded_file.name}")

    # PREDICTION COLUMN 
    with col2:
        st.subheader("📊 Prediction Results")

        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs = torch.sigmoid(model(image_tensor))[0]

        predicted_attrs = [
            ATTR_NAMES[i] for i in range(NUM_ATTRS)
            if probs[i] >= threshold
        ]

        if predicted_attrs:
            st.success("Attributes Detected:")
            for attr in predicted_attrs:
                st.markdown(f"✅ **{attr}**")
        else:
            st.warning("No attributes crossed the threshold")

        st.divider()

        st.markdown("### 🔢 Confidence Scores")
        for i, attr in enumerate(ATTR_NAMES):
            st.write(f"{attr}: `{probs[i]:.3f}`")
