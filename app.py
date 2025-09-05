import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from io import BytesIO
from PIL import Image
import gdown
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- CONFIG ---
MODEL_PATH = "models/best_resnet18.pth"
MODEL_DRIVE_ID = "1yySFeUxgcN0uqiGbenRCIxhKlhgDwQFJ"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Download model if not found ---
os.makedirs("models", exist_ok=True)
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}", MODEL_PATH, quiet=False)

# --- Define Transform ---
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.CenterCrop(IMG_SIZE, IMG_SIZE),
    A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, value=255),
    A.Normalize(),
    ToTensorV2()
])

# --- Load Model ---
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)  # Binary classification
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

st.title("🧾 Receipt Forgery Detection with Grad-CAM")
st.write("Upload a receipt image to check if it’s **Genuine** or **Forged**. 🚀")

uploaded_file = st.file_uploader("Upload Receipt", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Receipt", width=400)

    img_array = np.array(image)
    transformed = transform(image=img_array)
    input_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

    # --- Prediction ---
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        pred_class = 1 if prob > 0.5 else 0
        confidence = prob if pred_class == 1 else 1 - prob

    # --- Display Results ---
    st.subheader(f"Prediction: {'GENUINE 🟢' if pred_class == 0 else 'FORGED 🔴'}")
    st.write(f"Confidence: {confidence*100:.2f}%")

    # --- Confidence Bar ---
    color = "green" if pred_class == 0 else "red"
    st.markdown(
        f"""
        <div style="width: 100%; background-color: #ddd; border-radius: 8px; margin-bottom:10px;">
            <div style="width: {confidence*100:.1f}%; background-color: {color};
                        padding: 4px; border-radius: 8px; text-align: center;
                        color: white; font-weight: bold;">
                {confidence*100:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- OPTIONAL: Confidence Gauge ---
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence*100,
        title={'text': "Model Confidence"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green" if pred_class == 0 else "red"},
            'steps': [
                {'range': [0, 50], 'color': "lightcoral"},
                {'range': [50, 100], 'color': "lightgreen"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # --- Grad-CAM Section (Optional Visualization) ---
    st.info("Grad-CAM visualization coming soon (optional enhancement).")
