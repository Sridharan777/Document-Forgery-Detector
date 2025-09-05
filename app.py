import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import imageio
import gdown
import os

# -------------------------------
# 1️⃣ CONFIG
# -------------------------------
MODEL_PATH = "models/best_resnet18.pth"
MODEL_DRIVE_ID = "1yySFeUxgcN0uqiGbenRCIxhKlhgDwQFJ"
IMG_SIZE = 224
CLASS_NAMES = ["Real Receipt", "Fake Receipt"]

# -------------------------------
# 2️⃣ DOWNLOAD MODEL (if missing)
# -------------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("Model not found locally. Downloading from Google Drive...")
        os.makedirs("models", exist_ok=True)
        url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# -------------------------------
# 3️⃣ LOAD MODEL SAFELY
# -------------------------------
@st.cache_resource
def load_model():
    try:
        download_model()
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
        model.eval()
        return model
    except Exception as e:
        st.error(f"⚠️ Failed to load model: {e}")
        return None

# -------------------------------
# 4️⃣ IMAGE PREPROCESSING
# -------------------------------
def preprocess_image(image):
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])
    image_np = np.array(image)
    image = transform(image=image_np)["image"]
    return image.unsqueeze(0)

# -------------------------------
# 5️⃣ PREDICTION FUNCTION
# -------------------------------
def predict_image(model, image):
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)
        return CLASS_NAMES[predicted], conf.item()

# -------------------------------
# 6️⃣ GRAD-CAM FUNCTION (Optional Visualization)
# -------------------------------
def generate_gradcam(model, image):
    # Simple Grad-CAM implementation
    image = image.requires_grad_()
    model.zero_grad()
    outputs = model(image)
    class_idx = outputs.argmax(dim=1)
    outputs[0, class_idx].backward()

    gradients = model.layer4[1].conv2.weight.grad
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = model.layer4[1].conv2(image).detach()

    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()
    heatmap = cv2.resize(heatmap.numpy(), (IMG_SIZE, IMG_SIZE))
    return heatmap

# -------------------------------
# 7️⃣ STREAMLIT UI
# -------------------------------
st.title("🧾 Receipt Forgery Detector")
st.write("Upload a receipt image to detect if it is **Real** or **Fake**.")

uploaded_file = st.file_uploader("Upload Receipt", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    model = load_model()
    if model:
        input_tensor = preprocess_image(image)
        label, confidence = predict_image(model, input_tensor)

        st.subheader(f"Prediction: **{label}**")
        st.write(f"Confidence: **{confidence*100:.2f}%**")

        # Grad-CAM visualization
        heatmap = generate_gradcam(model, input_tensor)
        overlay = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        blended = cv2.addWeighted(np.array(image.resize((IMG_SIZE, IMG_SIZE))), 0.5, overlay, 0.5, 0)

        st.image(blended, caption="Model Attention (Grad-CAM)", use_container_width=True)

        # Download button
        buf = BytesIO()
        Image.fromarray(blended).save(buf, format="PNG")
        buf.seek(0)
        st.download_button("Download Heatmap", buf, file_name="heatmap.png", mime="image/png")

else:
    st.info("👆 Please upload an image to start.")
