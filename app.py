import subprocess
import sys

# --- Dynamic install for pytorch-grad-cam if missing ---
try:
    import pytorch_grad_cam
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-grad-cam"])
    import pytorch_grad_cam


import streamlit as st
import os, io
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import imageio

try:
    import gdown
except:
    gdown = None

IMG_SIZE = 224
LOCAL_MODEL_PATH = "models/best_resnet18.pth"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def download_model_if_needed():
    if os.path.exists(LOCAL_MODEL_PATH):
        return
    if "MODEL_GDRIVE_ID" in st.secrets and gdown:
        os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
        url = f"https://drive.google.com/uc?id={st.secrets['MODEL_GDRIVE_ID']}"
        st.info("Downloading model from Google Drive... please wait.")
        gdown.download(url, LOCAL_MODEL_PATH, quiet=False)
    else:
        st.error("Model not found locally and MODEL_GDRIVE_ID not set in secrets (or gdown missing).")

def preprocess_for_model(img_rgb):
    tfm = A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, value=255),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return tfm(image=img_rgb)["image"].unsqueeze(0)

@st.cache_resource(show_spinner=False)
def load_model():
    download_model_if_needed()
    if not os.path.exists(LOCAL_MODEL_PATH):
        st.stop()  # stop if download failed
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    ckpt = torch.load(LOCAL_MODEL_PATH, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model

def compute_gradcam(tensor_img, display_img):
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=tensor_img, targets=None, aug_smooth=False, eigen_smooth=False)[0]
    cam_image = show_cam_on_image(display_img, grayscale_cam, use_rgb=True)
    return cam_image

st.set_page_config(page_title="Receipt Forgery Detector", layout="centered")
st.title("🧾 Receipt Forgery Detector (Grad-CAM)")

model = load_model()

uploaded_file = st.file_uploader("Upload a receipt image (png/jpg)", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    display_img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)) / 255.0
    tensor = preprocess_for_model(img_rgb)

    with torch.no_grad():
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()
        pred = int(prob >= 0.5)

    label = "FORGED" if pred == 1 else "GENUINE"
    st.markdown(f"**Prediction:** {label}    **Confidence:** {prob:.3f}")

    cam_img = compute_gradcam(tensor, display_img)
    st.image(cam_img, caption="Grad-CAM overlay (model attention)", use_column_width=True)

    buf = io.BytesIO()
    imageio.imwrite(buf, cam_img)
    buf.seek(0)
    st.download_button("Download heatmap", buf, file_name="heatmap.png", mime="image/png")
