# app.py — Compact 3-Column Layout with Gauge
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import os, traceback, time, zipfile, gdown
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
IMG_SIZE = 224
MODEL_PATH = "models/best_resnet50.pth"
FALLBACK_GDRIVE_ID = "1zMrv6S6rOWyiTQ0Fgw0VG0o0fEnrWzE-"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

MAX_HEIGHT, MAX_WIDTH = 450, 350  # controlled aspect ratio for both images

# ---------------- HELPERS ----------------
def download_model_if_missing(gdrive_id: str):
    if os.path.exists(MODEL_PATH): return True
    if not gdrive_id: return False
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    try:
        st.info("📥 Downloading model from Google Drive...")
        gdown.download(url, MODEL_PATH, quiet=False)
        time.sleep(1.0)
        return os.path.exists(MODEL_PATH)
    except Exception as e:
        st.error(f"Model download failed: {e}")
        return False

@st.cache_resource(show_spinner=True)
def load_model():
    try:
        drive_id = None
        try:
            drive_id = st.secrets.get("MODEL_GDRIVE_ID", None)
        except Exception:
            pass

        if not os.path.exists(MODEL_PATH):
            if not download_model_if_missing(drive_id) and not download_model_if_missing(FALLBACK_GDRIVE_ID):
                st.error("Model not found locally and no valid Drive ID available.")
                return None

        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        state_dict = ckpt
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model_state_dict", "net", "state"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state_dict = ckpt[key]
                    break
            if state_dict is ckpt and any(k.endswith(".weight") for k in ckpt.keys()):
                state_dict = ckpt

        if not isinstance(state_dict, dict):
            st.error("Checkpoint format not recognized.")
            return None

        new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
        out_features = new_state.get("fc.weight", torch.empty((2,))).shape[0] if "fc.weight" in new_state else 2

        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, out_features)
        model.load_state_dict(new_state, strict=False)
        model.to(DEVICE).eval()
        st.success("✅ Model loaded (ResNet50).")
        return model

    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.text(traceback.format_exc())
        return None

def pil_to_tensor(img_pil: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(img_pil).unsqueeze(0).to(DEVICE)

def predict_single(model, input_tensor):
    with torch.no_grad():
        out = model(input_tensor).to(torch.float32)
        if out.shape[1] == 1:
            prob = torch.sigmoid(out)[0, 0].item()
            return ("FORGED 🔴", prob, out) if prob >= 0.5 else ("GENUINE 🟢", 1.0 - prob, out)
        probs = torch.softmax(out, dim=1)[0]
        idx = int(torch.argmax(probs))
        return ("FORGED 🔴" if idx == 1 else "GENUINE 🟢", float(probs[idx]), out)

def compute_gradcam(model, input_tensor, target_layer=None):
    activations, gradients = [], []
    if target_layer is None: target_layer = model.layer4[-1]
    def forward_hook(m, i, o): activations.append(o.detach().cpu())
    def backward_hook(m, gi, go): gradients.append(go[0].detach().cpu())
    fh = target_layer.register_forward_hook(forward_hook)
    try:
        bh = target_layer.register_full_backward_hook(lambda m, gi, go: backward_hook(m, gi, go))
    except:
        bh = target_layer.register_backward_hook(lambda m, gi, go: backward_hook(m, gi, go))
    model.zero_grad()
    out = model(input_tensor)
    score = out[:, 0].sum() if out.shape[1] == 1 else out[0, int(torch.argmax(out, dim=1))]
    score.backward()
    grads, acts = gradients[0].squeeze(0), activations[0].squeeze(0)
    weights = grads.mean(dim=(1, 2))
    cam = np.maximum((weights[:, None, None] * acts).sum(dim=0).numpy(), 0)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

def overlay_heatmap_on_pil(pil_img, cam, alpha=0.4):
    img_resized = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))).astype(np.uint8)
    heatmap = np.uint8(255 * cam)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_resized, 1 - alpha, heatmap_color, alpha, 0)

def resize_for_display(pil_img, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    w, h = pil_img.size
    scale = min(max_width / w, max_height / h)
    return pil_img.resize((int(w * scale), int(h * scale)))

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Receipt Forgery Detector", layout="wide")

theme_dark = st.sidebar.checkbox("🌗 Dark Mode", value=False)
if theme_dark:
    st.markdown("<style>body, .stApp {background-color:#0e1117;color:white;}</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>body, .stApp {background-color:white;color:black;}</style>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center;'>🧾 Receipt Forgery Detector</h2>", unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload receipt image(s)", type=["png","jpg","jpeg"], accept_multiple_files=True)
if not uploaded_files: st.stop()

model = load_model()
if model is None: st.stop()

for i, uploaded in enumerate(uploaded_files):
    pil_img = Image.open(uploaded).convert("RGB")
    resized_img = resize_for_display(pil_img)
    input_tensor = pil_to_tensor(pil_img)
    label, confidence, raw_out = predict_single(model, input_tensor)

    # 3-column layout: image | gradcam | gauge
    col1, col2, col3 = st.columns([1.3, 1.3, 0.8], gap="small")
    with col1:
        st.image(resized_img, caption=f"Uploaded Receipt ({label})", use_container_width=False)

    with col2:
        with st.spinner("Generating Grad-CAM..."):
            cam = compute_gradcam(model, input_tensor)
            overlay = overlay_heatmap_on_pil(pil_img, cam)
            overlay_resized = Image.fromarray(overlay).resize(resized_img.size)
            st.image(overlay_resized, caption="Grad-CAM Overlay", use_container_width=False)

    with col3:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=confidence*100,
            title={'text': "Confidence (%)"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#2ecc71" if "GENUINE" in label else "#e74c3c"}}
        ))
        st.plotly_chart(fig, use_container_width=True, key=f"gauge_{i}")
