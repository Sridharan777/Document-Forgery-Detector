# app.py — Final Polished Version with Dark/Light Mode + Matched Grad-CAM Size
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import os, traceback, gdown, time
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
IMG_SIZE = 224  # only for model inference
MODEL_PATH = "models/best_resnet50.pth"
FALLBACK_GDRIVE_ID = "1zMrv6S6rOWyiTQ0Fgw0VG0o0fEnrWzE-"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------- HELPERS ----------------
def download_model_if_missing(gdrive_id):
    if os.path.exists(MODEL_PATH): return True
    if not gdrive_id: return False
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    try:
        st.info("📥 Downloading model from Google Drive...")
        gdown.download(url, MODEL_PATH, quiet=False)
        time.sleep(1)
        return os.path.exists(MODEL_PATH)
    except Exception as e:
        st.error(f"Model download failed: {e}")
        return False

@st.cache_resource(show_spinner=True)
def load_model():
    try:
        drive_id = None
        try: drive_id = st.secrets.get("MODEL_GDRIVE_ID", None)
        except Exception: pass
        if not os.path.exists(MODEL_PATH):
            ok = download_model_if_missing(drive_id) if drive_id else False
            if not ok: ok = download_model_if_missing(FALLBACK_GDRIVE_ID)
            if not ok: 
                st.error("❌ Model not found.")
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
        new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}

        out_features = None
        for cand in ("fc.weight", "classifier.weight", "head.weight"):
            if cand in new_state:
                out_features = new_state[cand].shape[0]; break
        if out_features is None: out_features = 2

        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, out_features)
        model.load_state_dict(new_state, strict=False)
        model.to(DEVICE); model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.text(traceback.format_exc())
        return None

def pil_to_tensor(img_pil):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(img_pil).unsqueeze(0).to(DEVICE)

def predict_single(model, input_tensor):
    with torch.no_grad():
        out = model(input_tensor)
        if out.shape[1] == 1:
            prob = torch.sigmoid(out)[0, 0].item()
            label = "FORGED 🔴" if prob >= 0.5 else "GENUINE 🟢"
            confidence = prob if prob >= 0.5 else 1 - prob
        else:
            probs = torch.softmax(out, dim=1)[0]
            idx = int(torch.argmax(probs).item())
            label = "FORGED 🔴" if idx == 1 else "GENUINE 🟢"
            confidence = float(probs[idx].item())
    return label, confidence, out

def compute_gradcam(model, input_tensor, target_layer=None):
    activations, gradients = [], []
    if target_layer is None: target_layer = model.layer4[-1]
    def forward_hook(m, i, o): activations.append(o.detach().cpu())
    def backward_hook(m, gi, go): gradients.append(go[0].detach().cpu())
    fh = target_layer.register_forward_hook(forward_hook)
    try: bh = target_layer.register_full_backward_hook(backward_hook)
    except: bh = target_layer.register_backward_hook(backward_hook)
    model.zero_grad()
    out = model(input_tensor)
    score = out[:, 0].sum() if out.shape[1] == 1 else out[0, int(torch.argmax(out, 1))]
    score.backward()
    if not activations or not gradients:
        fh.remove(); bh.remove(); return None
    acts, grads = activations[0].squeeze(0), gradients[0].squeeze(0)
    weights = grads.mean(dim=(1, 2))
    cam = (weights[:, None, None] * acts).sum(dim=0).numpy()
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    fh.remove(); bh.remove()
    return cam

def overlay_heatmap(pil_img, cam, alpha=0.4):
    # resize CAM to match original image size
    img = np.array(pil_img).astype(np.uint8)
    if cam is None: return img
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(cam_resized * 255)
    heat = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 1 - alpha, heat, alpha, 0)

# ---------------- UI ----------------
st.set_page_config(page_title="Receipt Forgery Detector", layout="wide")
st.markdown("<h2 style='text-align:center;'>🧾 Receipt Forgery Detector</h2>", unsafe_allow_html=True)
st.caption("Upload receipts → See prediction, confidence, Grad-CAM overlay")

# 🔘 Dark/Light Theme Toggle
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

# Theme colors
BG_COLOR = "#1e1e1e" if dark_mode else "white"
TEXT_COLOR = "white" if dark_mode else "black"

uploaded_files = st.file_uploader("Upload receipt image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
if not uploaded_files:
    st.info("Upload receipt images to start."); st.stop()

with st.spinner("Loading model..."):
    model = load_model()
if model is None: st.stop()

show_heatmap = st.checkbox("Show Grad-CAM", value=True)
show_gauge = st.checkbox("Show Confidence Gauge", value=True)

for i, uploaded in enumerate(uploaded_files):
    pil_img = Image.open(uploaded).convert("RGB")
    input_tensor = pil_to_tensor(pil_img)
    label, confidence, raw_out = predict_single(model, input_tensor)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"<div style='background:{BG_COLOR}; color:{TEXT_COLOR}; padding:10px; border-radius:10px;'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:{TEXT_COLOR};'>Prediction: {label}</h4>", unsafe_allow_html=True)
        st.image(pil_img, caption="Uploaded Receipt", use_container_width=True)

        if show_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence*100,
                title={'text': "Confidence (%)", 'font': {'size': 16, 'color': TEXT_COLOR}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#2ecc71" if "GENUINE" in label else "#e74c3c"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightcoral"},
                        {'range': [50, 100], 'color': "lightgreen"},
                    ]
                }
            ))
            fig.update_layout(height=230, margin=dict(t=0, b=0))
            st.plotly_chart(fig, use_container_width=True, key=f"gauge_{i}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<div style='background:{BG_COLOR}; color:{TEXT_COLOR}; padding:10px; border-radius:10px;'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:{TEXT_COLOR};'>Grad-CAM</h4>", unsafe_allow_html=True)
        if show_heatmap:
            cam = compute_gradcam(model, input_tensor)
            overlay = overlay_heatmap(pil_img, cam)
            st.image(overlay, caption="Model Attention", use_container_width=True)
        else:
            st.info("Enable Grad-CAM to view attention.")
        st.markdown("</div>", unsafe_allow_html=True)
