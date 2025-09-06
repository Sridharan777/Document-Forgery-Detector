# app.py
"""
Receipt Forgery Detector — ResNet50 + Grad-CAM + Modern SaaS UI
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import os
import traceback
import gdown
import plotly.graph_objects as go
import zipfile
import time

# ---------------- CONFIG ----------------
IMG_SIZE = 224
MODEL_PATH = "models/best_resnet50.pth"
FALLBACK_GDRIVE_ID = "1zMrv6S6rOWyiTQ0Fgw0VG0o0fEnrWzE-"  # optional fallback
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CARD_CSS = """
<style>
.result-card {
  border-radius: 14px;
  padding: 14px;
  margin: 10px 0;
  background-color: white;
  box-shadow: 0 4px 14px rgba(0,0,0,0.06);
}
.result-title {
  font-size: 20px;
  font-weight: 700;
  margin-bottom: 6px;
}
.footer { color: #999; font-size:12px; margin-top: 18px; }
</style>
"""

# ---------------- HELPERS ----------------
def download_model_if_missing(gdrive_id: str):
    if os.path.exists(MODEL_PATH):
        return True
    if not gdrive_id:
        return False
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
            ok = download_model_if_missing(drive_id) if drive_id else False
            if not ok:
                ok = download_model_if_missing(FALLBACK_GDRIVE_ID)
            if not ok:
                st.error("❌ Model not found and no Google Drive ID worked.")
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
                out_features = new_state[cand].shape[0]
                break
        if out_features is None:
            out_features = 2

        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, out_features)
        model.load_state_dict(new_state, strict=False)
        model.to(DEVICE)
        model.eval()
        st.success("✅ Model loaded successfully.")
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

def predict_single(model: nn.Module, input_tensor: torch.Tensor):
    model.eval()
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
    if target_layer is None:
        target_layer = model.layer4[-1]

    def forward_hook(m, i, o): activations.append(o.detach().cpu())
    def backward_hook(m, gi, go): gradients.append(go[0].detach().cpu())

    fh = target_layer.register_forward_hook(forward_hook)
    try:
        bh = target_layer.register_full_backward_hook(backward_hook)
    except Exception:
        bh = target_layer.register_backward_hook(backward_hook)

    model.zero_grad()
    out = model(input_tensor)
    score = out[:, 0].sum() if out.shape[1] == 1 else out[0, int(torch.argmax(out, 1))]
    score.backward()

    if not activations or not gradients:
        fh.remove(); bh.remove()
        return None

    acts = activations[0].squeeze(0)
    grads = gradients[0].squeeze(0)
    weights = grads.mean(dim=(1, 2))
    cam = (weights[:, None, None] * acts).sum(dim=0).numpy()
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    fh.remove(); bh.remove()
    return cam

def overlay_heatmap(pil_img, cam, alpha=0.4):
    img = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))).astype(np.uint8)
    if cam is None or np.isnan(cam).any() or cam.size == 0:
        return img
    cam = np.nan_to_num(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heatmap = np.uint8(cam * 255)
    try:
        heat = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(img, 1 - alpha, heat, alpha, 0)
    except cv2.error:
        return img

def make_zip_download(original_pil, overlay_arr, filename_prefix):
    mem_zip = BytesIO()
    with zipfile.ZipFile(mem_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        buf1 = BytesIO()
        original_pil.save(buf1, format="PNG")
        zf.writestr(f"{filename_prefix}_original.png", buf1.getvalue())

        buf2 = BytesIO()
        Image.fromarray(overlay_arr).save(buf2, format="PNG")
        zf.writestr(f"{filename_prefix}_gradcam.png", buf2.getvalue())
    mem_zip.seek(0)
    return mem_zip.getvalue()

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Receipt Forgery Detector", layout="wide")
st.markdown(CARD_CSS, unsafe_allow_html=True)

st.sidebar.title("🧾 Receipt Forgery Detector")
st.sidebar.markdown("Upload receipts, get predictions, and visualize Grad-CAM.")
st.sidebar.markdown("---")
st.sidebar.header("Model Info")
st.sidebar.write(f"Model file: `{MODEL_PATH}`")
st.sidebar.markdown("---")
theme_dark = st.sidebar.checkbox("🌙 Dark Mode", value=False)

st.title("📊 Receipt Forgery Detector — ResNet50")
st.caption("AI-powered detection with confidence gauge + Grad-CAM attention")

uploaded_files = st.file_uploader("Upload receipt image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Upload one or more receipt images to start.")
    st.stop()

with st.spinner("Loading model..."):
    model = load_model()
if model is None:
    st.stop()

show_heatmap = st.checkbox("Show Grad-CAM", value=True)
show_gauge = st.checkbox("Show Confidence Gauge", value=True)
show_bar = st.checkbox("Show Confidence Bar", value=True)

for i, uploaded in enumerate(uploaded_files):
    pil_img = Image.open(uploaded).convert("RGB")
    input_tensor = pil_to_tensor(pil_img)
    label, confidence, raw_out = predict_single(model, input_tensor)

    col1, col2 = st.columns([1, 1])
    with col1:
        result_color = "#dff3e6" if "GENUINE" in label else "#fde6e6"
        st.markdown(f"<div class='result-card' style='background:{result_color};'>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-title'>Prediction: {label}</div>", unsafe_allow_html=True)
        st.image(pil_img, caption="Uploaded Receipt", use_container_width=True)

        if show_bar:
            color = "#2ecc71" if "GENUINE" in label else "#e74c3c"
            st.markdown(f"""
            <div style="width:100%; background:#eee; border-radius:8px; margin:6px 0;">
              <div style="width:{confidence*100:.2f}%; background:{color}; padding:6px; 
                          border-radius:8px; text-align:center; color:white; font-weight:600;">
                {confidence*100:.2f}%
              </div>
            </div>
            """, unsafe_allow_html=True)

        if show_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence*100,
                title={'text': "Confidence (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#2ecc71" if "GENUINE" in label else "#e74c3c"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightcoral"},
                        {'range': [50, 100], 'color': "lightgreen"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True, key=f"gauge_{i}")

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown("<div class='result-title'>Grad-CAM Heatmap</div>", unsafe_allow_html=True)
        if show_heatmap:
            cam = compute_gradcam(model, input_tensor)
            overlay = overlay_heatmap(pil_img, cam)
            st.image(overlay, caption="Model Attention", use_container_width=True)

            zip_bytes = make_zip_download(pil_img, overlay, os.path.splitext(uploaded.name)[0])
            st.download_button("⬇ Download Original + Heatmap (ZIP)", data=zip_bytes,
                               file_name=f"images_{os.path.splitext(uploaded.name)[0]}.zip",
                               mime="application/zip")
        else:
            st.info("Enable Grad-CAM to view model attention.")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>Built with ❤️ using Streamlit + PyTorch</div>", unsafe_allow_html=True)
