# app.py
"""
Receipt Forgery Detector — ResNet50 + Grad-CAM + Polished UI (Dark/Light toggle fixed)
Replace previous app.py with this file. Ensure requirements.txt contains:
streamlit, torch, torchvision, numpy, opencv-python-headless, pillow, gdown, plotly
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
# put your fallback google drive id (optional) or set MODEL_GDRIVE_ID in Streamlit secrets
FALLBACK_GDRIVE_ID = "1zMrv6S6rOWyiTQ0Fgw0VG0o0fEnrWzE-"  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ---------------- HELPERS ----------------
def download_model_if_missing(gdrive_id: str):
    """Download model from Google Drive if MODEL_PATH missing."""
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
    """Robust loader: prefer local MODEL_PATH, fall back to secrets/fallback drive id."""
    try:
        # determine Drive ID (secret > fallback)
        try:
            drive_id = st.secrets.get("MODEL_GDRIVE_ID", None)
        except Exception:
            drive_id = None

        if not os.path.exists(MODEL_PATH):
            drive_ok = False
            if drive_id:
                drive_ok = download_model_if_missing(drive_id)
            if not drive_ok and FALLBACK_GDRIVE_ID:
                drive_ok = download_model_if_missing(FALLBACK_GDRIVE_ID)
            if not drive_ok and not os.path.exists(MODEL_PATH):
                st.error("Model not found locally and no valid Drive id available.")
                return None

        ckpt = torch.load(MODEL_PATH, map_location="cpu")

        # resolve state_dict
        state_dict = ckpt
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model_state_dict", "net", "state"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state_dict = ckpt[key]
                    break
            if state_dict is ckpt and any(k.endswith(".weight") or k.endswith(".bias") for k in ckpt.keys()):
                state_dict = ckpt

        if not isinstance(state_dict, dict):
            st.error("Checkpoint format not recognized (not a state_dict).")
            return None

        # strip 'module.' prefix
        new_state = {}
        for k, v in state_dict.items():
            nk = k[len("module."):] if k.startswith("module.") else k
            new_state[nk] = v

        # infer out_features
        out_features = None
        for candidate in ("fc.weight", "classifier.weight", "head.weight"):
            if candidate in new_state:
                out_features = new_state[candidate].shape[0]
                break
        if out_features is None:
            out_features = 2  # default to 2-class

        # build resnet50 with matching head
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, out_features)

        load_result = model.load_state_dict(new_state, strict=False)
        # show small warnings if keys mismatched
        if hasattr(load_result, "missing_keys") and load_result.missing_keys:
            st.warning(f"⚠️ Missing keys when loading model: {load_result.missing_keys[:8]}")
        if hasattr(load_result, "unexpected_keys") and load_result.unexpected_keys:
            st.warning(f"⚠️ Unexpected keys when loading model: {load_result.unexpected_keys[:8]}")

        model.to(DEVICE)
        model.eval()
        st.success(f"✅ Model loaded (ResNet50) — outputs: {out_features}")
        return model

    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.text(traceback.format_exc())
        return None

# ---------------- PREPROCESS ----------------
def pil_to_tensor(img_pil: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(img_pil).unsqueeze(0).to(DEVICE)

# ---------------- PREDICTION ----------------
def predict_single(model: nn.Module, input_tensor: torch.Tensor):
    model.eval()
    with torch.no_grad():
        out = model(input_tensor)
        out = out.to(torch.float32)
        if out.shape[1] == 1:
            prob = torch.sigmoid(out)[0, 0].item()
            if prob >= 0.5:
                label = "FORGED 🔴"
                confidence = prob
            else:
                label = "GENUINE 🟢"
                confidence = 1.0 - prob
        else:
            probs = torch.softmax(out, dim=1)[0]
            idx = int(torch.argmax(probs).item())
            label = "FORGED 🔴" if idx == 1 else "GENUINE 🟢"
            confidence = float(probs[idx].item())
    return label, float(confidence), out

# ---------------- GRAD-CAM ----------------
def compute_gradcam(model: nn.Module, input_tensor: torch.Tensor, target_layer=None):
    activations, gradients = [], []
    if target_layer is None:
        target_layer = model.layer4[-1]

    def forward_hook(module, inp, out):
        activations.append(out.detach().cpu())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach().cpu())

    fh = target_layer.register_forward_hook(forward_hook)
    try:
        bh = target_layer.register_full_backward_hook(backward_hook)
    except Exception:
        bh = target_layer.register_backward_hook(backward_hook)

    model.zero_grad()
    out = model(input_tensor)

    # pick a scalar to backprop: predicted logit (works for 1 or 2 outputs)
    if out.shape[1] == 1:
        score = out[:, 0].sum()
    else:
        pred_idx = int(torch.argmax(out, dim=1)[0].item())
        score = out[0, pred_idx]

    score.backward()

    try:
        grads = gradients[0].squeeze(0)
        acts = activations[0].squeeze(0)
    except Exception as e:
        fh.remove(); bh.remove()
        st.error(f"Grad-CAM hook failed: {e}")
        return None

    weights = grads.mean(dim=(1, 2))
    cam = (weights[:, None, None] * acts).sum(dim=0).numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    fh.remove(); bh.remove()
    return cam

def overlay_heatmap_on_pil(pil_img: Image.Image, cam: np.ndarray, cmap_name="JET", alpha=0.4):
    img_resized = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))).astype(np.uint8)
    # pick colormap safely
    cmap = getattr(cv2, f"COLORMAP_{cmap_name.upper()}", cv2.COLORMAP_JET)
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cmap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_resized, 1.0 - alpha, heatmap, alpha, 0)
    return overlay

# ---------------- UTIL: zip download ----------------
def make_zip_download(original_pil, overlay_arr, filename_prefix):
    mem_zip = BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        buf1 = BytesIO()
        original_pil.save(buf1, format="PNG")
        zf.writestr(f"{filename_prefix}_original.png", buf1.getvalue())
        buf2 = BytesIO()
        Image.fromarray(overlay_arr).save(buf2, format="PNG")
        zf.writestr(f"{filename_prefix}_gradcam.png", buf2.getvalue())
    mem_zip.seek(0)
    return mem_zip.getvalue()

# ---------------- UI & CSS ----------------
BASE_CARD_CSS = """
<style>
.result-card {
  border-radius: 12px;
  padding: 12px;
  margin: 8px 0;
  box-shadow: 0 3px 10px rgba(0,0,0,0.06);
}
.result-title { font-size:18px; font-weight:700; margin-bottom:6px; }
.small-muted { color: #666; font-size:12px; }
.footer { color:#999; font-size:12px; margin-top:12px; }
</style>
"""

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Receipt Forgery Detector", layout="wide")
st.markdown(BASE_CARD_CSS, unsafe_allow_html=True)

# Sidebar content
st.sidebar.title("🧾 Receipt Forgery Detector")
st.sidebar.write("ResNet50 + Grad-CAM explainability")
st.sidebar.markdown("---")
st.sidebar.header("Model")
st.sidebar.write(f"Model path: `{MODEL_PATH}`")
st.sidebar.write("- If missing, set MODEL_GDRIVE_ID in Streamlit secrets or put model in `models/`.")
if FALLBACK_GDRIVE_ID:
    st.sidebar.write("- Fallback Drive id configured.")
st.sidebar.markdown("---")
st.sidebar.header("UI / Display")
theme_dark = st.sidebar.checkbox("Use dark accents (UI only)", value=False)
colormap_choice = st.sidebar.selectbox("Grad-CAM colormap", ["JET", "VIRIDIS", "HOT", "BONE"])
st.sidebar.write("Tip: Try both genuine and forged receipts to test Grad-CAM.")
st.sidebar.markdown("---")
st.sidebar.write(f"Device: `{DEVICE}`")

st.title("🧾 Receipt Forgery Detector — ResNet50 (Explainable)")
st.caption("Upload a receipt and see prediction, confidence and Grad-CAM.")

# Dark/light UI styles (card bg/text)
if theme_dark:
    # darker card background and lighter text
    st.markdown("""
    <style>
    .result-card { background:#111217; color:#e6eef8; box-shadow: 0 3px 12px rgba(0,0,0,0.6); }
    .result-title { color:#dbe9ff; }
    .small-muted { color:#aeb7c3; }
    .footer { color:#9aa3af; }
    </style>
    """, unsafe_allow_html=True)
else:
    # light theme default; shadow already in BASE_CARD_CSS
    st.markdown("""
    <style>
    .result-card { background:#ffffff; color:#0b2436; }
    .result-title { color:#0b2436; }
    </style>
    """, unsafe_allow_html=True)

# File uploader
uploaded_files = st.file_uploader("Upload receipt image(s) (png/jpg)", type=["png","jpg","jpeg"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Upload one or more receipt images to run detection.")
    st.stop()

# Load model (spinner)
with st.spinner("Loading model..."):
    model = load_model()
if model is None:
    st.error("Model failed to load. Check logs / secrets / model file.")
    st.stop()

# Per-run options
col_options, _ = st.columns([1, 3])
with col_options:
    show_heatmap = st.checkbox("Show Grad-CAM heatmap", value=True)
    show_gauge = st.checkbox("Show Confidence Gauge", value=True)
    show_bar = st.checkbox("Show Confidence bar", value=True)

# Process uploads
for i, uploaded in enumerate(uploaded_files):
    try:
        pil_img = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error(f"Cannot open {uploaded.name}. Upload a valid image.")
        continue

    # two columns: left = uploaded + info, right = grad-cam + downloads
    col1, col2 = st.columns([1, 1])

    # Predict
    input_tensor = pil_to_tensor(pil_img)
    label, confidence, raw_out = predict_single(model, input_tensor)

    is_genuine = "GENUINE" in label
    result_color_light = "#eaf8ee" if is_genuine else "#fff0f0"
    accent = "#2ecc71" if is_genuine else "#e74c3c"
    text_color = "#0b3b20" if is_genuine else "#5a1313"

    # Left column (card)
    with col1:
        st.markdown(f"<div class='result-card' style='background:{result_color_light};'>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-title' style='color:{text_color};'>Prediction: {label}</div>", unsafe_allow_html=True)
        # Use width='stretch' per deprecation guidance so the layout matches
        try:
            st.image(pil_img, caption="Uploaded receipt", width='stretch')
        except Exception:
            # fallback to a numeric width
            st.image(pil_img, caption="Uploaded receipt", width=380)
        st.markdown(f"<div class='small-muted'>Model output (raw): {raw_out.shape}</div>", unsafe_allow_html=True)

        if show_bar:
            bar_html = f"""
            <div style="width:100%; background:#eee; border-radius:8px; margin-top:8px;">
              <div style="width:{confidence*100:.2f}%; background:{accent}; padding:8px; 
                          border-radius:8px; text-align:center; color:white; font-weight:700;">
                {confidence*100:.2f}%
              </div>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)

        if show_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence*100,
                title={'text': "Model Confidence (%)"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': accent},
                       'steps': [{'range': [0, 50], 'color': "lightcoral"},
                                 {'range': [50, 100], 'color': "lightgreen"}]}))
            # unique key for each gauge to avoid duplicate element id
            chart_key = f"gauge_{i}_{os.path.basename(uploaded.name)}"
            # use width='stretch' to fill column
            try:
                st.plotly_chart(fig, width='stretch', use_container_width=False, key=chart_key)
            except Exception:
                # older Streamlit might accept use_container_width
                st.plotly_chart(fig, use_container_width=True, key=chart_key)

        st.markdown("</div>", unsafe_allow_html=True)

    # Right column: Grad-CAM & downloads
    with col2:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown("<div class='result-title'>Model Explainability</div>", unsafe_allow_html=True)

        if show_heatmap:
            with st.spinner("Generating Grad-CAM..."):
                cam = compute_gradcam(model, input_tensor)
                if cam is None:
                    st.warning("Could not generate Grad-CAM for this image.")
                else:
                    # choose colormap based on theme & user choice
                    cmap_choice = colormap_choice
                    # For dark theme, pick a high-contrast colormap if desired:
                    if theme_dark and cmap_choice == "JET":
                        cmap_choice = "HOT"

                    overlay = overlay_heatmap_on_pil(pil_img, cam, cmap_name=cmap_choice, alpha=0.4)

                    try:
                        st.image(overlay, caption="Grad-CAM Overlay (attention)", width='stretch')
                    except Exception:
                        st.image(overlay, caption="Grad-CAM Overlay (attention)", width=380)

                    # single PNG download
                    buf = BytesIO()
                    Image.fromarray(overlay).save(buf, format="PNG")
                    buf.seek(0)
                    st.download_button("Download overlay PNG", data=buf.getvalue(),
                                       file_name=f"heatmap_{os.path.splitext(uploaded.name)[0]}.png", mime="image/png")

                    # ZIP (original + overlay)
                    zip_bytes = make_zip_download(pil_img, overlay, os.path.splitext(uploaded.name)[0])
                    st.download_button("Download original + overlay (ZIP)", data=zip_bytes,
                                       file_name=f"images_{os.path.splitext(uploaded.name)[0]}.zip", mime="application/zip")
        else:
            st.info("Enable 'Show Grad-CAM heatmap' to see model attention.")

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>Built with ❤️ — add your name and GitHub link in the sidebar.</div>", unsafe_allow_html=True)
