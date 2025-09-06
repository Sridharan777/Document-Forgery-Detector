# app.py
"""
Receipt Forgery Detector — ResNet50 + Grad-CAM + Polished UI
- Two-column layout: uploaded image + Grad-CAM overlay side-by-side
- Confidence bar + Plotly gauge (unique keys)
- Sidebar with model info and secret-driven model download
- Download overlay + original image as ZIP
- Robust model loader (handles several checkpoint formats)
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
MODEL_PATH = "models/best_resnet50.pth"            # model file inside repo (preferred)
FALLBACK_GDRIVE_ID = "1zMrv6S6rOWyiTQ0Fgw0VG0o0fEnrWzE-"                            # optionally set your fallback Drive file id
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Small style CSS for card-like UI
CARD_CSS = """
<style>
.result-card {
  border-radius: 12px;
  padding: 12px;
  margin: 8px 0;
  box-shadow: 0 3px 10px rgba(0,0,0,0.08);
}
.result-title {
  font-size: 20px;
  font-weight: 700;
  margin-bottom: 6px;
}
.small-muted { color: #666; font-size:12px; }
.footer { color: #999; font-size:12px; margin-top: 18px; }
</style>
"""

# ---------------- HELPERS ----------------
def download_model_if_missing(gdrive_id: str):
    """Downloads model from Google Drive if MODEL_PATH is missing."""
    if os.path.exists(MODEL_PATH):
        return True
    if not gdrive_id:
        return False
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    try:
        st.info("📥 Downloading model from Google Drive...")
        gdown.download(url, MODEL_PATH, quiet=False)
        # small pause to ensure file appears
        time.sleep(1.0)
        return os.path.exists(MODEL_PATH)
    except Exception as e:
        st.error(f"Model download failed: {e}")
        return False


@st.cache_resource(show_spinner=True)
def load_model():
    """
    Robust model loader:
      - Checks MODEL_PATH in repo first.
      - If missing, tries Streamlit secret MODEL_GDRIVE_ID, then FALLBACK_GDRIVE_ID.
      - Accepts file that is a state_dict or dict containing state_dict keys.
      - Constructs ResNet50 and tries to load weights non-strictly.
    """
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
            # heuristic: if dict looks like weights (contains .weight keys), treat as state_dict
            if state_dict is ckpt and any(k.endswith(".weight") or k.endswith(".bias") for k in ckpt.keys()):
                state_dict = ckpt

        if not isinstance(state_dict, dict):
            st.error("Checkpoint format not recognized (not a state_dict).")
            return None

        # remove 'module.' prefix if present
        new_state = {}
        for k, v in state_dict.items():
            nk = k
            if k.startswith("module."):
                nk = k[len("module."):]
            new_state[nk] = v

        # infer out_features from common fc keys
        out_features = None
        for candidate in ("fc.weight", "classifier.weight", "head.weight"):
            if candidate in new_state:
                out_features = new_state[candidate].shape[0]
                break
        if out_features is None:
            out_features = 2  # assume 2-class default

        # build resnet50
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, out_features)

        load_result = model.load_state_dict(new_state, strict=False)
        if hasattr(load_result, "missing_keys") and load_result.missing_keys:
            st.warning(f"⚠️ Missing keys when loading model: {load_result.missing_keys[:8]}")
        if hasattr(load_result, "unexpected_keys") and load_result.unexpected_keys:
            st.warning(f"⚠️ Unexpected keys when loading model: {load_result.unexpected_keys[:8]}")

        model.to(DEVICE)
        model.eval()
        st.success("✅ Model loaded (ResNet50).")
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
        # single-logit
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
    """
    Manual Grad-CAM:
    - registers hooks on target_layer (default: last block of layer4)
    - returns normalized cam HxW in range 0..1 resized to IMG_SIZE
    """
    activations, gradients = [], []

    if target_layer is None:
        target_layer = model.layer4[-1]

    def forward_hook(module, inp, out):
        activations.append(out.detach().cpu())

    def backward_hook(module, grad_in, grad_out):
        # grad_out is a tuple; we want the first element
        gradients.append(grad_out[0].detach().cpu())

    fh = target_layer.register_forward_hook(forward_hook)
    try:
        bh = target_layer.register_full_backward_hook(lambda m, gi, go: backward_hook(m, gi, go))
    except Exception:
        bh = target_layer.register_backward_hook(lambda m, gi, go: backward_hook(m, gi, go))

    model.zero_grad()
    out = model(input_tensor)

    # choose scalar to backprop: predicted logit
    if out.shape[1] == 1:
        score = out[:, 0].sum()
    else:
        pred_idx = int(torch.argmax(out, dim=1)[0].item())
        score = out[0, pred_idx]

    score.backward()

    try:
        grads = gradients[0].squeeze(0)   # [C, H, W]
        acts  = activations[0].squeeze(0) # [C, H, W]
    except Exception as e:
        fh.remove(); bh.remove()
        st.error(f"Grad-CAM hook failed: {e}")
        return None

    weights = grads.mean(dim=(1, 2))     # [C]
    cam = (weights[:, None, None] * acts).sum(dim=0).numpy()  # [H, W]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    fh.remove(); bh.remove()
    return cam


def overlay_heatmap_on_pil(pil_img: Image.Image, cam: np.ndarray, alpha=0.4):
    img_resized = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))).astype(np.uint8)
    heatmap = np.uint8(255 * cam)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_resized, 1.0 - alpha, heatmap_color, alpha, 0)
    return overlay


# ---------------- UTIL: zip download ----------------
def make_zip_download(original_pil, overlay_arr, filename_prefix):
    mem_zip = BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # original image
        buf1 = BytesIO()
        original_pil.save(buf1, format="PNG")
        zf.writestr(f"{filename_prefix}_original.png", buf1.getvalue())

        # overlay image (numpy array -> PIL)
        buf2 = BytesIO()
        Image.fromarray(overlay_arr).save(buf2, format="PNG")
        zf.writestr(f"{filename_prefix}_gradcam.png", buf2.getvalue())

    mem_zip.seek(0)
    return mem_zip.getvalue()


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Receipt Forgery Detector", layout="wide")
st.markdown(CARD_CSS, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🧾 Receipt Forgery Detector")
st.sidebar.write("ResNet50 + Grad-CAM explainability")
st.sidebar.write("Upload receipts and the app will predict Genuine vs Forged.")
st.sidebar.markdown("---")
st.sidebar.header("Model")
st.sidebar.write(f"Model file: `{MODEL_PATH}`")
st.sidebar.write("- If missing, set `MODEL_GDRIVE_ID` in Streamlit secrets or add the model in `models/`.")
if FALLBACK_GDRIVE_ID:
    st.sidebar.write("- Fallback Drive id configured.")
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.write("Built with PyTorch, torchvision, Streamlit and Grad-CAM.")
st.sidebar.write("Author: You — add your name/links here.")
st.sidebar.markdown("---")
theme_dark = st.sidebar.checkbox("Use dark accents (UI only)", value=False)
st.sidebar.write("Tip: Try both genuine and forged receipts to test Grad-CAM.")

st.title("🧾 Receipt Forgery Detector — ResNet50 (Explainable)")
st.caption("Upload a receipt and see prediction, model confidence and Grad-CAM attention.")

# File uploader
uploaded_files = st.file_uploader("Upload receipt image(s) (png/jpg)", type=["png","jpg","jpeg"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Upload one or more receipt images to run detection.")
    st.stop()

# Load model
with st.spinner("Loading model..."):
    model = load_model()
if model is None:
    st.error("Model failed to load. Stop here and check logs/secrets.")
    st.stop()

# Options
col_options, _ = st.columns([1, 3])
with col_options:
    show_heatmap = st.checkbox("Show Grad-CAM heatmap", value=True)
    show_gauge = st.checkbox("Show Confidence Gauge", value=True)
    show_bar = st.checkbox("Show Confidence bar", value=True)

# Process each uploaded file in sequence
for i, uploaded in enumerate(uploaded_files):
    try:
        pil_img = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error(f"Cannot open {uploaded.name}. Upload a valid image.")
        continue

    # layout: two columns (image + heatmap)
    col1, col2 = st.columns([1, 1])

    # prediction
    input_tensor = pil_to_tensor(pil_img)
    label, confidence, raw_out = predict_single(model, input_tensor)

    # stylize result card colors
    is_genuine = "GENUINE" in label
    result_color = "#dff3e6" if is_genuine else "#fde6e6"
    accent = "#2ecc71" if is_genuine else "#e74c3c"
    text_color = "#085f2a" if is_genuine else "#5a1313"

    # Left column: uploaded image + basic info
    with col1:
        st.markdown(f"<div class='result-card' style='background:{result_color};'>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-title' style='color:{text_color};'>Prediction: {label}</div>", unsafe_allow_html=True)
        st.image(pil_img, caption="Uploaded receipt", width='stretch')
        st.markdown(f"<div class='small-muted'>Model output (raw): {raw_out.shape}</div>", unsafe_allow_html=True)

        # Confidence visuals
        if show_bar:
            # confidence bar
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
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': accent},
                    'steps': [
                        {'range': [0, 50], 'color': "lightcoral"},
                        {'range': [50, 100], 'color': "lightgreen"},
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True, key=f"gauge_{i}_{uploaded.name}")

        st.markdown("</div>", unsafe_allow_html=True)  # close card

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
                    overlay = overlay_heatmap_on_pil(pil_img, cam, alpha=0.4)
                    st.image(overlay, caption="Grad-CAM Overlay (attention)", width='stretch')

                    # single PNG download
                    buf = BytesIO()
                    Image.fromarray(overlay).save(buf, format="PNG")
                    buf.seek(0)
                    st.download_button("Download overlay PNG", data=buf.getvalue(),
                                       file_name=f"heatmap_{uploaded.name}.png", mime="image/png")

                    # ZIP download (original + overlay)
                    zip_bytes = make_zip_download(pil_img, overlay, os.path.splitext(uploaded.name)[0])
                    st.download_button("Download original + overlay (ZIP)", data=zip_bytes,
                                       file_name=f"images_{os.path.splitext(uploaded.name)[0]}.zip", mime="application/zip")
        else:
            st.info("Enable 'Show Grad-CAM heatmap' to see model attention.")

        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>Built with ❤️ — add your name and GitHub link in the sidebar.</div>", unsafe_allow_html=True)
