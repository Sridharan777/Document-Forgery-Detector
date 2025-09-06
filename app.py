# app.py
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

# ---------------- CONFIG ----------------
IMG_SIZE = 224
MODEL_PATH = "models/best_resnet50.pth"  # ✅ Now using ResNet50
FALLBACK_GDRIVE_ID = "1zMrv6S6rOWyiTQ0Fgw0VG0o0fEnrWzE-"  # Replace if you want

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ---------------- HELPERS: download + load model ----------------
def download_model_if_missing(gdrive_id: str):
    if os.path.exists(MODEL_PATH):
        return True
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    try:
        st.info("📥 Downloading model from Google Drive...")
        gdown.download(url, MODEL_PATH, quiet=False)
        return os.path.exists(MODEL_PATH)
    except Exception as e:
        st.error(f"Model download failed: {e}")
        return False

@st.cache_resource(show_spinner=True)
def load_model():
    try:
        try:
            drive_id = st.secrets.get("MODEL_GDRIVE_ID", None)
        except Exception:
            drive_id = None
        if drive_id is None:
            drive_id = FALLBACK_GDRIVE_ID

        ok = download_model_if_missing(drive_id)
        if not ok:
            st.error("❌ Model not available. Upload to repo or set MODEL_GDRIVE_ID secret.")
            return None

        ckpt = torch.load(MODEL_PATH, map_location="cpu")

        # handle state_dict
        state_dict = ckpt
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model_state_dict", "net", "state"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state_dict = ckpt[key]
                    break

        # strip 'module.' if present
        new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # infer output features
        out_features = 1
        for candidate in ("fc.weight", "classifier.weight", "head.weight"):
            if candidate in new_state:
                out_features = new_state[candidate].shape[0]
                break

        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, out_features)

        load_result = model.load_state_dict(new_state, strict=False)
        if load_result.missing_keys:
            st.warning(f"⚠️ Missing keys: {load_result.missing_keys[:5]}")
        if load_result.unexpected_keys:
            st.warning(f"⚠️ Unexpected keys: {load_result.unexpected_keys[:5]}")

        model.to(DEVICE)
        model.eval()
        st.success("✅ Model loaded successfully (ResNet50).")
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

# ---------------- Manual Grad-CAM ----------------
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
        bh = target_layer.register_full_backward_hook(lambda m, gi, go: backward_hook(m, gi, go))
    except:
        bh = target_layer.register_backward_hook(lambda m, gi, go: backward_hook(m, gi, go))

    model.zero_grad()
    out = model(input_tensor)
    if out.shape[1] == 1:
        score = out[:, 0].sum()
    else:
        score = out[0, int(torch.argmax(out, dim=1)[0])]
    score.backward()

    grads = gradients[0].squeeze(0)
    acts  = activations[0].squeeze(0)
    weights = grads.mean(dim=(1, 2))
    cam = (weights[:, None, None] * acts).sum(dim=0).numpy()
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

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Receipt Forgery Detector", layout="centered")
st.title("🧾 Receipt Forgery Detector — ResNet50 + Grad-CAM + Gauge")

uploaded_files = st.file_uploader("Upload receipt image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Upload one or more receipt images to run detection.")
else:
    model = load_model()
    if model is None:
        st.stop()

    show_heatmap = st.checkbox("Show Grad-CAM heatmap", value=True)
    show_gauge = st.checkbox("Show Confidence Gauge", value=True)

    for uploaded in uploaded_files:
        try:
            pil_img = Image.open(uploaded).convert("RGB")
        except Exception:
            st.error("Cannot open image. Upload valid png/jpg file.")
            continue

        st.subheader(f"📄 {uploaded.name}")
        st.image(pil_img, caption="Uploaded receipt", use_container_width=True)

        input_tensor = pil_to_tensor(pil_img)
        label, confidence, raw_out = predict_single(model, input_tensor)

        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Confidence:** {confidence*100:.2f}%")

        # Confidence Bar
        color = "#2ecc71" if "GENUINE" in label else "#e74c3c"
        bar_html = f"""
        <div style="width:100%; background:#eee; border-radius:8px;">
          <div style="width:{confidence*100:.2f}%; background:{color}; padding:6px; 
                      border-radius:8px; text-align:center; color:white; font-weight:600;">
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
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 50], 'color': "lightcoral"},
                        {'range': [50, 100], 'color': "lightgreen"},
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

        if show_heatmap:
            with st.spinner("Generating Grad-CAM..."):
                cam = compute_gradcam(model, input_tensor)
                overlay = overlay_heatmap_on_pil(pil_img, cam, alpha=0.4)
                st.image(overlay, caption="Grad-CAM Heatmap", use_container_width=True)
                buf = BytesIO()
                Image.fromarray(overlay).save(buf, format="PNG")
                buf.seek(0)
                st.download_button("Download heatmap PNG", data=buf.getvalue(),
                                   file_name=f"heatmap_{uploaded.name}.png", mime="image/png")
