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
MODEL_PATH = "models/best_resnet50.pth"  # <---- UPDATED TO RESNET50
# fallback Drive id (replace with your own or use Streamlit secrets MODEL_GDRIVE_ID)
FALLBACK_GDRIVE_ID = "1zMrv6S6rOWyiTQ0Fgw0VG0o0fEnrWzE-"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------- HELPERS ----------------
def download_model_if_missing(gdrive_id: str):
    if os.path.exists(MODEL_PATH):
        return True
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    try:
        st.info("Downloading model from Google Drive...")
        gdown.download(url, MODEL_PATH, quiet=False)
        return os.path.exists(MODEL_PATH)
    except Exception as e:
        st.error(f"Model download failed: {e}")
        return False

@st.cache_resource(show_spinner=True)
def load_model():
    try:
        # Use secret if available, else fallback
        drive_id = None
        try:
            drive_id = st.secrets.get("MODEL_GDRIVE_ID", None)
        except Exception:
            drive_id = None
        if drive_id is None:
            drive_id = FALLBACK_GDRIVE_ID

        ok = download_model_if_missing(drive_id)
        if not ok:
            st.error("Model not available and download failed.")
            return None

        ckpt = torch.load(MODEL_PATH, map_location="cpu")

        state_dict = None
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model_state_dict", "net", "state"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state_dict = ckpt[key]
                    break
            if state_dict is None and any(k.endswith(".weight") for k in ckpt.keys()):
                state_dict = ckpt

        if state_dict is None:
            st.error("No state_dict found in checkpoint")
            return None

        # Remove "module." prefix if needed
        new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}

        out_features = None
        for candidate in ("fc.weight", "classifier.weight", "head.weight"):
            if candidate in new_state:
                out_features = new_state[candidate].shape[0]
                break
        if out_features is None:
            out_features = 2  # Default to 2-class output for forged/genuine

        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, out_features)

        load_result = model.load_state_dict(new_state, strict=False)
        if load_result.missing_keys:
            st.warning(f"Missing keys: {load_result.missing_keys[:5]}")
        if load_result.unexpected_keys:
            st.warning(f"Unexpected keys: {load_result.unexpected_keys[:5]}")

        model.to(DEVICE)
        model.eval()
        st.success("ResNet50 model loaded successfully!")
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

def compute_gradcam(model: nn.Module, input_tensor: torch.Tensor, target_layer=None):
    activations = []
    gradients = []

    if target_layer is None:
        target_layer = model.layer4[-1]

    def forward_hook(module, inp, out):
        activations.append(out.detach().cpu())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach().cpu())

    fh = target_layer.register_forward_hook(forward_hook)
    try:
        bh = target_layer.register_full_backward_hook(lambda m, gi, go: backward_hook(m, gi, go))
    except Exception:
        bh = target_layer.register_backward_hook(lambda m, gi, go: backward_hook(m, gi, go))

    model.zero_grad()
    out = model(input_tensor)
    if out.shape[1] == 1:
        score = out[:, 0].sum()
    else:
        pred_idx = int(torch.argmax(out, dim=1)[0].item())
        score = out[0, pred_idx]

    score.backward()
    grads = gradients[0].squeeze(0)
    acts = activations[0].squeeze(0)
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
st.title("🧾 Receipt Forgery Detector — ResNet50 (Explainable)")

st.sidebar.header("Model / App info")
st.sidebar.write("• Model: ResNet50 with 2 outputs (Genuine/Forged).")
st.sidebar.write("• Set `MODEL_GDRIVE_ID` in Streamlit secrets if needed.")

uploaded_files = st.file_uploader("Upload receipt image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Upload one or more receipt images to run detection.")
else:
    model = load_model()
    if model is None:
        st.error("Model failed to load.")
        st.stop()

    show_heatmap = st.checkbox("Show Grad-CAM heatmap", value=True)
    show_gauge = st.checkbox("Show Confidence Gauge", value=True)

    for uploaded in uploaded_files:
        try:
            pil_img = Image.open(uploaded).convert("RGB")
        except Exception:
            st.error("Cannot open image.")
            continue

        st.subheader(f"📄 {uploaded.name}")
        st.image(pil_img, caption="Uploaded receipt", width="stretch")

        input_tensor = pil_to_tensor(pil_img)
        label, confidence, raw_out = predict_single(model, input_tensor)

        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Confidence:** {confidence*100:.2f}%")

        color = "#2ecc71" if "GENUINE" in label else "#e74c3c"

