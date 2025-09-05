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
MODEL_PATH = "models/best_resnet18.pth"
# fallback Drive id (replace with your own or use Streamlit secrets MODEL_GDRIVE_ID)
FALLBACK_GDRIVE_ID = "1yySFeUxgcN0uqiGbenRCIxhKlhgDwQFJ"

# Use CPU in Streamlit cloud (works reliably)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalization (used for ResNet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ---------------- HELPERS: download + load model ----------------
def download_model_if_missing(gdrive_id: str):
    """Download model from Google Drive to MODEL_PATH if it doesn't already exist."""
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
    """
    Robust loader:
      - Downloads model if missing (uses st.secrets['MODEL_GDRIVE_ID'] if set else fallback)
      - Accepts checkpoints saved in several common formats:
          * state_dict (plain)
          * dict with "state_dict" or "model_state_dict"
          * full model object (nn.Module) — will return it directly
      - Infers fc out_features from checkpoint when possible.
      - Loads weights with strict=False to be tolerant.
    Returns model (eval mode) or None with error printed to app.
    """
    try:
        # determine drive id (prefer secret)
        drive_id = None
        try:
            drive_id = st.secrets.get("MODEL_GDRIVE_ID", None)
        except Exception:
            drive_id = None
        if drive_id is None:
            drive_id = FALLBACK_GDRIVE_ID

        ok = download_model_if_missing(drive_id)
        if not ok:
            st.error("Model not available and download failed. Please upload model to repo or set MODEL_GDRIVE_ID secret.")
            return None

        ckpt = torch.load(MODEL_PATH, map_location="cpu")

        # If ckpt is an nn.Module object (rare), return it directly
        if isinstance(ckpt, nn.Module):
            model_obj = ckpt
            model_obj.eval()
            return model_obj

        # Extract state_dict
        state_dict = None
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model_state_dict", "net", "state"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state_dict = ckpt[key]
                    break
            if state_dict is None:
                # Heuristic: if keys look like weights (contain .weight/.bias), treat ckpt as state_dict
                if any(k.endswith(".weight") or k.endswith(".bias") for k in ckpt.keys()):
                    state_dict = ckpt

        if state_dict is None:
            st.error("No state_dict found in checkpoint (and it's not a model object). Please check your checkpoint.")
            return None

        # Strip 'module.' prefix if present (DataParallel)
        new_state = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state[k[len("module."):]] = v
            else:
                new_state[k] = v

        # Try to infer out_features from usual keys
        out_features = None
        for candidate in ("fc.weight", "classifier.weight", "head.weight"):
            if candidate in new_state:
                out_features = new_state[candidate].shape[0]
                break
        if out_features is None:
            # fallback to 1 (single-logit)
            out_features = 1

        # Construct ResNet18
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, out_features)

        # Load weights non-strictly to avoid small mismatches
        load_result = model.load_state_dict(new_state, strict=False)
        if load_result.missing_keys:
            st.warning(f"Missing keys (first 10): {load_result.missing_keys[:10]}")
        if load_result.unexpected_keys:
            st.warning(f"Unexpected keys (first 10): {load_result.unexpected_keys[:10]}")

        model.to(DEVICE)
        model.eval()
        st.success("Model loaded successfully.")
        return model

    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.text(traceback.format_exc())
        return None


# ---------------- PREPROCESS ----------------
def pil_to_tensor(img_pil: Image.Image):
    """
    Convert PIL image to normalized tensor [1,C,H,W] for ResNet
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(img_pil).unsqueeze(0).to(DEVICE)


# ---------------- PREDICTION ----------------
def predict_single(model: nn.Module, input_tensor: torch.Tensor):
    """
    Returns (label_str, confidence_float, raw_output_tensor)
    For single-output model (out_features == 1) uses sigmoid and returns prob-> forged.
    For multi-output model uses softmax and returns argmax.
    """
    model.eval()
    with torch.no_grad():
        out = model(input_tensor)  # shape [1, out_features]
        out = out.to(torch.float32)
        if out.shape[1] == 1:
            prob = torch.sigmoid(out)[0, 0].item()
            # interpret probability as "forged probability"
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
    """
    Manual Grad-CAM:
      - hooks activations and gradients on target_layer
      - returns normalized cam (HxW, values 0..1) resized to IMG_SIZE
    """
    activations = []
    gradients = []

    if target_layer is None:
        # Default: last block of layer4
        target_layer = model.layer4[-1]

    def forward_hook(module, inp, out):
        activations.append(out.detach().cpu())

    # try register_full_backward_hook; fall back to register_backward_hook if necessary
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach().cpu())

    fh = target_layer.register_forward_hook(forward_hook)
    # prefer full backward hook (more correct)
    try:
        bh = target_layer.register_full_backward_hook(lambda module, grad_in, grad_out: backward_hook(module, grad_in, grad_out))
    except Exception:
        bh = target_layer.register_backward_hook(lambda module, grad_in, grad_out: backward_hook(module, grad_in, grad_out))

    model.zero_grad()
    out = model(input_tensor)  # forward

    # choose scalar for backprop: predicted logit or single logit
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
    """
    Overlay normalized cam (HxW, 0..1) onto the PIL RGB image (resizing to IMG_SIZE).
    Returns uint8 numpy RGB image (IMG_SIZE x IMG_SIZE x 3).
    """
    img_resized = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))).astype(np.uint8)
    heatmap = np.uint8(255 * cam)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_resized, 1.0 - alpha, heatmap_color, alpha, 0)
    return overlay


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Receipt Forgery Detector", layout="centered")
st.title("🧾 Receipt Forgery Detector — Single-logit model (explainable)")

st.sidebar.header("Model / App info")
st.sidebar.write("• Model: ResNet18 (single-logit expected).")
st.sidebar.write("• If your checkpoint uses two outputs, re-train or provide that checkpoint.")
st.sidebar.write("• Set `MODEL_GDRIVE_ID` in Streamlit secrets to point to your model (optional).")

# file uploader (single or multiple)
uploaded_files = st.file_uploader("Upload receipt image(s) (png/jpg)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Upload one or more receipt images to run detection.")
else:
    model = load_model()
    if model is None:
        st.error("Model failed to load. Check logs/Streamlit secrets.")
        st.stop()

    # UI options
    show_heatmap = st.checkbox("Show Grad-CAM heatmap", value=True)
    show_gauge = st.checkbox("Show Plotly gauge", value=True)

    for uploaded in uploaded_files:
        try:
            pil_img = Image.open(uploaded).convert("RGB")
        except Exception:
            st.error("Cannot open image. Upload valid png/jpg file.")
            continue

        st.subheader(f"📄 {uploaded.name}")
        st.image(pil_img, caption="Uploaded receipt", width="stretch")

        # preprocess
        input_tensor = pil_to_tensor(pil_img)

        # predict
        label, confidence, raw_out = predict_single(model, input_tensor)
        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Confidence:** {confidence*100:.2f}%")

        # confidence bar (HTML)
        color = "#2ecc71" if "GENUINE" in label else "#e74c3c"
        bar_html = f"""
        <div style="width:100%; background:#eee; border-radius:8px; margin:6px 0;">
          <div style="width:{confidence*100:.2f}%; background:{color}; padding:6px 4px;
                      border-radius:8px; text-align:center; color:white; font-weight:600;">
            {confidence*100:.2f}%
          </div>
        </div>
        """
        st.markdown(bar_html, unsafe_allow_html=True)

        # optional gauge
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

        # Grad-CAM
        if show_heatmap:
            with st.spinner("Generating Grad-CAM..."):
                cam = compute_gradcam(model, input_tensor, target_layer=None)
                if cam is None:
                    st.warning("Could not generate Grad-CAM for this image.")
                else:
                    overlay = overlay_heatmap_on_pil(pil_img, cam, alpha=0.4)
                    st.image(overlay, caption="Model attention (Grad-CAM)", width="stretch")

                    # Download heatmap
                    buf = BytesIO()
                    Image.fromarray(overlay).save(buf, format="PNG")
                    buf.seek(0)
                    st.download_button("Download heatmap PNG", data=buf.getvalue(), file_name=f"heatmap_{uploaded.name}.png", mime="image/png")
