# app.py  (paste entire contents, replacing existing file)
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import os
import gdown
import traceback

# ---------- CONFIG ----------
IMG_SIZE = 224
MODEL_PATH = "models/best_resnet18.pth"
# fallback Drive ID (if you haven't set Streamlit secret MODEL_GDRIVE_ID)
DEFAULT_GDRIVE_ID = "1yySFeUxgcN0uqiGbenRCIxhKlhgDwQFJ"

# ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ---------- HELPERS ----------
def download_model_from_drive(gdrive_id):
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        st.info("Downloading model from Google Drive...")
        try:
            gdown.download(url, MODEL_PATH, quiet=False)
        except Exception as e:
            st.error(f"Download failed: {e}")
            return False
    return True

@st.cache_resource(show_spinner=True)
def load_model():
    """
    Safe loader: downloads model if missing (from st.secrets or DEFAULT_GDRIVE_ID),
    inspects checkpoint to infer output size and loads state_dict (strips module. prefix).
    Returns model (in eval mode) or None on error.
    """
    try:
        # determine drive id (prefer secret)
        gdrive_id = None
        try:
            gdrive_id = st.secrets.get("MODEL_GDRIVE_ID", None)
        except Exception:
            gdrive_id = None
        if gdrive_id is None:
            gdrive_id = DEFAULT_GDRIVE_ID

        if not os.path.exists(MODEL_PATH):
            ok = download_model_from_drive(gdrive_id)
            if not ok:
                st.error("Model download failed. Provide MODEL_GDRIVE_ID in Streamlit secrets or upload model to repo.")
                return None

        ckpt = torch.load(MODEL_PATH, map_location="cpu")

        # extract state_dict if present or consider ckpt itself as state_dict
        state_dict = None
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model_state_dict", "net", "state"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state_dict = ckpt[key]
                    break
            if state_dict is None:
                # If keys look like parameter names, treat ckpt as state_dict
                if any(k.endswith(".weight") or k.endswith(".bias") for k in ckpt.keys()):
                    state_dict = ckpt
        else:
            # ckpt might be a full model object -> attempt to use it directly
            if isinstance(ckpt, nn.Module):
                model_obj = ckpt
                model_obj.eval()
                return model_obj
            else:
                st.error("Unrecognized checkpoint format.")
                return None

        if state_dict is None:
            st.error("Could not find a state_dict in the checkpoint.")
            return None

        # Strip 'module.' prefix from keys if present
        new_state = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state[k[len("module."):]] = v
            else:
                new_state[k] = v

        # Infer output size from fc weight if possible
        out_features = None
        for candidate in ("fc.weight", "classifier.weight", "head.weight"):
            if candidate in new_state:
                out_features = new_state[candidate].shape[0]
                break
        if out_features is None:
            # fallback to 1
            out_features = 1

        # Build model and set final layer accordingly
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, out_features)

        # Load weights with strict=False (so minor mismatch won't crash)
        load_info = model.load_state_dict(new_state, strict=False)
        if load_info.missing_keys:
            st.warning(f"Missing keys (showing first 10): {load_info.missing_keys[:10]}")
        if load_info.unexpected_keys:
            st.warning(f"Unexpected keys (first 10): {load_info.unexpected_keys[:10]}")

        model.eval()
        st.success("Model loaded successfully.")
        return model

    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.text(traceback.format_exc())
        return None

# ---------- PREPROCESS ----------
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def pil_to_tensor(img_pil):
    return preprocess(img_pil).unsqueeze(0)  # shape [1, C, H, W]

# ---------- PREDICT ----------
def predict_single(model, input_tensor):
    model.eval()
    with torch.no_grad():
        out = model(input_tensor)           # shape [1, out_features]
        if out.shape[1] == 1:
            prob = torch.sigmoid(out)[0,0].item()
            label = "FORGED 🔴" if prob >= 0.5 else "GENUINE 🟢"
            confidence = prob if prob >= 0.5 else 1.0 - prob
        else:
            probs = torch.softmax(out, dim=1)[0]
            idx = int(torch.argmax(probs).item())
            # assume idx 1 = forged, 0 = genuine, adjust if needed
            label = "FORGED 🔴" if idx == 1 else "GENUINE 🟢"
            confidence = float(probs[idx].item())
    return label, float(confidence), out

# ---------- MANUAL GRAD-CAM ----------
def compute_gradcam(model, input_tensor, target_layer=None):
    """
    Returns normalized CAM numpy array, shape (IMG_SIZE, IMG_SIZE), values 0..1
    """
    activations = []
    gradients = []

    if target_layer is None:
        # default target layer: last conv block of layer4
        target_layer = model.layer4[-1]

    def forward_hook(module, inp, out):
        activations.append(out.detach().cpu())

    # use full backward hook (future-proof)
    def backward_hook(module, grad_in, grad_out):
        # grad_out[0] is gradient wrt module output
        gradients.append(grad_out[0].detach().cpu())

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    out = model(input_tensor)  # forward

    # choose scalar score to backprop
    if out.shape[1] == 1:
        score = out[:, 0].sum()
    else:
        pred_class = torch.argmax(out, dim=1)
        # take the logit of predicted class
        score = out[0, pred_class].sum()

    score.backward()

    try:
        grads = gradients[0].squeeze(0)     # shape [C, H, W]
        acts  = activations[0].squeeze(0)   # shape [C, H, W]
    except Exception as e:
        fh.remove(); bh.remove()
        st.error(f"Grad-CAM hooks failed: {e}")
        return None

    # global-average-pool the grads over spatial dims
    weights = grads.mean(dim=(1,2))        # shape [C]
    cam = (weights[:, None, None] * acts).sum(dim=0).numpy()  # shape [H, W]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    fh.remove(); bh.remove()
    return cam

def overlay_heatmap_on_pil(pil_img, cam):
    """
    pil_img: PIL.Image RGB
    cam: numpy HxW normalized 0..1
    returns numpy uint8 RGB overlay (IMG_SIZE x IMG_SIZE)
    """
    img_resized = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))).astype(np.uint8)
    heatmap = np.uint8(255 * cam)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_resized, 0.6, heatmap_color, 0.4, 0)
    return overlay

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Receipt Forgery Detector", layout="centered")
st.title("🧾 Receipt Forgery Detector — (Single-logit model)")

st.sidebar.header("Model info / Notes")
st.sidebar.write("- This app expects your trained checkpoint with a single output (one logit).")
st.sidebar.write("- If you trained a two-output model instead, we would need a different checkpoint.")
st.sidebar.write("- If model isn't present, set `MODEL_GDRIVE_ID` in Streamlit secrets or use the fallback id in code.")

uploaded = st.file_uploader("Upload a receipt (png/jpg)", type=["png", "jpg", "jpeg"])
if uploaded is None:
    st.info("Upload an image to run detection.")
else:
    try:
        img = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error("Couldn't open image. Upload a valid PNG/JPG.")
        st.stop()

    st.image(img, caption="Uploaded receipt", width="stretch")

    model = load_model()
    if model is None:
        st.error("Model failed to load. Check logs/streamlit secrets.")
        st.stop()

    # Preprocess
    input_tensor = pil_to_tensor(img)

    # Prediction
    label, confidence, raw_out = predict_single(model, input_tensor)
    st.markdown(f"### Prediction: **{label}**")
    st.markdown(f"**Confidence:** {confidence*100:.2f}%")

    # Grad-CAM toggle
    if st.checkbox("Show Grad-CAM heatmap", value=True):
        with st.spinner("Generating Grad-CAM..."):
            cam = compute_gradcam(model, input_tensor, target_layer=None)
            if cam is None:
                st.warning("Grad-CAM failed for this image.")
            else:
                overlay = overlay_heatmap_on_pil(img, cam)
                st.image(overlay, caption="Model attention (Grad-CAM)", width="stretch")

                # Download heatmap
                buf = BytesIO()
                Image.fromarray(overlay).save(buf, format="PNG")
                buf.seek(0)
                st.download_button("Download heatmap", data=buf.getvalue(), file_name="heatmap.png", mime="image/png")
