import streamlit as st
import os, io
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import imageio
from io import BytesIO


# ---------------------------
# CONFIG
# ---------------------------
IMG_SIZE = 224
LOCAL_MODEL_PATH = "models/best_resnet18.pth"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ---------------------------
# MODEL LOADING
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    # Download from Google Drive if missing
    if not os.path.exists(LOCAL_MODEL_PATH):
        if "MODEL_GDRIVE_ID" in st.secrets:
            import gdown
            os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
            url = f"https://drive.google.com/uc?id={st.secrets['MODEL_GDRIVE_ID']}"
            st.info("Downloading model from Google Drive...")
            gdown.download(url, LOCAL_MODEL_PATH, quiet=False)
        else:
            st.error("Model not found and no Google Drive ID provided!")
            st.stop()

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    ckpt = torch.load(LOCAL_MODEL_PATH, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model

# ---------------------------
# PREPROCESS
# ---------------------------
def preprocess_for_model(img_rgb):
    tfm = A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, value=255),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return tfm(image=img_rgb)["image"].unsqueeze(0)

# ---------------------------
# MANUAL GRAD-CAM
# ---------------------------
def gradcam_heatmap(model, input_tensor, target_layer):
    # Forward pass & hook activations
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)

    model.zero_grad()
    out = model(input_tensor)
    out.backward(torch.ones_like(out))  # backprop from output

    # Compute CAM
    grads = gradients[0]             # [B, C, H, W]
    acts = activations[0]            # [B, C, H, W]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = torch.nn.functional.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # normalize 0-1

    handle_f.remove()
    handle_b.remove()
    return cam

def overlay_heatmap_on_image(img_rgb, cam):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.4 * heatmap + 0.6 * img_rgb)
    return overlay

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Receipt Forgery Detector", layout="centered")
st.title("🧾 Receipt Forgery Detector with Grad-CAM")

model = load_model()

uploaded_file = st.file_uploader("Upload a receipt image (png/jpg)", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    tensor = preprocess_for_model(img_rgb)

    # Prediction
    with torch.no_grad():
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()
        pred = int(prob >= 0.5)

    label = "FORGED 🔴" if pred == 1 else "GENUINE 🟢"
    st.markdown(f"### **Prediction:** {label}")
    st.markdown(f"**Confidence:** {prob:.3f}")

    # Grad-CAM Heatmap
    with st.spinner("Generating Grad-CAM heatmap..."):
        cam = gradcam_heatmap(model, tensor, model.layer4[-1])
        overlay = overlay_heatmap_on_image(img_rgb_resized, cam)
        st.image(overlay, caption="Model Attention (Grad-CAM)", use_container_width=True)

        from PIL import Image
        buf = BytesIO()
        # Convert to uint8 and ensure RGB mode
        overlay_img = Image.fromarray((overlay * 255).astype("uint8")).convert("RGB")
        overlay_img.save(buf, format="PNG")
        buf.seek(0)
        st.image(buf, caption="Grad-CAM Heatmap", use_container_width=True)

