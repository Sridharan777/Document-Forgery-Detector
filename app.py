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
import pandas as pd

# ---------------- CONFIG ----------------
IMG_SIZE = 224
MODEL_PATH = "models/best_resnet50.pth"
FALLBACK_GDRIVE_ID = "1zMrv6S6rOWyiTQ0Fgw0VG0o0fEnrWzE-"  # <-- Replace with your ID if needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------- HELPERS ----------------
def download_model_if_missing(gdrive_id: str):
    if os.path.exists(MODEL_PATH):
        return True
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    try:
        st.info("Downloading model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={gdrive_id}", MODEL_PATH, quiet=False)
        return os.path.exists(MODEL_PATH)
    except Exception as e:
        st.error(f"Model download failed: {e}")
        return False

@st.cache_resource(show_spinner=True)
def load_model():
    try:
        drive_id = st.secrets.get("MODEL_GDRIVE_ID", FALLBACK_GDRIVE_ID)
        if not download_model_if_missing(drive_id):
            st.error("Model not found. Upload to repo or set MODEL_GDRIVE_ID secret.")
            return None

        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)

        # Fix 'module.' prefix if present
        new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)  # single logit

        model.load_state_dict(new_state, strict=False)
        model.to(DEVICE).eval()
        st.toast("✅ Model loaded successfully!", icon="✅")
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
    with torch.no_grad():
        out = model(input_tensor)
        prob = torch.sigmoid(out)[0, 0].item()
        label = "FORGED 🔴" if prob >= 0.5 else "GENUINE 🟢"
        confidence = prob if prob >= 0.5 else 1 - prob
    return label, confidence, out

def compute_gradcam(model, input_tensor, target_layer=None):
    activations, gradients = [], []
    if target_layer is None:
        target_layer = model.layer4[-1]

    def forward_hook(module, inp, out): activations.append(out.detach().cpu())
    def backward_hook(module, grad_in, grad_out): gradients.append(grad_out[0].detach().cpu())

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    out = model(input_tensor)
    score = out[:, 0].sum()
    score.backward()

    grads, acts = gradients[0].squeeze(0), activations[0].squeeze(0)
    weights = grads.mean(dim=(1, 2))
    cam = (weights[:, None, None] * acts).sum(dim=0).numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    fh.remove(); bh.remove()
    return cam

def overlay_heatmap_on_pil(pil_img: Image.Image, cam: np.ndarray, cmap="JET", alpha=0.4):
    img_resized = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))).astype(np.uint8)
    colormap = getattr(cv2, f"COLORMAP_{cmap.upper()}", cv2.COLORMAP_JET)
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_resized, 1.0 - alpha, heatmap, alpha, 0)

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Receipt Forgery Detector", layout="wide")

st.markdown("<h1 style='text-align:center; color:#2c3e50;'>🧾 Receipt Forgery Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Powered by ResNet50 + Grad-CAM | Built by YOU 🚀</p>", unsafe_allow_html=True)

tabs = st.tabs(["🔍 Prediction", "ℹ️ About Project"])

with tabs[0]:
    uploaded_files = st.file_uploader("Upload receipt image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    st.sidebar.header("⚙️ Options")
    colormap_choice = st.sidebar.selectbox("Grad-CAM Colormap", ["JET", "VIRIDIS", "HOT", "BONE"])
    show_gauge = st.sidebar.toggle("Show Gauge", value=True)

    if "history" not in st.session_state:
        st.session_state["history"] = []

    if uploaded_files:
        model = load_model()
        if model:
            for uploaded in uploaded_files:
                pil_img = Image.open(uploaded).convert("RGB")
                input_tensor = pil_to_tensor(pil_img)
                label, confidence, _ = predict_single(model, input_tensor)

                st.subheader(f"📄 {uploaded.name}")
                st.image(pil_img, caption="Uploaded receipt", width="50%")

                st.markdown(f"**Prediction:** {label}")
                st.markdown(f"**Confidence:** {confidence*100:.2f}%")

                # HTML confidence bar
                color = "#2ecc71" if "GENUINE" in label else "#e74c3c"
                st.markdown(f"""
                <div style="width:100%; background:#eee; border-radius:8px;">
                  <div style="width:{confidence*100:.1f}%; background:{color}; padding:6px;
                              border-radius:8px; text-align:center; color:white; font-weight:600;">
                    {confidence*100:.1f}%
                  </div>
                </div>
                """, unsafe_allow_html=True)

                if show_gauge:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence * 100,
                        gauge={'axis': {'range': [0, 100]},
                               'bar': {'color': color},
                               'steps': [
                                   {'range': [0, 50], 'color': "lightcoral"},
                                   {'range': [50, 100], 'color': "lightgreen"},
                               ]},
                        title={'text': "Model Confidence"}
                    ))
                    st.plotly_chart(fig, use_container_width=True, key=f"gauge_{uploaded.name}")

                with st.spinner("Generating Grad-CAM..."):
                    cam = compute_gradcam(model, input_tensor)
                    overlay = overlay_heatmap_on_pil(pil_img, cam, cmap=colormap_choice)
                    st.image(overlay, caption="Grad-CAM Heatmap", width="50%")

                # Add to history
                st.session_state["history"].append({
                    "file": uploaded.name,
                    "prediction": label,
                    "confidence": round(confidence * 100, 2)
                })

    if st.session_state["history"]:
        st.subheader("📊 Prediction History")
        df = pd.DataFrame(st.session_state["history"])
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download History CSV", data=csv, file_name="prediction_history.csv", mime="text/csv")

with tabs[1]:
    st.markdown("### 📖 About This Project")
    st.write("""
    **Document Forgery Detection using CNN (ResNet50) + Grad-CAM**
    
    - **Trained on:** Your custom receipt dataset
    - **Architecture:** ResNet50 (Fine-tuned)
    - **Explainability:** Grad-CAM heatmaps
    - **Confidence:** Gauge + bar visualization
    - **Built with:** PyTorch, Streamlit, Plotly
    
    **Developer:** Your Name  
    🔗 [LinkedIn](https://www.linkedin.com/) | [GitHub](https://github.com/)
    """)
