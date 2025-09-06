# app.py — iPhone Pro / Glassmorphism UI
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import os
import gdown
import plotly.graph_objects as go
import zipfile
import time
import traceback

# ---------------- CONFIG ----------------
IMG_SIZE = 224
MODEL_PATH = "models/best_resnet50.pth"
FALLBACK_GDRIVE_ID = "1zMrv6S6rOWyiTQ0Fgw0VG0o0fEnrWzE-"  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ---------------- GLASSMORPHISM / iOS CSS ----------------
STYLES = """
<style>
body { font-family: 'Helvetica Neue', sans-serif; background: linear-gradient(145deg,#d7e1ec,#f9faff); }

/* Glass cards */
.result-card {
    border-radius: 25px;
    padding: 25px;
    margin: 15px 0;
    background: rgba(255,255,255,0.25);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px 0 rgba(31,38,135,0.15);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.result-card:hover { transform: translateY(-6px); box-shadow: 0 12px 40px 0 rgba(31,38,135,0.25); }

.result-title { font-size:22px; font-weight:700; margin-bottom:12px; }
.small-muted { color: #555; font-size:13px; }
.footer { color:#777; font-size:13px; margin-top:20px; text-align:center; }

/* Buttons */
.stDownloadButton>button { 
    border-radius: 15px;
    background: linear-gradient(90deg,#6a11cb,#2575fc);
    color:white; font-weight:700; padding:10px 20px;
    transition: transform 0.2s ease;
}
.stDownloadButton>button:hover { transform: translateY(-3px); }

/* Dark mode overrides */
body.dark-mode { background: linear-gradient(145deg,#1e1e1f,#2a2a2d); color:#eee; }
body.dark-mode .result-card { background: rgba(28,28,30,0.45); box-shadow: 0 8px 32px rgba(0,0,0,0.35); color:#e6eef8; }
body.dark-mode .small-muted { color:#aaa; }

/* File uploader style */
.css-1avcm0n { border-radius: 20px !important; }
</style>
"""

# ---------------- STREAMLIT PAGE ----------------
st.set_page_config(page_title="Document Forgery Detector", layout="wide")
st.markdown(STYLES, unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧾 Document Forgery Detector")
st.sidebar.write("Grad-CAM explainability")
st.sidebar.header("Model")
st.sidebar.write(f"`{MODEL_PATH}`")
if FALLBACK_GDRIVE_ID:
    st.sidebar.write("- Fallback Drive id configured.")

st.sidebar.header("UI / Display")
theme_dark = st.sidebar.checkbox("Enable Dark Mode", value=False)
colormap_choice = st.sidebar.selectbox("Grad-CAM colormap", ["JET","VIRIDIS","HOT","BONE"])
st.sidebar.write(f"Device: `{DEVICE}`")

st.title("🧾 Document Forgery Detector")
st.caption("Upload receipt images to see predictions, confidence & Grad-CAM with iOS-style UI.")

# ---------------- MODEL LOADING ----------------
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            if FALLBACK_GDRIVE_ID:
                gdown.download(f"https://drive.google.com/uc?id={FALLBACK_GDRIVE_ID}", MODEL_PATH, quiet=False)
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        new_state = {k.replace("module.", ""):v for k,v in state_dict.items()}
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(new_state, strict=False)
        model.to(DEVICE).eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}\n{traceback.format_exc()}")
        return None

with st.spinner("Loading model..."):
    model = load_model()
if model is None: st.stop()

# ---------------- IMAGE PROCESSING ----------------
def pil_to_tensor(img_pil):
    transform = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD)])
    return transform(img_pil).unsqueeze(0).to(DEVICE)

def predict_single(model, input_tensor):
    with torch.no_grad():
        out = model(input_tensor)
        probs = torch.softmax(out, dim=1)[0]
        idx = int(torch.argmax(probs).item())
        label = "FORGED 🔴" if idx==1 else "GENUINE 🟢"
        conf = float(probs[idx].item())
    return label, conf, out

def compute_gradcam(model, input_tensor, target_layer=None):
    activations, gradients = [], []
    if target_layer is None: target_layer = model.layer4[-1]
    fh = target_layer.register_forward_hook(lambda m,i,o: activations.append(o.detach().cpu()))
    try: bh = target_layer.register_full_backward_hook(lambda m,gi,go: gradients.append(go[0].detach().cpu()))
    except: bh = target_layer.register_backward_hook(lambda m,gi,go: gradients.append(go[0].detach().cpu()))
    model.zero_grad()
    out = model(input_tensor)
    score = out[0,int(torch.argmax(out,dim=1)[0].item())]
    score.backward()
    grads = gradients[0].squeeze(0); acts = activations[0].squeeze(0)
    weights = grads.mean(dim=(1,2))
    cam = (weights[:,None,None]*acts).sum(dim=0).numpy()
    cam = np.maximum(cam,0); cam = cv2.resize(cam,(IMG_SIZE,IMG_SIZE))
    cam = (cam-cam.min())/(cam.max()-cam.min()+1e-8)
    fh.remove(); bh.remove()
    return cam

def overlay_heatmap_on_pil(pil_img, cam, cmap_name="JET", alpha=0.4):
    img_resized = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))).astype(np.uint8)
    cmap = getattr(cv2, f"COLORMAP_{cmap_name.upper()}", cv2.COLORMAP_JET)
    heatmap = cv2.applyColorMap((cam*255).astype(np.uint8), cmap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_resized, 1-alpha, heatmap, alpha, 0)

def make_zip_download(original_pil, overlay_arr, filename_prefix):
    mem_zip = BytesIO()
    with zipfile.ZipFile(mem_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        buf1 = BytesIO(); original_pil.save(buf1,"PNG"); zf.writestr(f"{filename_prefix}_original.png", buf1.getvalue())
        buf2 = BytesIO(); Image.fromarray(overlay_arr).save(buf2,"PNG"); zf.writestr(f"{filename_prefix}_gradcam.png", buf2.getvalue())
    mem_zip.seek(0)
    return mem_zip.getvalue()

# ---------------- FILE UPLOADER ----------------
uploaded_files = st.file_uploader("Upload receipt images (png/jpg)", type=["png","jpg","jpeg"], accept_multiple_files=True)
if not uploaded_files: st.info("Upload at least one image."); st.stop()

show_heatmap = st.checkbox("Show Grad-CAM heatmap", value=True)
show_gauge = st.checkbox("Show Confidence Gauge", value=True)
show_bar = st.checkbox("Show Confidence bar", value=True)

# ---------------- PROCESS FILES ----------------
for i, uploaded in enumerate(uploaded_files):
    try: pil_img = Image.open(uploaded).convert("RGB")
    except: st.error(f"Cannot open {uploaded.name}"); continue
    input_tensor = pil_to_tensor(pil_img)
    label, confidence, raw_out = predict_single(model, input_tensor)
    accent = "#4cd137" if "GENUINE" in label else "#ff4757"

    col1, col2 = st.columns([1,1])

    # LEFT: Prediction card
    with col1:
        st.markdown(f"<div class='result-card'><div class='result-title'>Prediction: {label}</div>", unsafe_allow_html=True)
        st.image(pil_img, caption="Uploaded", width='stretch')
        if show_bar:
            bar_html = f"<div style='width:100%;background:rgba(255,255,255,0.2);border-radius:15px;margin-top:10px;'><div style='width:{confidence*100:.2f}%;background:{accent};padding:8px;text-align:center;color:white;font-weight:700;border-radius:15px;'>{confidence*100:.2f}%</div></div>"
            st.markdown(bar_html, unsafe_allow_html=True)
        if show_gauge:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=confidence*100, title={'text':"Confidence (%)"},
                                         gauge={'axis':{'range':[0,100]}, 'bar':{'color':accent},
                                                'steps':[{'range':[0,50],'color':'lightcoral'},{'range':[50,100],'color':'lightgreen'}]}))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT: Grad-CAM
    with col2:
        st.markdown("<div class='result-card'><div class='result-title'>Model Explainability</div>", unsafe_allow_html=True)
        if show_heatmap:
            with st.spinner("Generating Grad-CAM..."):
                cam = compute_gradcam(model, input_tensor)
                if cam is not None:
                    overlay = overlay_heatmap_on_pil(pil_img, cam, cmap_name=colormap_choice)
                    st.image(overlay, caption="Grad-CAM Overlay", width='stretch')
                    zip_bytes = make_zip_download(pil_img, overlay, os.path.splitext(uploaded.name)[0])
                    st.download_button("Download Original + Overlay (ZIP)", data=zip_bytes, file_name=f"{uploaded.name}_images.zip", mime="application/zip")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>Built with ❤️ — Sridharan</div>", unsafe_allow_html=True)
