import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import os, traceback, zipfile, gdown
import plotly.graph_objects as go

# -------- CONFIG (same as before) --------
IMG_SIZE = 224
MODEL_PATH = "models/best_resnet50.pth"
FALLBACK_GDRIVE_ID = "1w4EufvzDfAeVpvL7hfyFdqOce67XV8ks"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
MAX_HEIGHT, MAX_WIDTH = 450, 350

# ----------- UI THEME ENHANCEMENTS -----------
st.set_page_config(
    page_title="Receipt Forensics",
    page_icon="🧾",
    layout="wide"
)

# -------- THEME TOGGLE IN SIDEBAR -----------
theme = st.sidebar.radio("Select Theme", ["Light", "Dark"], index=1, help="Switch between Light and Dark mode.")

# Apply CSS styles dynamically based on theme
if theme == "Dark":
    st.markdown("""
        <style>
            body, .stApp {
                background-color: #131416;
                color: #f5f6fa;
            }
            .result-card {
                background: #21252b;
                box-shadow: 0 4px 32px #003e9622;
                border-radius: 1.1em;
                padding: 1.25em 1.4em 1em 1.4em;
                margin-bottom: 1.1em;
            }
            .stButton>button {
                color: #fff;
                background: linear-gradient(90deg,#007BFF 60%,#5f61e6 100%);
                border: none;
                border-radius: .35em;
                font-weight: 600;
            }
            a {
                color: #60c1e3;
            }
            .stFileUploader {
                background-color: #21252b;
                border-radius: 0.5em;
                padding: 1em;
            }
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            body, .stApp {
                background-color: #f9f9f9;
                color: #141414;
            }
            .result-card {
                background: #ffffff;
                box-shadow: 0 4px 32px #aaa;
                border-radius: 1.1em;
                padding: 1.25em 1.4em 1em 1.4em;
                margin-bottom: 1.1em;
            }
            .stButton>button {
                color: #fff;
                background: linear-gradient(90deg,#007BFF 60%,#5f61e6 100%);
                border: none;
                border-radius: .35em;
                font-weight: 600;
            }
            a {
                color: #007bff;
            }
            .stFileUploader {
                background-color: #ffffff;
                border-radius: 0.5em;
                padding: 1em;
            }
        </style>
        """, unsafe_allow_html=True)

# ------------ HELPERS (same as before) ------------
def download_model_if_missing(gdrive_id: str):
    if os.path.exists(MODEL_PATH): return True
    if not gdrive_id: return False
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
        drive_id = None
        try:
            drive_id = st.secrets.get("MODEL_GDRIVE_ID", None)
        except:
            pass
        if not os.path.exists(MODEL_PATH):
            if not download_model_if_missing(drive_id) and not download_model_if_missing(FALLBACK_GDRIVE_ID):
                st.error("Model not found locally and no valid Drive ID available.")
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
        out_features = new_state.get("fc.weight", torch.empty((2,))).shape[0] if "fc.weight" in new_state else 2
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, out_features)
        model.load_state_dict(new_state, strict=False)
        model.to(DEVICE).eval()
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

def predict_single(model, input_tensor):
    with torch.no_grad():
        out = model(input_tensor).to(torch.float32)
        if out.shape[1] == 1:
            prob = torch.sigmoid(out)[0, 0].item()
            return ("FORGED 🔴", prob, out) if prob >= 0.5 else ("GENUINE 🟢", 1.0 - prob, out)
        probs = torch.softmax(out, dim=1)[0]
        idx = int(torch.argmax(probs))
        return ("FORGED 🔴" if idx == 1 else "GENUINE 🟢", float(probs[idx]), out)

def compute_gradcam(model, input_tensor, target_layer=None):
    activations, gradients = [], []
    if target_layer is None: target_layer = model.layer4[-1]
    def forward_hook(m, i, o): activations.append(o.detach().cpu())
    def backward_hook(m, gi, go): gradients.append(go[0].detach().cpu())
    fh = target_layer.register_forward_hook(forward_hook)
    try:
        bh = target_layer.register_full_backward_hook(lambda m, gi, go: backward_hook(m, gi, go))
    except:
        bh = target_layer.register_backward_hook(lambda m, gi, go: backward_hook(m, gi, go))
    model.zero_grad()
    out = model(input_tensor)
    score = out[:, 0].sum() if out.shape[1] == 1 else out[0, int(torch.argmax(out, dim=1))]
    score.backward()
    grads, acts = gradients[0].squeeze(0), activations[0].squeeze(0)
    weights = grads.mean(dim=(1, 2))
    cam = np.maximum((weights[:, None, None] * acts).sum(dim=0).numpy(), 0)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

def overlay_heatmap_on_pil(pil_img, cam, alpha=0.4):
    img_resized = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))).astype(np.uint8)
    heatmap = np.uint8(255 * cam)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_resized, 1 - alpha, heatmap_color, alpha, 0)

def resize_for_display(pil_img, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    w, h = pil_img.size
    scale = min(max_width / w, max_height / h)
    return pil_img.resize((int(w * scale), int(h * scale)))

# -------- HEADER & NAVBAR --------
with st.container():
    st.markdown("""
    <div style='display:flex;align-items:center;justify-content:space-between;padding:0.9em 1.2em 0.7em 0em;background:rgba(8,16,32,0.17);border-radius:16px;margin-bottom:1.3em;'>
        <div style='font-weight:bold;font-size:1.5em; letter-spacing:0.5px; color:#70c1b3;'>🧾 Receipt Forgery Detector</div>
        <a style='color:#4ea1d3;text-decoration:none;font-size:1.1em;' href='https://github.com/Sridharan777' target='_blank'>GitHub</a>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#0a0a0a;font-size:1.13em;margin-bottom:1.7em;'>Upload one or more receipts to detect forgery using deep learning and get visual Grad-CAM explanations.</p>", 
        unsafe_allow_html=True
    )

# ----------- MAIN APP INTERACTION ---------------
uploaded_files = st.file_uploader(
    "Upload receipt image(s)", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True, 
    help="You can upload multiple receipts at once."
)

if not uploaded_files:
    st.info("No images uploaded. Please add one or more receipt images to begin.")
    st.stop()

model = load_model()
if model is None:
    st.stop()

overlay_buffers = []
tabs = st.tabs([f"Receipt {i+1}" for i in range(len(uploaded_files))])

for i, (uploaded, tab) in enumerate(zip(uploaded_files, tabs)):
    with tab:
        pil_img = Image.open(uploaded).convert("RGB")
        resized_img = resize_for_display(pil_img)
        input_tensor = pil_to_tensor(pil_img)
        label, confidence, raw_out = predict_single(model, input_tensor)
        cam = compute_gradcam(model, input_tensor)
        overlay = overlay_heatmap_on_pil(pil_img, cam)
        overlay_resized = Image.fromarray(overlay).resize(resized_img.size)
        buf = BytesIO()
        overlay_resized.save(buf, format="PNG")
        overlay_buffers.append((f"gradcam_{i+1}.png", buf.getvalue()))

        # The result card
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1.1, 1.05, 0.95], gap="medium")
        with c1:
            st.image(resized_img, caption=f"📄 Original Receipt", use_container_width=True)
        with c2:
            st.image(overlay_resized, caption="🔥 Grad-CAM", use_container_width=True)
        with c3:
            st.markdown("<div style='font-size:1.25em'><b>Prediction:</b></div>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:2em;font-weight:bold;margin-top:0.22em;color:{'#30e394' if 'GENUINE' in label else '#ff5264'}'>{label}</div>", 
                unsafe_allow_html=True)
            st.markdown(
                f"<span style='font-size:1.1em;color:#8fb9d2;'>Confidence:</span> <b>{confidence:.2%}</b>", 
                unsafe_allow_html=True
            )
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=confidence*100,
                title={'text': "Confidence"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "#30e394" if "GENUINE" in label else "#ff5264"}}
            ))
            st.plotly_chart(fig, use_container_width=True, key=f"gauge_{i}")
            st.download_button("💾 Download Grad-CAM", buf.getvalue(), file_name=f"gradcam_{i+1}.png", mime="image/png")
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Go to next tab for more results or download overlays.")

# ------- ZIP Download for All Overlays ---------
if overlay_buffers:
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for fname, data in overlay_buffers:
            zipf.writestr(fname, data)
    st.download_button("📦 Download ALL Overlays (ZIP)", zip_buffer.getvalue(),
                       file_name="all_gradcams.zip", mime="application/zip")

# --------- BEAUTIFUL FOOTER ----------
st.markdown("""
<div style='text-align:center; padding-top:2em; font-size:1em; color:#8fb9d2;'>
    Built with ❤️ using Streamlit • <a href="https://github.com/Sridharan777" style='color:#60c1e3;' target="_blank">Source on GitHub</a>
</div>
""", unsafe_allow_html=True)
