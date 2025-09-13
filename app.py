import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
from io import BytesIO
import os
import traceback
import zipfile
import gdown
import plotly.graph_objects as go
from fpdf import FPDF
import tempfile
import matplotlib.cm as cm

# --------- Theme  & CSS  ----------
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"  

def set_theme_css(mode):
    if mode == "Dark":
        css = """
        body, .stApp {
            background-color: #121517 !important;
            color: #e4e9ee !important;
            font-family: 'Montserrat', sans-serif;
        }
        a {
            color: #60c1e3 !important;
            transition: color 0.4s ease;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #70c1b3 !important;
            font-weight: 700;
        }
        .result-card {
            background: #21252b !important;
            box-shadow: 0 4px 32px #003e9622 !important;
            border-radius: 1.1em !important;
            padding: 1.25em 1.4em 1em 1.4em !important;
            margin-bottom: 1.1em !important;
            transition: background-color 0.4s ease;
        }
        .stButton>button {
            color: #fff !important;
            background: linear-gradient(90deg,#007BFF 60%,#5f61e6 100%) !important;
            border: none !important;
            border-radius: .35em !important;
            font-weight: 600 !important;
            font-size: 1.1em !important;
            transition: background 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg,#5f61e6 60%,#007BFF 100%) !important;
            cursor: pointer;
        }
        [data-testid="stSidebar"] {
            background-color: #21252b !important;
            color: #e4e9ee !important;
        }
        [data-testid="stSidebar"] .stTextInput>div>div>input {
            background-color: #2a2e37 !important;
            color: #e4e9ee !important;
            border: none !important;
            border-radius: 0.3em !important;
        }
        .stFileUploader {
            background-color: #21252b !important;
            border-radius: 0.5em !important;
            padding: 1em !important;
            transition: background-color 0.4s ease;
        }
        ::-webkit-scrollbar {
            width: 8px !important;
        }
        ::-webkit-scrollbar-track {
            background: #181a1f !important;
        }
        ::-webkit-scrollbar-thumb {
            background-color: #3d3f47 !important;
            border-radius: 10px !important;
            border: 2px solid #181a1f !important;
        }
        body, .stApp, .result-card:hover {
            transition: background-color 0.4s ease, color 0.4s ease;
        }
        """
    else:  # Light theme CSS
        css = """
        body, .stApp {
            background-color: #f9f9f9 !important;
            color: #141414 !important;
            font-family: 'Montserrat', sans-serif;
        }
        a {
            color: #007bff !important;
            transition: color 0.4s ease;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2563eb !important;
            font-weight: 700;
        }
        .result-card {
            background: #ffffff !important;
            box-shadow: 0 4px 32px #aaa !important;
            border-radius: 1.1em !important;
            padding: 1.25em 1.4em 1em 1.4em !important;
            margin-bottom: 1.1em !important;
            transition: background-color 0.4s ease;
        }
        .stButton>button {
            color: #fff !important;
            background: linear-gradient(90deg,#007bff 60%,#5a86ff 100%) !important;
            border: none !important;
            border-radius: .35em !important;
            font-weight: 600 !important;
            font-size: 1.1em !important;
            transition: background 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg,#5a86ff 60%,#007bff 100%) !important;
            cursor: pointer;
        }
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            color: #141414 !important;
        }
        [data-testid="stSidebar"] .stTextInput>div>div>input {
            background-color: #fafafa !important;
            color: #141414 !important;
            border: 1px solid #ddd !important;
            border-radius: 0.3em !important;
        }
        .stFileUploader {
            background-color: #ffffff !important;
            border-radius: 0.5em !important;
            padding: 1em !important;
            transition: background-color 0.4s ease;
        }
        ::-webkit-scrollbar {
            width: 8px !important;
        }
        ::-webkit-scrollbar-track {
            background: #f0f0f0 !important;
        }
        ::-webkit-scrollbar-thumb {
            background-color: #b0b0b0 !important;
            border-radius: 10px !important;
            border: 2px solid #f0f0f0 !important;
        }
        body, .stApp, .result-card:hover {
            transition: background-color 0.4s ease, color 0.4s ease;
        }
        """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

theme_choice = st.sidebar.radio("Select Theme", ["Light", "Dark"], index=1, key="theme_mode")
set_theme_css(theme_choice)

# -----------  layout and cards -----------

st.set_page_config(layout="wide")
st.markdown("""
<style>
.result-card {
    background: #21252b;
    border-radius: 16px;
    padding: 28px;
    max-width: 980px !important;
    width: 96vw !important;
    margin-left: auto;
    margin-right: auto;
    font-size: 1.14em;
    box-shadow: 0 6px 24px rgba(0,0,0,0.18);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    cursor: pointer;
    margin-bottom: 1.2em;
}
.result-card:hover {
    transform: translateY(-7px);
    box-shadow: 0 14px 34px rgba(0,123,255,0.15);
}
.stButton>button {
    background: linear-gradient(90deg, #3b82f6 60%, #2563eb 100%);
    border: none;
    border-radius: 10px;
    color: white;
    font-weight: bold;
    font-size: 1.15em;
    padding: 12px 28px;
    transition: background 0.3s ease, transform 0.2s ease;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #2563eb 60%, #3b82f6 100%);
    transform: scale(1.06);
}
.hover-zoom img {
    transition: transform 0.3s ease;
    border-radius: 15px;
}
.hover-zoom:hover img {
    transform: scale(1.05);
}
.stPlotlyChart > div {
    margin-left: auto !important;
    margin-right: auto !important;
}
</style>
""", unsafe_allow_html=True)

# -----------  ML and UI  -----------

IMG_SIZE = 224
MODEL_PATH = "models/best_resnet50.pth"
FALLBACK_GDRIVE_ID = "1w4EufvzDfAeVpvL7hfyFdqOce67XV8ks"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
MAX_HEIGHT, MAX_WIDTH = 600, 480


def download_model_if_missing(gdrive_id):
    if os.path.exists(MODEL_PATH): return True
    if not gdrive_id: return False
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    try:
        st.info(" Downloading model from Google Drive ...")
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

def pil_to_tensor(img_pil):
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
            return ("FORGED", prob, out) if prob >= 0.5 else ("GENUINE", 1.0 - prob, out)
        probs = torch.softmax(out, dim=1)[0]
        idx = int(torch.argmax(probs))
        return ("FORGED" if idx == 1 else "GENUINE", float(probs[idx]), out)

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
    cam_img = Image.fromarray((255 * ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8))).astype(np.uint8))
    cam_img = cam_img.resize((IMG_SIZE, IMG_SIZE))
    cam_norm = np.array(cam_img) / 255.0
    return cam_norm

def overlay_heatmap_on_pil(pil_img, cam, alpha=0.4):
    pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE)).convert("RGBA")
    heatmap_uint8 = (cam * 255).astype(np.uint8)
    colormap = cm.get_cmap("jet")
    heatmap_rgba = colormap(heatmap_uint8 / 255.0, bytes=True)
    heatmap_img = Image.fromarray(heatmap_rgba, mode="RGBA")
    heatmap_img.putalpha(int(255 * alpha))
    composed = Image.alpha_composite(pil_img, heatmap_img)
    return composed.convert("RGB")

def resize_for_display(pil_img, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    w, h = pil_img.size
    scale = min(max_width / w, max_height / h)
    return pil_img.resize((int(w * scale), int(h * scale)))

def generate_pdf_report(original_img, gradcam_img, prediction, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Receipt Forgery Detection Report", 0, 1, 'C')
    pdf.ln(5)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Prediction: {prediction}", 0, 1)
    pdf.cell(0, 10, f"Confidence: {confidence:.2%}", 0, 1)
    pdf.ln(10)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as orig_tmp, tempfile.NamedTemporaryFile(suffix=".png", delete=True) as gradcam_tmp:
        original_img.save(orig_tmp.name)
        gradcam_img.save(gradcam_tmp.name)
        pdf.image(orig_tmp.name, x=10, y=50, w=80)
        pdf.image(gradcam_tmp.name, x=110, y=50, w=80)
        return pdf.output(dest='S').encode('latin1')

def draw_confidence_gauge(confidence, label):
    color = "#30e394" if label == "GENUINE" else "#ff5264"
    arrow_color = "#30e394" if label == "GENUINE" else "#ff5264"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(confidence * 100),
        title={'text': "Confidence", "font": {"size": 32}},
        number={"suffix": "%", "font": {"size": 54}},
        delta={"reference": 50, "increasing": {"color": arrow_color}, "decreasing": {"color": arrow_color}, "font": {"size":22}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 2, "tickcolor": "#cbd5e1"},
            "bar": {"color": color, "thickness": 0.38},
            "bgcolor": "#f6f7fa",
            "borderwidth": 2,
            "bordercolor": "#e2e8f0",
            "steps": [
                {"range": [0, 50], "color": "#fee2e2"},
                {"range": [50, 100], "color": "#d1e7dd"},
            ],
            "threshold": {
                "line": {"color": color, "width": 8},
                "thickness": 0.85,
                "value": confidence * 100,
            }
        }
    ))
    fig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        width=370, height=340,
        paper_bgcolor="#f8fafc"
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------- App heading -----------
st.markdown("""
    <div style='display:flex;align-items:center;justify-content:space-between;padding:1.3em 2.2em 1em 0em;background:rgba(8,16,32,0.12);border-radius:22px;margin-bottom:1.5em;font-family: Montserrat, sans-serif;'>
        <div style='font-weight:700;font-size:2em;letter-spacing:1px; color:#60c1e3;'>🧾 Document Forgery Detector</div>
        <a style='color:#60c1e3;text-decoration:none;font-size:1.15em;' href='https://github.com/Sridharan777' target='_blank'>GitHub</a>
    </div>
""", unsafe_allow_html=True)

st.markdown("<p style='color:#d1d5db;font-size:1.23em;margin-bottom:1.7em;font-weight:500;'>Upload receipt image(s) to detect Forged or Genuine (with visual Grad-CAM explanations)</p>", unsafe_allow_html=True)

# ----------- Upload and history -----------
if "upload_history" not in st.session_state:
    st.session_state.upload_history = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}

uploaded_files = st.file_uploader("Receipt image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
if uploaded_files:
    for f in uploaded_files:
        if f.name not in st.session_state.upload_history:
            st.session_state.upload_history.append(f.name)

with st.sidebar.expander("Upload History 🕒", expanded=False):
    if st.session_state.upload_history:
        for fname in st.session_state.upload_history:
            st.markdown(f"• {fname}", unsafe_allow_html=True)
    else:
        st.write("No upload history yet.")

# ----------- Model loading -----------
with st.spinner(" Loading detection model, please wait..."):
    model = load_model()
if model is None:
    st.stop()

# ----------- Prediction and display -----------
if uploaded_files:
    overlay_buffers = []
    tabs = st.tabs([f"Receipt {i+1}" for i in range(len(uploaded_files))])
    progress_bar = st.progress(0)
    for i, (uploaded, tab) in enumerate(zip(uploaded_files, tabs)):
        with tab:
            pil_img = Image.open(uploaded).convert("RGB")
            resized_img = resize_for_display(pil_img)
            input_tensor = pil_to_tensor(pil_img)
            with st.spinner(f"Processing {uploaded.name} ..."):
                label, confidence, raw_out = predict_single(model, input_tensor)
                cam = compute_gradcam(model, input_tensor)
                overlay = overlay_heatmap_on_pil(pil_img, cam)
                overlay_resized = overlay.resize(resized_img.size)
            buf = BytesIO()
            overlay_resized.save(buf, format="PNG")
            overlay_buffers.append((f"gradcam_{i+1}.png", buf.getvalue()))
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1.2, 1.1, 1.2], gap="medium")
            with col1:
                st.markdown('<div class="hover-zoom">', unsafe_allow_html=True)
                st.image(resized_img, caption="Original Receipt", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="hover-zoom">', unsafe_allow_html=True)
                st.image(overlay_resized, caption="Grad-CAM", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                    <div style='text-align:center; font-size:2em; font-weight:700; color:#60c1e3; letter-spacing:1px; margin-bottom:0.3em;'>
                          Prediction Result
                    </div>
                    <div style='text-align:center; font-size:1.5em; font-weight:600; color:{ '#30e394' if label=='GENUINE' else '#ff5264' }; margin-bottom:0.3em;'>
                        {label} { '✓' if label == 'GENUINE' else '⚠️' }
                    </div>
                """, unsafe_allow_html=True)
                draw_confidence_gauge(confidence, label)
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    "Download PDF Report",
                    generate_pdf_report(pil_img, overlay_resized, label, confidence),
                    file_name=f"report_{i+1}.pdf",
                    mime="application/pdf"
                )
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption("Switch tabs to view more receipts or download Grad-CAM images.")
            progress_bar.progress((i + 1) / len(uploaded_files))
    progress_bar.empty()
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for fname, data in overlay_buffers:
            zipf.writestr(fname, data)
    st.download_button("Download ALL Overlays (ZIP)", zip_buffer.getvalue(), file_name="all_gradcams.zip", mime="application/zip")

st.markdown("""
    <div style='text-align:center; padding-top:2em; font-size:1.15em; color:#8585a1; font-family: Montserrat, sans-serif;'>
        Built using Streamlit • <a href="https://github.com/Sridharan777" style='color:#60c1e3;' target="_blank">Source on GitHub</a>
    </div>
""", unsafe_allow_html=True)
