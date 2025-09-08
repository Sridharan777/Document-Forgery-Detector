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
import zipfile
import gdown
import plotly.graph_objects as go
from fpdf import FPDF  # Added for PDF generation

# ---------------- Config ----------------
IMG_SIZE = 224
MODEL_PATH = "models/best_resnet50.pth"
FALLBACK_GDRIVE_ID = "1w4EufvzDfAeVpvL7hfyFdqOce67XV8ks"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
MAX_HEIGHT, MAX_WIDTH = 450, 350

# ---------- Page config and Google Fonts ----------
st.set_page_config(
    page_title="Receipt Forensics",
    page_icon="🧾",
    layout="wide",
)

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ---------- Sidebar: Theme toggle and history ----------
theme = st.sidebar.radio("Select Theme", ["Light", "Dark"], index=1, help="Switch between Light and Dark mode.")

if "upload_history" not in st.session_state:
    st.session_state.upload_history = []

if "feedback" not in st.session_state:
    st.session_state.feedback = {}

def apply_theme_css(theme_base):
    # ... (same CSS as before)
    # Paste your CSS strings for dark_styles and light_styles here exactly as before (omitted here for brevity)
    # Use previous apply_theme_css function code here
    pass

apply_theme_css(theme)

def tooltip(label, text):
    return f"""<span class="tooltip" style="border-bottom:1px dotted; cursor:help; position: relative;">{label}
        <span class="tooltiptext" style="visibility:hidden; opacity:0; width: 210px; background-color: #555; color: #fff; text-align: center;
        border-radius: 6px; padding: 5px 8px; position: absolute; z-index: 1;
        bottom: 125%; left: 50%; margin-left: -105px; font-size: 0.85em; transition: opacity 0.3s;">
        {text}</span>
        </span>
    """

def download_model_if_missing(gdrive_id: str):
    # ... unchanged code ...

@st.cache_resource(show_spinner=True)
def load_model():
    # ... unchanged code ...

def pil_to_tensor(img_pil: Image.Image):
    # ... unchanged code ...

def predict_single(model, input_tensor):
    # ... unchanged code ...

def compute_gradcam(model, input_tensor, target_layer=None):
    # ... unchanged code ...

def overlay_heatmap_on_pil(pil_img, cam, alpha=0.4):
    # ... unchanged code ...

def resize_for_display(pil_img, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    # ... unchanged code ...

def handle_feedback(file_key, upvote):
    # ... unchanged code ...

def display_feedback(file_key):
    # ... unchanged code ...

# ----------- PDF generation for reports -----------
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

    # Save images temporarily in memory
    original_buf = BytesIO()
    original_img.save(original_buf, format='PNG')
    gradcam_buf = BytesIO()
    gradcam_img.save(gradcam_buf, format='PNG')
    original_buf.seek(0)
    gradcam_buf.seek(0)

    # Add images to PDF
    pdf.image(original_buf, x=10, y=50, w=80)
    pdf.image(gradcam_buf, x=110, y=50, w=80)

    return pdf.output(dest='S').encode('latin1')

# ---------- Header & Branding ----------
with st.container():
    # ... same as before ...
    pass

st.markdown(tooltip("Upload receipt image(s) 📁", "Allowed: PNG, JPG, JPEG. You can upload multiple images at once."), unsafe_allow_html=True)
uploaded_files = st.file_uploader("", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        if f.name not in st.session_state.upload_history:
            st.session_state.upload_history.append(f.name)

with st.sidebar:
    st.header("Upload History 🕒")
    if st.session_state.upload_history:
        for fname in st.session_state.upload_history:
            st.markdown(f"• {fname}")
    else:
        st.write("No history yet.")

with st.spinner("🧠 Loading detection model, please wait..."):
    model = load_model()
if model is None:
    st.stop()

if uploaded_files:
    overlay_buffers = []
    tabs = st.tabs([f"Receipt {i+1}" for i in range(len(uploaded_files))])
    progress_bar = st.progress(0)
    for i, (uploaded, tab) in enumerate(zip(uploaded_files, tabs)):
        with tab:
            pil_img = Image.open(uploaded).convert("RGB")
            resized_img = resize_for_display(pil_img)
            input_tensor = pil_to_tensor(pil_img)

            with st.spinner(f"Processing {uploaded.name}..."):
                label, confidence, raw_out = predict_single(model, input_tensor)
                cam = compute_gradcam(model, input_tensor)
                overlay = overlay_heatmap_on_pil(pil_img, cam)
                overlay_resized = Image.fromarray(overlay).resize(resized_img.size)

            buf = BytesIO()
            overlay_resized.save(buf, format="PNG")
            overlay_buffers.append((f"gradcam_{i+1}.png", buf.getvalue()))

            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1.1, 1.05, 0.95], gap="medium")

            with col1:
                st.image(resized_img, caption=f"📄 Original Receipt", use_column_width=True)
            with col2:
                st.image(overlay_resized, caption="🔥 Grad-CAM", use_column_width=True)
            with col3:
                st.markdown(f"<div style='font-size:1.25em'><b>{tooltip('Prediction:', 'Whether the receipt is Genuine or Forged')}</b></div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div style='font-weight:bold;font-size:2em;margin-top:0.22em;color:{'#30e394' if 'GENUINE' in label else '#ff5264'}'>{label}</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<span style='font-size:1.1em;color:#8fb9d2;'>{tooltip('Confidence:', 'Confidence score of the prediction')}</span> <b>{confidence:.2%}</b>",
                    unsafe_allow_html=True)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=confidence*100,
                    title={'text': "Confidence"},
                    delta={'reference': st.session_state.feedback.get(uploaded.name, {}).get("up", 0)},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "#30e394" if "GENUINE" in label else "#ff5264"}},
                    number={'suffix': '%'},
                ))
                st.plotly_chart(fig, use_container_width=True, key=f"gauge_{i}")

                # PDF Report Download Button
                pdf_bytes = generate_pdf_report(pil_img, overlay_resized, label, confidence)
                st.download_button("📄 Download PDF Report", pdf_bytes, file_name=f"report_{i+1}.pdf", mime="application/pdf")

                col_up, col_down = st.columns([1, 1])
                with col_up:
                    if st.button("👍 Useful", key=f"up_{uploaded.name}"):
                        handle_feedback(uploaded.name, True)
                with col_down:
                    if st.button("👎 Not Useful", key=f"down_{uploaded.name}"):
                        handle_feedback(uploaded.name, False)

                display_feedback(uploaded.name)

            st.markdown("</div>", unsafe_allow_html=True)
            st.caption("Switch tabs to view more receipts or download Grad-CAM images.")

            progress_bar.progress((i + 1) / len(uploaded_files))
    progress_bar.empty()

    # ZIP download of overlays
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for fname, data in overlay_buffers:
            zipf.writestr(fname, data)
    st.download_button("📦 Download ALL Overlays (ZIP)", zip_buffer.getvalue(), file_name="all_gradcams.zip", mime="application/zip")

# ---------- Footer ----------
st.markdown("""
    <div style='text-align:center; padding-top:2em; font-size:1em; color:#8fb9d2; font-family: Montserrat, sans-serif;'>
        Built with ❤️ using Streamlit • <a href="https://github.com/Sridharan777" style='color:#60c1e3;' target="_blank">Source on GitHub</a>
    </div>
""", unsafe_allow_html=True)
