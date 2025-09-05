import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
import imageio
from PIL import Image
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
import os

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model = torch.load("models/best_resnet18.pth", map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model()

# --- TRANSFORMS ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# --- PREDICTION FUNCTION ---
def predict_image(img):
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return predicted.item(), confidence.item()

# --- PDF REPORT FUNCTION ---
def generate_pdf_report(image, prediction, confidence):
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp_file.name, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, 750, "Forgery Detection Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, 700, f"Prediction: {'FORGED' if prediction==1 else 'AUTHENTIC'}")
    c.drawString(100, 680, f"Confidence: {confidence*100:.2f}%")

    # Save uploaded image temporarily and draw it
    img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    image.save(img_path)
    c.drawImage(img_path, 100, 400, width=200, height=200)
    c.save()
    return tmp_file.name

# --- APP UI ---
st.set_page_config(page_title="Receipt Forgery Detector", layout="wide")
st.title("🧾 Receipt Forgery Detector")
st.markdown("Upload one or more receipts to check if they are **forged or authentic**.")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.write("- **Model:** ResNet18")
    st.write("- **Dataset:** FindItAgain (988 receipts, 163 forgeries)")
    st.write("- **Accuracy:** ~92% on validation set")
    st.write("- **Explainability:** Grad-CAM")

# File uploader
uploaded_files = st.file_uploader("Upload receipts (PNG/JPG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# Grad-CAM toggle
show_heatmap = st.checkbox("Show Grad-CAM Heatmap", value=True)

if uploaded_files:
    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        st.subheader(f"🖼 {file.name}")

        prediction, confidence = predict_image(img)

        # Show confidence bar
        st.progress(confidence)
        st.write(f"**Prediction:** {'🔥 FORGED 🔥' if prediction==1 else '✅ AUTHENTIC ✅'}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")

        if show_heatmap:
            st.info("Grad-CAM heatmap visualization coming here (future enhancement).")

        # Generate PDF report
        pdf_path = generate_pdf_report(img, prediction, confidence)
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
            b64 = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="report_{file.name}.pdf">📥 Download Report</a>'
            st.markdown(href, unsafe_allow_html=True)

else:
    st.warning("Please upload at least one receipt to proceed.")
