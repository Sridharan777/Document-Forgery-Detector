import streamlit as st
import streamlit_authenticator as stauth
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

# ---------------- Config ----------------
IMG_SIZE = 224
MODEL_PATH = "models/best_resnet50.pth"
FALLBACK_GDRIVE_ID = "1w4EufvzDfAeVpvL7hfyFdqOce67XV8ks"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
MAX_HEIGHT, MAX_WIDTH = 450, 350

# User credentials (bcrypt hashes for passwords: alicepass, bobpass)
users = {
    "usernames": {
        "alice": {
            "name": "Alice",
            "password": "$2b$12$ZiicYZGLRhPDa0/HQsSC5uTqNnaNtSBGTFDEa7BcP6JCsG36izQkG",
        },
        "bob": {
            "name": "Bob",
            "password": "$2b$12$1Wgzm99QJMKm7NiO3TgDN.XoXcicUUuuCQxoRwoqc24k6LxjMn7aK",
        },
    }
}

authenticator = stauth.Authenticate(
    credentials=users,
    cookie_name="some_cookie_name",
    key="some_signature_key",
    cookie_expiry_days=1,
)

# Correct login call using positional location parameter
name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status:
    st.sidebar.write(f"Welcome {name}")
    if st.sidebar.button("Logout"):
        authenticator.logout("main")
        st.experimental_rerun()

    if "upload_history" not in st.session_state:
        st.session_state.upload_history = {}
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}

    if username not in st.session_state.upload_history:
        st.session_state.upload_history[username] = []
    if username not in st.session_state.feedback:
        st.session_state.feedback[username] = {}

    def add_to_history(filename):
        if filename not in st.session_state.upload_history[username]:
            st.session_state.upload_history[username].append(filename)

    def handle_feedback_local(file_key, upvote):
        if file_key not in st.session_state.feedback[username]:
            st.session_state.feedback[username][file_key] = {"up": 0, "down": 0}
        if upvote:
            st.session_state.feedback[username][file_key]["up"] += 1
        else:
            st.session_state.feedback[username][file_key]["down"] += 1

    def display_feedback_local(file_key):
        up = st.session_state.feedback[username].get(file_key, {}).get("up", 0)
        down = st.session_state.feedback[username].get(file_key, {}).get("down", 0)
        st.markdown(f"👍 {up} &nbsp;&nbsp;&nbsp; 👎 {down}")

    def apply_theme_css(theme_base):
        dark_styles = """
            body, .stApp {
                background-color: #131416;
                color: #f5f6fa;
                font-family: 'Montserrat', sans-serif;
                transition: background-color 0.4s ease, color 0.4s ease;
            }
            .result-card {
                background: #21252b;
                box-shadow: 0 4px 32px #003e9622;
                border-radius: 1.1em;
                padding: 1.25em 1.4em 1em 1.4em;
                margin-bottom: 1.1em;
                transition: background-color 0.4s ease;
            }
            .stButton>button {
                color: #fff;
                background: linear-gradient(90deg,#007BFF 60%,#5f61e6 100%);
                border: none;
                border-radius: .35em;
                font-weight: 600;
                transition: background 0.3s ease;
            }
            .stButton>button:hover {
                background: linear-gradient(90deg,#5f61e6 60%,#007BFF 100%);
            }
            a {
                color: #60c1e3;
                transition: color 0.4s ease;
            }
            .stFileUploader {
                background-color: #21252b;
                border-radius: 0.5em;
                padding: 1em;
                transition: background-color 0.4s ease;
            }
            ::-webkit-scrollbar {
                width: 8px;
            }
            ::-webkit-scrollbar-track {
                background: #181a1f;
            }
            ::-webkit-scrollbar-thumb {
                background-color: #3d3f47;
                border-radius: 10px;
                border: 2px solid #181a1f;
            }
            .tooltip:hover .tooltiptext {
                visibility: visible;
                opacity: 1;
                transition: opacity 0.4s;
            }
        """
        light_styles = """
            body, .stApp {
                background-color: #f9f9f9;
                color: #141414;
                font-family: 'Montserrat', sans-serif;
                transition: background-color 0.4s ease, color 0.4s ease;
            }
            .result-card {
                background: #ffffff;
                box-shadow: 0 4px 32px #aaa;
                border-radius: 1.1em;
                padding: 1.25em 1.4em 1em 1.4em;
                margin-bottom: 1.1em;
                transition: background-color 0.4s ease;
            }
            .stButton>button {
                color: #fff;
                background: linear-gradient(90deg,#007BFF 60%,#5f61e6 100%);
                border: none;
                border-radius: .35em;
                font-weight: 600;
                transition: background 0.3s ease;
            }
            .stButton>button:hover {
                background: linear-gradient(90deg,#5f61e6 60%,#007BFF 100%);
            }
            a {
                color: #007bff;
                transition: color 0.4s ease;
            }
            .stFileUploader {
                background-color: #ffffff;
                border-radius: 0.5em;
                padding: 1em;
                transition: background-color 0.4s ease;
            }
            ::-webkit-scrollbar {
                width: 8px;
            }
            ::-webkit-scrollbar-track {
                background: #f9f9f9;
            }
            ::-webkit-scrollbar-thumb {
                background-color: #a2a2a2;
                border-radius: 10px;
                border: 2px solid #f9f9f9;
            }
            .tooltip:hover .tooltiptext {
                visibility: visible;
                opacity: 1;
                transition: opacity 0.4s;
            }
        """
        st.markdown(f"<style>{dark_styles if theme_base == 'Dark' else light_styles}</style>", unsafe_allow_html=True)

    theme_choice = st.sidebar.radio("Select Theme", ["Light", "Dark"], index=1, help="Switch between Light and Dark mode.")
    apply_theme_css(theme_choice)

    def tooltip(label, text):
        return f"""<span class="tooltip" style="border-bottom:1px dotted; cursor:help; position: relative;">{label}
            <span class="tooltiptext" style="visibility:hidden; opacity:0; width: 210px; background-color: #555; color: #fff; text-align: center;
            border-radius: 6px; padding: 5px 8px; position: absolute; z-index: 1;
            bottom: 125%; left: 50%; margin-left: -105px; font-size: 0.85em; transition: opacity 0.3s;">
            {text}</span>
            </span>
        """

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
        safe_prediction = prediction
        pdf.cell(0, 10, f"Prediction: {safe_prediction}", 0, 1)
        pdf.cell(0, 10, f"Confidence: {confidence:.2%}", 0, 1)
        pdf.ln(10)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as orig_tmp, tempfile.NamedTemporaryFile(suffix=".png", delete=True) as gradcam_tmp:
            original_img.save(orig_tmp.name)
            gradcam_img.save(gradcam_tmp.name)
            pdf.image(orig_tmp.name, x=10, y=50, w=80)
            pdf.image(gradcam_tmp.name, x=110, y=50, w=80)
            return pdf.output(dest='S').encode('latin1')

    st.markdown("""
        <div style='display:flex;align-items:center;justify-content:space-between;padding:0.9em 1.2em 0.7em 0em;background:rgba(8,16,32,0.17);border-radius:16px;margin-bottom:1.3em;font-family: Montserrat, sans-serif;'>
            <div style='font-weight:bold;font-size:1.5em; letter-spacing:0.5px; color:#70c1b3;'>🧾 Receipt Forgery Detector</div>
            <a style='color:#4ea1d3;text-decoration:none;font-size:1.1em;' href='https://github.com/Sridharan777' target='_blank'>GitHub</a>
        </div>
    """, unsafe_allow_html=True)
    st.markdown(f"<p style='color:#e4e9ee;font-size:1.13em;margin-bottom:1.7em;'>Upload receipt image(s) to detect forgery using deep learning and get visual Grad-CAM explanations.</p>", unsafe_allow_html=True)
    st.markdown(tooltip("Upload receipt image(s) 📁", "Allowed: PNG, JPG, JPEG. You can upload multiple images at once."), unsafe_allow_html=True)

    uploaded_files = st.file_uploader("", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        for f in uploaded_files:
            add_to_history(f.name)

    with st.sidebar:
        st.header("Upload History 🕒")
        if st.session_state.upload_history[username]:
            for fname in st.session_state.upload_history[username]:
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
                    overlay_resized = overlay.resize(resized_img.size)
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
                    st.markdown(f"<div style='font-weight:bold;font-size:2em;margin-top:0.22em;color:{'#30e394' if 'GENUINE' in label else '#ff5264'}'>{label}</div>", unsafe_allow_html=True)
                    st.markdown(f"<span style='font-size:1.1em;color:#8fb9d2;'>{tooltip('Confidence:', 'Confidence score of the prediction')}</span> <b>{confidence:.2%}</b>", unsafe_allow_html=True)
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=confidence*100,
                        title={'text': "Confidence"},
                        delta={'reference': st.session_state.feedback[username].get(uploaded.name, {}).get("up", 0)},
                        gauge={'axis': {'range': [0, 100]},
                               'bar': {'color': "#30e394" if "GENUINE" in label else "#ff5264"}},
                        number={'suffix': '%'},
                    ))
                    st.plotly_chart(fig, use_container_width=True, key=f"gauge_{i}")
                    pdf_bytes = generate_pdf_report(pil_img, overlay_resized, label, confidence)
                    st.download_button("📄 Download PDF Report", pdf_bytes, file_name=f"report_{i+1}.pdf", mime="application/pdf")
                    col_up, col_down = st.columns([1, 1])
                    with col_up:
                        if st.button("👍 Useful", key=f"up_{uploaded.name}"):
                            handle_feedback_local(uploaded.name, True)
                    with col_down:
                        if st.button("👎 Not Useful", key=f"down_{uploaded.name}"):
                            handle_feedback_local(uploaded.name, False)
                    display_feedback_local(uploaded.name)
                st.markdown("</div>", unsafe_allow_html=True)
                st.caption("Switch tabs to view more receipts or download Grad-CAM images.")
                progress_bar.progress((i + 1) / len(uploaded_files))
        progress_bar.empty()
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            for fname, data in overlay_buffers:
                zipf.writestr(fname, data)
        st.download_button("📦 Download ALL Overlays (ZIP)", zip_buffer.getvalue(), file_name="all_gradcams.zip", mime="application/zip")

    st.markdown("""
        <div style='text-align:center; padding-top:2em; font-size:1em; color:#8fb9d2; font-family: Montserrat, sans-serif;'>
            Built with ❤️ using Streamlit • <a href="https://github.com/Sridharan777" style='color:#60c1e3;' target="_blank">Source on GitHub</a>
        </div>
    """, unsafe_allow_html=True)

else:
    if authentication_status is False:
        st.error("Username/password is incorrect")
    elif authentication_status is None:
        st.warning("Please enter your username and password")
    st.stop()
