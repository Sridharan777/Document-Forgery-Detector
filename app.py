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

# Import Google Fonts
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ---------- Sidebar: Theme toggle and history ----------
theme = st.sidebar.radio("Select Theme", ["Light", "Dark"], index=1, help="Switch between Light and Dark mode.")

# Initialize session state collections if not present
if "upload_history" not in st.session_state:
    st.session_state.upload_history = []

if "feedback" not in st.session_state:
    st.session_state.feedback = {}

# ---------- Dynamic Theming CSS ----------
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
        /* Scrollbar for dark mode */
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
        /* Tooltip styling */
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
        /* Scrollbar for light mode */
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
        /* Tooltip styling */
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
            transition: opacity 0.4s;
        }
    """
    st.markdown(f"<style>{dark_styles if theme_base == 'Dark' else light_styles}</style>", unsafe_allow_html=True)

apply_theme_css(theme)

# ---------- Tooltip helper ----------
def tooltip(label, text):
    return f"""<span class="tooltip" style="border-bottom:1px dotted; cursor:help;">{label}
        <span class="tooltiptext" style="visibility:hidden; opacity:0;
        width: 210px; background-color: #555; color: #fff; text-align: center;
        border-radius: 6px; padding: 5px 8px; position: absolute; z-index: 1;
        bottom: 125%; left: 50%; margin-left: -105px; font-size: 0.85em;
        transition: opacity 0.3s;">
        {text}</span>
        </span>
    """

# ------------ Helpers (same as before) ------------

# ... (All your helper functions like download_model_if_missing, load_model, etc., remain unchanged)

# --------------- File upload fixed show tooltip -------------

st.markdown(tooltip("Upload receipt image(s) 📁", "Allowed: PNG, JPG, JPEG. You can upload multiple images at once."), unsafe_allow_html=True)
uploaded_files = st.file_uploader("", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# ... (Rest of your app logic unchanged, just replace the file_uploader line with above two lines)

