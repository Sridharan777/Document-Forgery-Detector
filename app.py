import streamlit as st
import streamlit_authenticator as stauth
import bcrypt
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
from fpdf import FPDF
import tempfile

# ---------------- Config ----------------
IMG_SIZE = 224
MODEL_PATH = "models/best_resnet50.pth"
FALLBACK_GDRIVE_ID = "1w4EufvzDfAeVpvL7hfyFdqOce67XV8ks"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
MAX_HEIGHT, MAX_WIDTH = 450, 350

# ---------- User credentials setup ----------
users = {
    "usernames": {
        "alice": {
            "name": "Alice",
            # bcrypt hash for password 'alicepass'
            "password": "$2b$12$ZiicYZGLRhPDa0/HQsSC5uTqNnaNtSBGTFDEa7BcP6JCsG36izQkG",
        },
        "bob": {
            "name": "Bob",
            # bcrypt hash for password 'bobpass'
            "password": "$2b$12$1Wgzm99QJMKm7NiO3TgDN.XoXcicUUuuCQxoRwoqc24k6LxjMn7aK",
        },
    }
}

authenticator = stauth.Authenticate(
    credentials=users,
    cookie_name="streamlit_auth",
    key="some_random_signature_key_for_security",
    cookie_expiry_days=1,
)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status:
    st.sidebar.write(f"Welcome {name}")
    if st.sidebar.button("Logout"):
        authenticator.logout("main")
        st.experimental_rerun()

    # Initialize per-user upload history and feedback
    if "upload_history" not in st.session_state:
        st.session_state.upload_history = {}
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}

    if username not in st.session_state.upload_history:
        st.session_state.upload_history[username] = []
    if username not in st.session_state.feedback:
        st.session_state.feedback[username] = {}

    # Helper: add uploaded files per user
    def add_to_history(filename):
        if filename not in st.session_state.upload_history[username]:
            st.session_state.upload_history[username].append(filename)

    # Helper: feedback per user
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

    # Theme CSS & tooltip (same as previous)
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

    apply_theme_css(st.sidebar.radio("Select Theme", ["Light", "Dark"], index=1, help="Switch between Light and Dark mode."))

    def tooltip(label, text):
        return f"""<span class="tooltip" style="border-bottom:1px dotted; cursor:help; position: relative;">{label}
            <span class="tooltiptext" style="visibility:hidden; opacity:0; width: 210px; background-color: #555; color: #fff; text-align: center;
            border-radius: 6px; padding: 5px 8px; position: absolute; z-index: 1;
            bottom: 125%; left: 50%; margin-left: -105px; font-size: 0.85em; transition: opacity 0.3s;">
            {text}</span>
            </span>
        """

    # Paste your existing helper functions here, e.g., load_model, predict_single, compute_gradcam, etc.
    # Modify upload history and feedback code to use per-user dicts, e.g.:
    # st.session_state.upload_history[username]
    # st.session_state.feedback[username]

    st.markdown(tooltip("Upload receipt image(s) 📁", "Allowed: PNG, JPG, JPEG. You can upload multiple images at once."), unsafe_allow_html=True)
    uploaded_files = st.file_uploader("", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        for f in uploaded_files:
            if f.name not in st.session_state.upload_history[username]:
                st.session_state.upload_history[username].append(f.name)

    with st.sidebar:
        st.header("Upload History 🕒")
        if st.session_state.upload_history[username]:
            for fname in st.session_state.upload_history[username]:
                st.markdown(f"• {fname}")
        else:
            st.write("No history yet.")

    # Model load, prediction display, grad-cam, PDF generation, feedback buttons, etc.
    # Use handle_feedback_local() and display_feedback_local() adapted above for feedback

else:
    if authentication_status is False:
        st.error("Username/password is incorrect")
    elif authentication_status is None:
        st.warning("Please enter your username and password")
    st.stop()
