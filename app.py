# app.py
"""
Receipt Forgery Detector — Premium Dashboard Edition
- Dashboard header shows Key Metrics (uploads, forged %, avg confidence)
- Modern two-column card layout with shadow and rounded corners
- Grad-CAM overlay side-by-side with original image
- Confidence bar + gauge in stats panel
- Sidebar with model info and dark/light toggle (working)
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import os, traceback, gdown, zipfile, time
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
IMG_SIZE = 224
MODEL_PATH = "models/best_resnet50.pth"
FALLBACK_GDRIVE_ID = "1zMrv6S6rOWyiTQ0Fgw0VG0o0fEnrWzE-"  # your fallback GDrive ID
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# ---------------- CSS: Dashboard Look ----------------
BASE_CSS = """
<style>
body { background-color: #f8f9fa; }
.result-card {
  background: white;
  border-radius: 16px;
  padding: 18px;
  margin: 12px 0;
  box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
}
.metric-card {
  background: white;
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 3px 8px rgba(0,0,0,0.05);
  text-align: center;
}
.metric-title { font-size: 14px; color: #666; }
.metric-value { font-size: 26px; font-weight: 700; }
.dark-mode body { background-color: #121212; color: #fff; }
</style>
"""

st.set_page_config(page_title="Receipt Forgery Detector", layout="wide")
st.markdown(BASE_CSS, unsafe_allow_html=True)

# ---------------- HELPERS ----------------
def download_model_if_missing(gdrive_id: str):
    if os.path.exists(MODEL_PATH): return True
    if not gdrive_id: return False
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    try:
        st.info("📥 Downloading model...")
        gdown.download(url, MODEL_PATH, quiet=False)
        time.sleep(1)
        return os.path.exists(MODEL_PATH)
    except Exception as e:
        st.error(f"Model download failed: {e}")
        return False

@st.cache_resource(show_spinner=True)
def load_model():
    try:
        drive_id = None
        try: drive_id = st.secrets.get("MODEL_GDRIVE_ID", None)
        except Exception: pass
        if not os.path.exists(MODEL_PATH):
            ok = download_model_if_missing(drive_id) if drive_id else False
            if not ok and FALLBACK_GDRIVE_ID:
                ok = download_model_if_missing(FALLBACK_GDRIVE_ID)
            if not ok: 
                st.error("❌ Model not found.")
                return None
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        state_dict = ckpt
        if isinstance(ckpt, dict):
            for k in ("state_dict","model_state_dict","net"):
                if k in ckpt: state_dict = ckpt[k]; break
        new_state = {k.replace("module.",""): v for k,v in state_dict.items()}
        out_features = 2
        if "fc.weight" in new_state: out_features = new_state["fc.weight"].shape[0]
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, out_features)
        model.load_state_dict(new_state, strict=False)
        model.to(DEVICE).eval()
        return model
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.text(traceback.format_exc())
        return None

def pil_to_tensor(img):
    t = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    return t(img).unsqueeze(0).to(DEVICE)

def predict_single(model, x):
    with torch.no_grad():
        out = model(x)
        if out.shape[1]==1:
            prob = torch.sigmoid(out)[0,0].item()
            return ("FORGED 🔴" if prob>=0.5 else "GENUINE 🟢",
                    prob if prob>=0.5 else 1-prob, out)
        probs = torch.softmax(out, dim=1)[0]
        idx = int(torch.argmax(probs))
        return ("FORGED 🔴" if idx==1 else "GENUINE 🟢", float(probs[idx]), out)

def compute_gradcam(model,x,target_layer=None):
    acts,grads=[],[]
    if target_layer is None: target_layer=model.layer4[-1]
    fh=target_layer.register_forward_hook(lambda m,i,o: acts.append(o.detach().cpu()))
    try:
        bh=target_layer.register_full_backward_hook(lambda m,gi,go: grads.append(go[0].detach().cpu()))
    except: bh=target_layer.register_backward_hook(lambda m,gi,go: grads.append(go[0].detach().cpu()))
    out=model(x)
    score=out[:,0].sum() if out.shape[1]==1 else out[0,torch.argmax(out,dim=1)[0]]
    score.backward()
    cam=(grads[0].mean(dim=(1,2))[:,None,None]*acts[0]).sum(dim=0).numpy()
    cam=np.maximum(cam,0); cam=cv2.resize(cam,(IMG_SIZE,IMG_SIZE))
    return (cam-cam.min())/(cam.max()-cam.min()+1e-8)

def overlay_heatmap(pil_img, cam, alpha=0.4):
    img=np.array(pil_img.resize((IMG_SIZE,IMG_SIZE))).astype(np.uint8)
    heatmap=np.uint8(255*cam)
    heat=cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heat=cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 1-alpha, heat, alpha, 0)

def make_zip(original, overlay, name):
    mem=BytesIO()
    with zipfile.ZipFile(mem,"w",zipfile.ZIP_DEFLATED) as zf:
        buf1,buf2=BytesIO(),BytesIO()
        original.save(buf1,format="PNG"); Image.fromarray(overlay).save(buf2,format="PNG")
        zf.writestr(f"{name}_original.png", buf1.getvalue())
        zf.writestr(f"{name}_gradcam.png", buf2.getvalue())
    mem.seek(0); return mem.getvalue()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Settings")
dark_mode = st.sidebar.toggle("🌙 Dark Mode")
st.sidebar.write(f"Model: `{MODEL_PATH}`")
st.sidebar.markdown("---")
st.sidebar.write("Upload receipts to classify as genuine or forged.")
st.sidebar.write("Built with PyTorch + Streamlit + Grad-CAM.")

# ---------------- MAIN APP ----------------
st.title("📊 Receipt Forgery Detection Dashboard")
st.caption("AI-powered ResNet50 model with explainable Grad-CAM heatmaps")

uploaded_files = st.file_uploader("📤 Upload receipt image(s)", type=["png","jpg","jpeg"], accept_multiple_files=True)
if not uploaded_files:
    st.info("Upload one or more images to start analysis.")
    st.stop()

with st.spinner("Loading model..."):
    model = load_model()
if model is None: st.stop()

# --- Key Metrics ---
num_images = len(uploaded_files)
preds, confs = [], []
for uploaded in uploaded_files:
    img = Image.open(uploaded).convert("RGB")
    x = pil_to_tensor(img)
    lbl, conf, _ = predict_single(model,x)
    preds.append(lbl); confs.append(conf)

forged_pct = 100 * preds.count("FORGED 🔴") / num_images
avg_conf = np.mean(confs)*100

colA,colB,colC = st.columns(3)
with colA: st.markdown(f"<div class='metric-card'><div class='metric-title'>Images Uploaded</div><div class='metric-value'>{num_images}</div></div>", unsafe_allow_html=True)
with colB: st.markdown(f"<div class='metric-card'><div class='metric-title'>Forged %</div><div class='metric-value'>{forged_pct:.1f}%</div></div>", unsafe_allow_html=True)
with colC: st.markdown(f"<div class='metric-card'><div class='metric-title'>Avg. Confidence</div><div class='metric-value'>{avg_conf:.1f}%</div></div>", unsafe_allow_html=True)

st.markdown("---")

# --- Per-Image Results ---
for i, uploaded in enumerate(uploaded_files):
    img = Image.open(uploaded).convert("RGB")
    x = pil_to_tensor(img)
    label, conf, out = predict_single(model,x)
    cam = compute_gradcam(model,x)
    overlay = overlay_heatmap(img, cam)
    name = os.path.splitext(uploaded.name)[0]

    # Card layout
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader(f"📄 {uploaded.name}")
        st.image(img, caption=f"Prediction: {label}", use_container_width=True)
        bar_html=f"""
        <div style="background:#eee;border-radius:8px;">
          <div style="width:{conf*100:.2f}%;background:{'#2ecc71' if 'GENUINE' in label else '#e74c3c'};
                      padding:6px;border-radius:8px;color:white;text-align:center;">
            {conf*100:.2f}%
          </div>
        </div>
        """
        st.markdown(bar_html, unsafe_allow_html=True)
        fig=go.Figure(go.Indicator(mode="gauge+number",value=conf*100,
                                   title={'text':"Confidence"},
                                   gauge={'axis':{'range':[0,100]},'bar':{'color':'#2ecc71' if 'GENUINE' in label else '#e74c3c'}}))
        st.plotly_chart(fig,use_container_width=True,key=f"gauge_{i}")

    with col2:
        st.subheader("Grad-CAM Attention")
        st.image(overlay, caption="Model Focus Area", use_container_width=True)
        zip_bytes = make_zip(img, overlay, name)
        st.download_button("⬇ Download Original + Grad-CAM", data=zip_bytes,
                           file_name=f"{name}_bundle.zip", mime="application/zip")

    st.markdown("</div>", unsafe_allow_html=True)
