import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import gdown
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0d;
    color: #f0f0f0;
}

.main { background-color: #0d0d0d; }

h1, h2, h3 {
    font-family: 'Space Mono', monospace;
}

.header-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #00d4ff33;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 0 40px #00d4ff22;
}

.header-box h1 {
    font-size: 2rem;
    color: #00d4ff;
    margin: 0;
    letter-spacing: -1px;
}

.header-box p {
    color: #aaa;
    margin-top: 0.5rem;
    font-size: 0.95rem;
}

.result-tumor {
    background: linear-gradient(135deg, #3d0000, #1a0000);
    border: 2px solid #ff4444;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 0 30px #ff444433;
}

.result-no-tumor {
    background: linear-gradient(135deg, #003d1a, #001a0d);
    border: 2px solid #00ff88;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 0 30px #00ff8833;
}

.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
}

.confidence-text {
    font-size: 1rem;
    color: #ccc;
    margin-top: 0.5rem;
}

.info-card {
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #aaa;
}

.stButton > button {
    background: linear-gradient(90deg, #00d4ff, #0066ff);
    color: #000;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-size: 1rem;
    width: 100%;
    transition: opacity 0.2s;
}

.stButton > button:hover { opacity: 0.85; }

.uploadedFile { display: none; }

div[data-testid="stFileUploader"] {
    background: #1a1a1a;
    border: 2px dashed #333;
    border-radius: 12px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🧠 Brain Tumor Detector</h1>
    <p>CNN-powered MRI analysis · Upload a scan · Get instant prediction</p>
</div>
""", unsafe_allow_html=True)

# ── Load Model (auto-download from Google Drive if not present) ───────────────
MODEL_PATH = "brain_tumor_model.keras"
GDRIVE_FILE_ID = "1mzVeqDD_TmlSKt-NMwwl05bOAfNPFP5n"

@st.cache_resource
def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏳ Downloading model... (first time only, ~38MB)"):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

try:
    model = load_cnn_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown("### 📤 Upload MRI Image")
uploaded_file = st.file_uploader(
    "Drag & drop or click to browse",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Uploaded Scan**")
        img_display = Image.open(uploaded_file)
        st.image(img_display, use_container_width=True)

    with col2:
        st.markdown("**Prediction**")

        # Preprocess
        img_resized = img_display.convert("RGB").resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Analyzing..."):
            result = model.predict(img_array)
            confidence = float(result[0][0])

        if confidence > 0.5:
            label = "TUMOR DETECTED"
            conf_pct = confidence * 100
            color_class = "result-tumor"
            emoji = "⚠️"
        else:
            label = "NO TUMOR"
            conf_pct = (1 - confidence) * 100
            color_class = "result-no-tumor"
            emoji = "✅"

        st.markdown(f"""
        <div class="{color_class}">
            <p class="result-label">{emoji} {label}</p>
            <p class="confidence-text">Confidence: <strong>{conf_pct:.1f}%</strong></p>
            <p class="confidence-text">Raw score: {confidence:.4f}</p>
        </div>
        """, unsafe_allow_html=True)

    # Info
    st.markdown("""
    <div class="info-card">
    ⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only. 
    It is not a substitute for professional medical diagnosis. 
    Always consult a qualified radiologist or neurologist.
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="info-card">
    💡 Upload a brain MRI image (JPG or PNG) to get a Tumor / No Tumor prediction with confidence score.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#555; font-size:0.8rem; font-family:Space Mono'>Built with TensorFlow · CNN · Streamlit</p>",
    unsafe_allow_html=True
)