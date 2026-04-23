import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
from PIL import Image
from utils.preprocessing import preprocess_image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Poultry Egg Classifier",
    page_icon="🥚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF !important;
    }
    
    [data-testid="stMainBlockContainer"] {
        padding: 0px !important;
    }
    
    .main {
        background: linear-gradient(135deg, #FFFFFF 0%, #F7F9FC 100%);
    }
    
    /* Hero Header */
    .hero-header {
        background: linear-gradient(135deg, #FF6B35 0%, #FF8555 100%);
        color: white;
        padding: 60px 40px;
        border-radius: 0;
        text-align: center;
        box-shadow: 0 4px 20px rgba(255, 107, 53, 0.15);
        margin-bottom: 40px;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 12px;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        color: white !important;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        opacity: 0.95;
        font-weight: 300;
        letter-spacing: 0.5px;
        color: white !important;
    }
    
    /* Upload Section */
    .upload-section {
        background: white;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin-bottom: 40px;
        border: 2px dashed #FF6B35;
    }
    
    .upload-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1F1F1F;
        margin-bottom: 8px;
    }
    
    .upload-hint {
        font-size: 0.95rem;
        color: #666;
        margin-bottom: 20px;
    }
    
    /* Main Content Grid */
    .content-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 40px;
        margin-bottom: 40px;
    }
    
    /* Image Container */
    .image-container {
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        text-align: center;
    }
    
    .image-container h3 {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 20px;
        color: #1F1F1F;
    }
    
    /* Results Container */
    .results-container {
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    
    .results-container h3 {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 20px;
        color: #1F1F1F;
    }
    
    /* Analyze Button */
    .stButton>button {
        background: linear-gradient(135deg, #FF6B35 0%, #FF8555 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        height: 55px !important;
        border: none !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #E55A2B 0%, #FF7540 100%) !important;
        box-shadow: 0 6px 25px rgba(255, 107, 53, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Result Card - Main */
    .result-card-main {
        background: #FFFFFF;
        border: 3px solid #2ECC71;
        border-radius: 15px;
        padding: 40px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.15);
    }
    
    .result-card-main.dead {
        background: #FFFFFF;
        border-color: #E74C3C;
        box-shadow: 0 4px 15px rgba(231, 76, 60, 0.15);
    }
    
    .result-card-main.infertile {
        background: #FFFFFF;
        border-color: #F39C12;
        box-shadow: 0 4px 15px rgba(243, 156, 18, 0.15);
    }
    
    .result-status-label {
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #666 !important;
        margin-bottom: 10px;
    }
    
    .result-status-value {
        font-size: 3.5rem;
        font-weight: 900;
        margin: 15px 0;
        line-height: 1;
        color: #2ECC71 !important;
    }
    
    .result-status-value.fertile {
        color: #2ECC71 !important;
    }
    
    .result-status-value.dead {
        color: #E74C3C !important;
    }
    
    .result-status-value.infertile {
        color: #F39C12 !important;
    }
    
    .result-confidence {
        font-size: 1.3rem;
        font-weight: 600;
        color: #333 !important;
        margin-top: 15px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #F8F9FA;
        padding: 24px;
        border-radius: 12px;
        border-left: 5px solid #FF6B35;
        margin: 15px 0;
    }
    
    .metric-card.success {
        border-left-color: #2ECC71;
        background: #FFFFFF;
        box-shadow: 0 2px 10px rgba(46, 204, 113, 0.1);
    }
    
    .metric-card.warning {
        border-left-color: #F39C12;
        background: #FFFFFF;
        box-shadow: 0 2px 10px rgba(243, 156, 18, 0.1);
    }
    
    .metric-card.danger {
        border-left-color: #E74C3C;
        background: #FFFFFF;
        box-shadow: 0 2px 10px rgba(231, 76, 60, 0.1);
    }
    
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #888 !important;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FF6B35 !important;
    }
    
    .metric-card.success .metric-value {
        color: #2ECC71 !important;
    }
    
    .metric-card.warning .metric-value {
        color: #F39C12 !important;
    }
    
    .metric-card.danger .metric-value {
        color: #E74C3C !important;
    }
    
    /* Probability Display */
    .probability-section {
        margin-top: 30px;
        padding-top: 30px;
        border-top: 2px solid #EFEFEF;
    }
    
    .probability-section h4 {
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 20px;
        color: #1F1F1F;
    }
    
    .prob-item-container {
        background: #F8F9FA;
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .prob-name {
        font-weight: 600;
        color: #1F1F1F !important;
        font-size: 1rem;
    }
    
    .prob-bar-container {
        flex: 1;
        margin: 0 20px;
        background: white;
        border-radius: 10px;
        overflow: hidden;
        height: 8px;
        border: 1px solid #E0E0E0;
    }
    
    .prob-bar {
        height: 100%;
        background: linear-gradient(90deg, #FF6B35, #FF8555);
        transition: width 0.3s ease;
    }
    
    .prob-percent {
        font-weight: 700;
        color: #FF6B35 !important;
        font-size: 1rem;
        min-width: 50px;
        text-align: right;
    }
    
    /* Status Messages */
    .status-success {
        background: linear-gradient(135deg, #F0FFF4 0%, #E8F8F5 100%);
        border-left: 5px solid #2ECC71;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #FFFBEA 0%, #FEF9E7 100%);
        border-left: 5px solid #F39C12;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    
    .status-error {
        background: linear-gradient(135deg, #FEF5E7 0%, #FADBD8 100%);
        border-left: 5px solid #E74C3C;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    
    /* Footer */
    .footer-section {
        text-align: center;
        padding: 40px 20px;
        color: #888;
        font-size: 0.9rem;
        border-top: 1px solid #E0E0E0;
        margin-top: 60px;
    }
    
    .footer-section p {
        margin: 8px 0;
    }
    
    .footer-section strong {
        color: #FF6B35;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FAFBFC 0%, #F5F7FA 100%) !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        padding: 0 20px;
    }
    
    /* File Uploader Styling */
    [data-testid="stFileUploadDropzone"] {
        padding: 30px !important;
        border-radius: 15px !important;
        border: 2px dashed #FF6B35 !important;
        background: #FAFBFC !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #FF6B35, #FF8555) !important;
    }
    
    /* Spinner */
    .stSpinner {
        color: #FF6B35 !important;
    }
    
    /* General text color fixes */
    body, div, p, span, h1, h2, h3, h4, h5, h6 {
        color: #1F1F1F !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CACHED MODEL LOADING ---
@st.cache_resource
def load_models():
    """Load both detection and classification models."""
    try:
        detector = tf.keras.models.load_model('egg_detector_final.h5')
        classifier = tf.keras.models.load_model('fertility_classifier_final.h5')
        
        with open('label_mapping.json', 'r') as f:
            mapping = json.load(f)
        # Reverse mapping: {index: class_name}
        reverse_mapping = {int(v): k for k, v in mapping.items()}
        
        return detector, classifier, reverse_mapping
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# --- UI COMPONENTS ---
def sidebar_content():
    with st.sidebar:
        # Logo/Icon
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("https://cdn-icons-png.flaticon.com/512/3474/3474345.png", width=80)
        
        st.markdown("---")
        
        # About Section
        st.markdown("### 🔬 About This System")
        st.markdown("""
        This **AI-powered egg analysis system** combines computer vision and deep learning to:
        
        - **Detect** if an image contains a poultry egg
        - **Classify** the fertility status accurately
        
        Perfect for hatchery management and quality control.
        """)
        
        st.markdown("---")
        
        # Model Details
        with st.expander("📊 Technical Details", expanded=False):
            st.markdown("""
            **Model Architecture:**
            - Convolutional Neural Networks (CNN)
            - Two-stage pipeline for accuracy
            
            **Input Specifications:**
            - Image Size: 224×224 pixels
            - Format: RGB (3 channels)
            - Normalized to [0, 1] range
            
            **Classification Classes:**
            - 🟢 **Fertile** - Viable for incubation
            - 🔴 **Dead** - Not viable
            - 🟡 **Infertile** - Cannot develop
            """)
        
        st.markdown("---")
        
        # Instructions
        with st.expander("📋 How to Use", expanded=False):
            st.markdown("""
            1. **Upload** a clear image of the egg
            2. **Click** "Analyze Egg" button
            3. **View** detection & classification results
            4. **Check** confidence scores
            
            **Tips for best results:**
            - Use good lighting
            - Single egg per image
            - Clear, direct view
            - Minimum 224×224 resolution
            """)
        
        st.markdown("---")
        
        # Footer Info
        st.markdown("""
        <div style="text-align: center; color: #888; font-size: 12px;">
            <p><strong>PoultryTech Solutions</strong></p>
            <p>Version 1.0.0 • 2026</p>
            <p style="margin-top: 12px;">Powered by TensorFlow & Streamlit</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    sidebar_content()
    
    # --- HERO HEADER ---
    st.markdown("""
        <div class="hero-header">
            <div class="hero-title">🥚 Poultry Egg Analyzer</div>
            <div class="hero-subtitle">Advanced AI Detection & Fertility Classification</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Load Models
    with st.spinner("🔄 Loading AI Models..."):
        detector, classifier, label_map = load_models()
    
    if not detector or not classifier:
        st.stop()

    # --- UPLOAD SECTION ---
    st.markdown("""
        <div class="upload-section">
            <div class="upload-title">📷 Upload Egg Image</div>
            <div class="upload-hint">Select a clear JPG or PNG image of a single egg</div>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        label="Choose image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            # --- MAIN CONTENT GRID ---
            st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)
            
            col_image, col_analysis = st.columns([1, 1.2], gap="large")
            
            # IMAGE COLUMN
            with col_image:
                st.markdown("""
                    <div class="image-container">
                        <h3>📸 Image Preview</h3>
                """, unsafe_allow_html=True)
                
                st.image(image, use_column_width=True)
                st.caption(f"**File:** {uploaded_file.name} | **Size:** {image.size[0]}×{image.size[1]}px")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # ANALYSIS COLUMN
            with col_analysis:
                st.markdown("""
                    <div class="results-container">
                        <h3>🔬 Analysis Results</h3>
                """, unsafe_allow_html=True)
                
                st.markdown("Click to analyze the egg image:")
                
                # ANALYZE BUTTON
                if st.button("🚀 ANALYZE EGG", use_container_width=True):
                    with st.spinner("⏳ Processing..."):
                        # Preprocess
                        processed_img = preprocess_image(image)
                        
                        # --- STAGE 1: DETECTION ---
                        det_pred = detector.predict(processed_img, verbose=0)
                        is_egg_prob = det_pred[0][0]
                        
                        if det_pred.shape[1] == 1:
                            is_egg = is_egg_prob > 0.5
                            egg_confidence = is_egg_prob if is_egg else (1 - is_egg_prob)
                        else:
                            is_egg = np.argmax(det_pred) == 0
                            egg_confidence = np.max(det_pred)

                        if not is_egg:
                            # NO EGG DETECTED
                            st.markdown(f"""
                                <div class="metric-card danger">
                                    <div class="metric-label">❌ Detection Status</div>
                                    <div class="metric-value">NO EGG</div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                                <div class="metric-card danger">
                                    <div class="metric-label">Detection Confidence</div>
                                    <div class="metric-value">{egg_confidence:.1%}</div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            st.warning("⚠️ No poultry egg detected. Please upload a clear egg image.")
                        
                        else:
                            # EGG DETECTED - PROCEED TO CLASSIFICATION
                            st.markdown(f"""
                                <div class="metric-card success">
                                    <div class="metric-label">✅ Detection Status</div>
                                    <div class="metric-value">EGG FOUND</div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                                <div class="metric-card success">
                                    <div class="metric-label">Detection Confidence</div>
                                    <div class="metric-value">{egg_confidence:.1%}</div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            st.success("✅ Analyzing fertility status...")
                            
                            # --- STAGE 2: CLASSIFICATION ---
                            cls_pred = classifier.predict(processed_img, verbose=0)[0]
                            cls_idx = np.argmax(cls_pred)
                            cls_name = label_map.get(cls_idx, "Unknown").lower()
                            cls_conf = cls_pred[cls_idx]
                            
                            # Result emoji
                            result_emoji = "🟢" if cls_name == "fertile" else ("🔴" if cls_name == "dead" else "🟡")
                            
                            # Main Result Card
                            st.markdown(f"""
                                <div class="result-card-main {cls_name}">
                                    <div class="result-status-label">Fertility Classification</div>
                                    <div class="result-status-value {cls_name}">{result_emoji} {cls_name.upper()}</div>
                                    <div class="result-confidence">Confidence: <strong>{cls_conf:.1%}</strong></div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Probability breakdown
                            st.markdown('<div class="probability-section">', unsafe_allow_html=True)
                            st.markdown('<h4>Classification Probabilities</h4>', unsafe_allow_html=True)
                            
                            top_indices = np.argsort(cls_pred)[::-1]
                            for idx in top_indices:
                                name = label_map.get(idx, f"Class {idx}").capitalize()
                                prob = cls_pred[idx]
                                
                                emoji_map = {"Fertile": "🟢", "Dead": "🔴", "Infertile": "🟡"}
                                emoji = emoji_map.get(name, "⚪")
                                
                                st.markdown(f"""
                                    <div class="prob-item-container">
                                        <span class="prob-name">{emoji} {name}</span>
                                        <div class="prob-bar-container">
                                            <div class="prob-bar" style="width: {prob*100}%"></div>
                                        </div>
                                        <span class="prob-percent">{prob:.1%}</span>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        except Exception as e:
            st.markdown("""
                <div class="status-error">
                    <strong>❌ Error:</strong> Failed to process image. Please ensure it's a valid JPG or PNG.
                </div>
            """, unsafe_allow_html=True)
    
    else:
        # NO IMAGE UPLOADED STATE
        st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("""
            ### 📝 Quick Guide
            
            **Simple Steps:**
            1. Upload an egg image
            2. Click "Analyze Egg"
            3. Get instant results
            
            **You'll receive:**
            - ✅ Egg detection status
            - 📊 Fertility classification  
            - 📈 Confidence scores
            - 🎯 Detailed probabilities
            """)
        
        with col2:
            st.markdown("""
            ### 💡 Tips for Best Results
            
            - **Lighting:** Use bright, even lighting
            - **Angle:** Photograph from directly above
            - **Single:** One egg per image
            - **Focus:** Keep egg in center and sharp
            - **Quality:** 224×224px minimum
            """)
        
        st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)
    
    # --- FOOTER ---
    st.markdown("""
        <div class="footer-section">
            <p><strong>Poultry Egg Analyzer v1.0</strong></p>
            <p>Powered by TensorFlow & Streamlit • © 2026 PoultryTech Solutions</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
