import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import zipfile
import os
import tempfile

# --- DASHBOARD CONFIG ---
st.set_page_config(page_title="Cow Ear-Tag AI", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.header("🚀 AI Control Panel")
# Using your exact model filename from the repo
model_file = "cow_eartag_yolov8n_100ep_clean_best.pt"
conf_level = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.4)

st.sidebar.divider()
st.sidebar.info("**Beginner Project Value:** This app demonstrates the full 'Edge-to-Cloud' pipeline: capturing barn data, processing with YOLOv8, and serving it via a web dashboard.")

# --- MAIN INTERFACE ---
st.title("🐄 Livestock ID Dashboard")
st.subheader("Automated Ear-Tag Recognition System")

uploaded_zip = st.file_uploader("📂 Upload ZIP file of Cow Images", type=["zip"])

if uploaded_zip:
    # Load the model
    try:
        model = YOLO(model_file)
    except Exception as e:
        st.error(f"Could not find `{model_file}`. Make sure it is in your GitHub root folder.")
        st.stop()

    # Create temporary directory to unzip files
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(uploaded_zip, "r") as z:
            z.extractall(tmp_dir)
        
        # Get list of images
        valid_exts = (".jpg", ".jpeg", ".png")
        image_paths = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.lower().endswith(valid_exts)]
        
        # Dashboard Overview Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Images", len(image_paths))
        m2.metric("Model Version", "YOLOv8n (100 Epochs)")
        m3.metric("Status", "Processing...")

        st.divider()

        # Process and Show Results in a Grid
        for path in image_paths:
            img_name = os.path.basename(path)
            results = model(path, conf=conf_level)
            
            # Create a box for each image result
            with st.expander(f"Analysis for {img_name}", expanded=True):
                col_img, col_data = st.columns([2, 1])
                
                # Inference result
                res_plotted = results[0].plot()
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                with col_img:
                    st.image(res_rgb, use_container_width=True)
                
                with col_data:
                    st.write("### 🏷️ Detected EARTAG Str")
                    if len(results[0].boxes) > 0:
                        for i, box in enumerate(results[0].boxes):
                            # Placeholder logic for the string ID (normally OCR would go here)
                            # In a production app, you'd crop the tag and pass to EasyOCR
                            confidence = box.conf[0]
                            st.success(f"**Tag {i+1} Found!**")
                            st.code(f"STR: [SCANNING_ID...]", language="text") 
                            st.progress(float(confidence), text=f"Confidence: {confidence:.2%}")
                    else:
                        st.warning("No tags detected in this view.")

else:
    st.info("Waiting for ZIP upload. This dashboard will automatically parse and identify livestock IDs.")
