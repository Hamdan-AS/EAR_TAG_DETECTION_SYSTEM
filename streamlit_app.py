import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import zipfile
import os
import tempfile
import easyocr

# --- DASHBOARD CONFIG ---
st.set_page_config(page_title="Cow Ear-Tag AI", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- CACHE MODELS TO PREVENT RELOADING ---
@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(model_path)

@st.cache_resource
def load_ocr_model():
    # gpu=False is recommended for free cloud tiers unless you have guaranteed GPU access
    return easyocr.Reader(['en'], gpu=False) 

# --- SIDEBAR ---
st.sidebar.header("🚀 AI Control Panel")
model_file = "cow_eartag_yolov8n_100ep_clean_best.pt"
conf_level = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.4)
st.sidebar.divider()

# --- MAIN INTERFACE ---
st.title("🐄 Livestock ID Dashboard")
st.subheader("Automated Ear-Tag Recognition System")

uploaded_zip = st.file_uploader("📂 Upload ZIP file of Cow Images", type=["zip"])

if uploaded_zip:
    # Load the models
    try:
        model = load_yolo_model(model_file)
        ocr_reader = load_ocr_model()
    except Exception as e:
        st.error(f"Error loading models. Ensure `{model_file}` is in your repo. Details: {e}")
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
        m2.metric("Pipeline", "YOLOv8 + EasyOCR")
        m3.metric("Status", "Processing...")

        st.divider()

        # Process and Show Results in a Grid
        for path in image_paths:
            img_name = os.path.basename(path)
            
            # Load original image using OpenCV for cropping
            orig_img = cv2.imread(path)
            
            # YOLO Inference
            results = model(path, conf=conf_level)
            
            # Create a box for each image result
            with st.expander(f"Analysis for {img_name}", expanded=True):
                col_img, col_data = st.columns([2, 1])
                
                # Plot YOLO bounding boxes
                res_plotted = results[0].plot()
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                with col_img:
                    st.image(res_rgb, use_container_width=True)
                
                with col_data:
                    st.write("### 🏷️ Detected EARTAG Str")
                    if len(results[0].boxes) > 0:
                        for i, box in enumerate(results[0].boxes):
                            confidence = box.conf[0]
                            st.success(f"**Tag {i+1} Found!**")
                            
                            # --- EASYOCR LOGIC ---
                            # 1. Get exact pixel coordinates of the bounding box
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # 2. Crop the ear tag from the original image
                            tag_crop = orig_img[y1:y2, x1:x2]
                            
                            # 3. Read the text from the cropped image
                            # detail=0 returns just the string instead of bounding boxes and confidences
                            ocr_result = ocr_reader.readtext(tag_crop, detail=0) 
                            
                            # 4. Clean up the output
                            if ocr_result:
                                extracted_text = " ".join(ocr_result)
                            else:
                                extracted_text = "UNREADABLE (Blur/Angle)"
                            
                            st.code(f"STR: {extracted_text}", language="text") 
                            st.progress(float(confidence), text=f"Detection Confidence: {confidence:.2%}")
                            
                            # Show the cropped tag to the user
                            # We convert BGR to RGB so Streamlit displays the colors correctly
                            st.image(cv2.cvtColor(tag_crop, cv2.COLOR_BGR2RGB), caption=f"Cropped Tag {i+1}", width=120)
                            st.divider()
                    else:
                        st.warning("No tags detected in this view.")

else:
    st.info("Waiting for ZIP upload. This dashboard will automatically parse and identify livestock IDs.")
