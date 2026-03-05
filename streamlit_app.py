import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import zipfile
import os
import tempfile

# --- DASHBOARD CONFIG ---
st.set_page_config(page_title="Livestock ID Dashboard", layout="wide")

st.sidebar.title("⚙️ Model Configuration")
model_path = st.sidebar.text_input("Enter YOLOv8 Model Path", "yolov8n.pt") # Defaulting to Nano
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)

# --- APP HEADER ---
st.title("🐄 Livestock Monitoring Dashboard")
st.markdown("### Ear Tag Detection & Identification")
st.divider()

# --- FILE UPLOAD ---
uploaded_zip = st.file_uploader("Upload Image ZIP File", type=["zip"])

if uploaded_zip is not None:
    # Load Model
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Create columns for the dashboard layout
    col1, col2 = st.columns([3, 1])

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Extract ZIP
        with zipfile.ZipFile(uploaded_zip, "r") as z:
            z.extractall(tmp_dir)
        
        image_files = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        st.success(f"Found {len(image_files)} images in ZIP.")

        # --- PROCESSING LOOP ---
        for img_path in image_files:
            results = model(img_path, conf=conf_threshold)
            
            # Display logic
            for r in results:
                im_array = r.plot()  # plot a BGR numpy array of predictions
                im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
                
                with st.container():
                    c_main, c_meta = st.columns([2, 1])
                    
                    with c_main:
                        st.image(im_rgb, caption=f"Processing: {os.path.basename(img_path)}", use_container_width=True)
                    
                    with c_meta:
                        st.subheader("Detected Tags")
                        # Placeholder for Ear Tag Strings (Logic: usually OCR after detection)
                        # Example strings based on your provided images:
                        mock_tags = ["40251", "2978", "67980", "2058"] 
                        
                        st.metric(label="Total Detections", value=len(r.boxes))
                        
                        for i, box in enumerate(r.boxes):
                            # In a real app, you would crop 'box' and pass it to EasyOCR
                            # Here we show how to display it clearly
                            tag_id = mock_tags[i % len(mock_tags)] # Mocking OCR output
                            st.info(f"📍 Tag {i+1} String: **{tag_id}**")
                    st.divider()

else:
    st.info("Please upload a ZIP file containing the cow images to begin analysis.")
    st.image("cow1103.jpg") # Displaying one of your images as a preview if available locally
