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
st.set_page_config(page_title="Cow Ear-Tag AI", layout="wide")

# Custom CSS for a clean, integrated look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stExpander"] { border: 1px solid #e6e9ef; border-radius: 10px; background-color: white; }
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 8px; border: 1px solid #eee; }
    </style>
    """, unsafe_allow_html=True)

# --- CACHE MODELS ---
@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(model_path)

@st.cache_resource
def load_ocr_model():
    return easyocr.Reader(['en'], gpu=False) 

# --- HELPER FUNCTION: Draw bboxes without OpenGL ---
def draw_boxes_pil(image_array, results, conf_threshold):
    """Draw YOLO boxes using PIL instead of cv2 (no OpenGL needed)"""
    pil_image = Image.fromarray(image_array)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(pil_image)
    
    for box in results[0].boxes:
        if box.conf[0] >= conf_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Draw rectangle (green color)
            draw.rectangle([x1, y1, x2, y2], outline='lime', width=3)
            # Draw confidence label
            conf_text = f"{float(box.conf[0]):.2f}"
            draw.text((x1, y1-10), conf_text, fill='lime')
    
    return np.array(pil_image)

# --- MAIN APP HEADER ---
st.title("🐄 Cattle Ear-Tag Detection")
st.markdown("Automated identification system for livestock monitoring.")

# --- INTEGRATED CONTROLS (Replacing Sidebar) ---
with st.container():
    c1, c2 = st.columns([2, 1])
    with c1:
        uploaded_zip = st.file_uploader("📂 Upload ZIP file or Single Images", type=["zip", "jpg", "jpeg", "png"], accept_multiple_files=False)
    with c2:
        conf_level = st.slider("Detection Confidence Threshold", 0.1, 1.0, 0.4)
        model_file = "cow_eartag_yolov8n_100ep_clean_best.pt"

st.divider()

if uploaded_zip:
    try:
        model = load_yolo_model(model_file)
        ocr_reader = load_ocr_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Create temporary directory to handle files
    with tempfile.TemporaryDirectory() as tmp_dir:
        image_paths = []
        
        # Handle ZIP vs Single Image
        if uploaded_zip.name.endswith(".zip"):
            with zipfile.ZipFile(uploaded_zip, "r") as z:
                z.extractall(tmp_dir)
            valid_exts = (".jpg", ".jpeg", ".png")
            image_paths = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.lower().endswith(valid_exts)]
        else:
            # Single Image Upload
            tfile = os.path.join(tmp_dir, uploaded_zip.name)
            with open(tfile, "wb") as f:
                f.write(uploaded_zip.getbuffer())
            image_paths = [tfile]

        # Dashboard Metrics
        m1, m2 = st.columns(2)
        m1.metric("Images to Process", len(image_paths))
        m2.metric("Status", "Active Scan")

        # --- PROCESSING LOOP ---
        for path in image_paths:
            img_name = os.path.basename(path)
            orig_img = cv2.imread(path)
            results = model(path, conf=conf_level)
            
            with st.expander(f"🔍 Analysis: {img_name}", expanded=True):
                col_img, col_data = st.columns([2, 1])
                
                # Draw boxes using PIL (no OpenGL required)
                orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                res_plotted = draw_boxes_pil(orig_rgb, results, conf_level)
                
                with col_img:
                    st.image(res_plotted, caption="Full Frame Detection", use_container_width=True)
                
                with col_data:
                    st.markdown("### 🏷️ EARTAG Str Results")
                    
                    if len(results[0].boxes) > 0:
                        for i, box in enumerate(results[0].boxes):
                            confidence = box.conf[0]
                            
                            # 1. Get Coordinates & Crop
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            tag_crop = orig_img[y1:y2, x1:x2]
                            tag_rgb = cv2.cvtColor(tag_crop, cv2.COLOR_BGR2RGB)
                            
                            # 2. OCR Extraction
                            ocr_result = ocr_reader.readtext(tag_crop, detail=0)
                            extracted_text = " ".join(ocr_result) if ocr_result else "TEXT UNREADABLE"
                            
                            # 3. Visual UI for results
                            st.info(f"**Tag {i+1} Found**")
                            
                            # SHOW THE CROP CLEARLY - This ensures the user sees the tag even if OCR fails
                            st.image(tag_rgb, caption=f"Cropped Tag {i+1} View", width=200)
                            
                            st.markdown(f"**Extracted ID:** `{extracted_text}`")
                            st.progress(float(confidence), text=f"Match Confidence: {confidence:.2%}")
                            st.divider()
                    else:
                        st.warning("No tags detected. Try lowering the Confidence Threshold above.")

else:
    st.info("Upload your livestock images to begin the identification process.")
