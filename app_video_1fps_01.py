import streamlit as st
import os
import gdown
import pandas as pd
import numpy as np
import cv2
from PIL import Image

st.set_page_config(layout="wide", page_title="Video Analysis Browser")

# --- INITIALIZATION ---
BASE_MODEL_DIR = os.path.join(os.getcwd(), "all_models")
SECRET_FOLDER_ID = st.secrets["drive_folder_id"]

@st.cache_resource
def setup_environment(drive_folder_id):
    # 1. Patch signal module to prevent thread crashes
    import signal
    import sys
    
    def dummy_signal_handler(signum, frame): pass
    original_signal = signal.signal
    def patched_signal(signalnum, handler):
        try: return original_signal(signalnum, handler)
        except ValueError: return dummy_signal_handler
    signal.signal = patched_signal

    # 2. Setup environment and imports
    os.environ["ULTRALYTICS_HUB_DISABLED"] = "true"
    from ultralytics import YOLO
    from transformers import pipeline
    from tensorflow import keras
    
    if not os.path.exists(BASE_MODEL_DIR):
        gdown.download_folder(id=drive_folder_id, output=BASE_MODEL_DIR, quiet=True)

    try:
        yolo = YOLO(os.path.join(BASE_MODEL_DIR, "yolo/yolov8n.pt"))
        emotion_pipe = pipeline("image-classification", model=os.path.join(BASE_MODEL_DIR, "emotion"))
        gender_pipe = pipeline("image-classification", model=os.path.join(BASE_MODEL_DIR, "gender"))
        age_model = keras.models.load_model(os.path.join(BASE_MODEL_DIR, "age/best_model.h5"), compile=False)
        return yolo, emotion_pipe, gender_pipe, age_model
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        return None

# --- MAIN APP ---
st.title("🎥 Video Face Analysis Browser")
models = setup_environment(SECRET_FOLDER_ID)

if models:
    yolo, emotion_pipe, gender_pipe, age_model = models
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

    if uploaded_video:
        # Save temp file
        with open("temp_vid.mp4", "wb") as f: f.write(uploaded_video.read())
        
        # Initialize session state for storage
        if 'processed_frames' not in st.session_state:
            with st.spinner("Processing video..."):
                cap = cv2.VideoCapture("temp_vid.mp4")
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames_data = {}
                frame_idx = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    frame_idx += 1
                    if frame_idx % 30 != 0: continue 
                    
                    results = yolo(frame, classes=[0], verbose=False)
                    coords = [list(map(int, b.xyxy[0])) for b in results[0].boxes]
                    
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frame_results = []
                    
                    for i, (x1, y1, x2, y2) in enumerate(coords):
                        face_id = i + 1
                        crop = pil_img.crop((x1, y1, x2, y2))
                        
                        # ... (your prediction logic remains the same) ...
                        
                        # Draw Rectangle around person (Original Orange)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                        
                        # Draw ID Label with Blue Box and White Text
                        label = f"ID: {face_id}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1.0 # Increased font size
                        thickness = 2
                        
                        # Calculate text size for the background box
                        (w, h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                        
                        # Draw filled blue rectangle for the ID background
                        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (255, 0, 0), -1)
                        
                        # Draw white text on top
                        cv2.putText(frame, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
                
                st.session_state['processed_frames'] = frames_data
                cap.release()

        # --- VIEWING INTERFACE ---
        if 'processed_frames' in st.session_state and st.session_state['processed_frames']:
            frame_keys = list(st.session_state['processed_frames'].keys())
            selection = st.selectbox("Select a frame to inspect:", frame_keys)
            
            frame_img, frame_data = st.session_state['processed_frames'][selection]
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.markdown("### Frame Results")
                if frame_data:
                    df = pd.DataFrame(frame_data)
                    # Force ID as a column and set as index for display
                    st.table(df[['ID', 'Age', 'Emotion', 'Gender']].set_index('ID'))
                else:
                    st.info("No faces detected.")
