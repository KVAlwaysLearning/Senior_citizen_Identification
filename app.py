import streamlit as st
import os
import gdown
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from tensorflow import keras
from transformers import pipeline
from ultralytics import YOLO

st.set_page_config(layout="wide", page_title="Video Face Analysis")

# --- INITIALIZATION ---
BASE_MODEL_DIR = os.path.join(os.getcwd(), "all_models")
SECRET_FOLDER_ID = st.secrets["drive_folder_id"]

@st.cache_resource
def setup_environment(drive_folder_id):
    if not os.path.exists(BASE_MODEL_DIR):
        gdown.download_folder(id=drive_folder_id, output=BASE_MODEL_DIR, quiet=True)
    yolo = YOLO(os.path.join(BASE_MODEL_DIR, "yolo/yolov8n.pt"))
    emotion_pipe = pipeline("image-classification", model=os.path.join(BASE_MODEL_DIR, "emotion"))
    gender_pipe = pipeline("image-classification", model=os.path.join(BASE_MODEL_DIR, "gender"))
    age_model = keras.models.load_model(os.path.join(BASE_MODEL_DIR, "age/best_model.h5"), compile=False)
    return yolo, emotion_pipe, gender_pipe, age_model

# --- UI & LOGIC ---
st.title("🎥 Video Face Analysis Dashboard")
models = setup_environment(SECRET_FOLDER_ID)

if models:
    yolo, emotion_pipe, gender_pipe, age_model = models
    uploaded_video = st.file_uploader("Upload a video for analysis", type=["mp4", "mov", "avi"])

    if uploaded_video:
        # Save video temporarily to process with OpenCV
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())
        
        cap = cv2.VideoCapture("temp_video.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        col1, col2 = st.columns([1.5, 1])
        video_display = col1.empty()
        
        results_list = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_idx += 1
            # Process every 10th frame to keep the app responsive
            if frame_idx % 10 != 0: continue
            
            timestamp = frame_idx / fps
            results = yolo(frame, classes=[0], verbose=False)
            boxes = [list(map(int, box.xyxy[0])) for box in results[0].boxes]
            
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                face_crop = pil_img.crop((x1, y1, x2, y2))
                age = float(age_model.predict(np.expand_dims(np.array(face_crop.resize((224, 224)), dtype=np.float32) / 255.0, axis=0), verbose=0)[0][0])
                emotion = max(emotion_pipe(face_crop), key=lambda x: x['score'])['label']
                gender = max(gender_pipe(face_crop), key=lambda x: x['score'])['label']
                
                results_list.append({
                    'Frame': frame_idx, 
                    'Time (s)': f"{timestamp:.2f}", 
                    'ID': i+1, 
                    'Age': int(age), 
                    'Emotion': emotion.capitalize(), 
                    'Gender': gender.capitalize()
                })
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 3)

            video_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            col2.markdown("### Analysis Report")
            col2.table(pd.DataFrame(results_list).tail(10)) # Show last 10 detections
