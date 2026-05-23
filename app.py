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

st.set_page_config(layout="wide", page_title="Video Analysis Browser")

# --- INITIALIZATION ---
BASE_MODEL_DIR = os.path.join(os.getcwd(), "all_models")
SECRET_FOLDER_ID = st.secrets["drive_folder_id"]

# Helper: Box Cleanup
def get_clean_boxes(boxes, iou_threshold=0.3):
    if not boxes: return []
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]), reverse=False)
    keep = []
    while boxes:
        current = boxes.pop(0)
        keep.append(current)
        next_boxes = []
        for box in boxes:
            x1, y1 = max(current[0], box[0]), max(current[1], box[1])
            x2, y2 = min(current[2], box[2]), min(current[3], box[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (current[2]-current[0]) * (current[3]-current[1])
            area2 = (box[2]-box[0]) * (box[3]-box[1])
            union = area1 + area2 - inter
            iou = inter / union if union > 0 else 0
            if iou < iou_threshold: next_boxes.append(box)
        boxes = next_boxes
    return keep

@st.cache_resource
def setup_environment(drive_folder_id):
    if not os.path.exists(BASE_MODEL_DIR):
        gdown.download_folder(id=drive_folder_id, output=BASE_MODEL_DIR, quiet=True)
    yolo = YOLO(os.path.join(BASE_MODEL_DIR, "yolo/yolov8n.pt"))
    emotion_pipe = pipeline("image-classification", model=os.path.join(BASE_MODEL_DIR, "emotion"))
    gender_pipe = pipeline("image-classification", model=os.path.join(BASE_MODEL_DIR, "gender"))
    age_model = keras.models.load_model(os.path.join(BASE_MODEL_DIR, "age/best_model.h5"), compile=False)
    return yolo, emotion_pipe, gender_pipe, age_model

# --- MAIN APP ---
st.title("🎥 5-Frame Video Analysis")
models = setup_environment(SECRET_FOLDER_ID)

if models:
    yolo, emotion_pipe, gender_pipe, age_model = models
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

    if uploaded_video:
        with open("temp_vid.mp4", "wb") as f: f.write(uploaded_video.read())
        cap = cv2.VideoCapture("temp_vid.mp4")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate indices for exactly 5 frames
        frame_indices = np.linspace(0, total_frames - 1, 5, dtype=int)

        if 'processed_frames' not in st.session_state:
            with st.spinner("Analyzing 5 key frames..."):
                frames_data = {}
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret: continue
                    
                    results = yolo(frame, classes=[0], verbose=False)
                    coords = get_clean_boxes([list(map(int, b.xyxy[0])) for b in results[0].boxes])
                    
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frame_results = []
                    
                    for i, (x1, y1, x2, y2) in enumerate(coords):
                        face_id = i + 1
                        crop = pil_img.crop((x1, y1, x2, y2))
                        age = int(age_model.predict(np.expand_dims(np.array(crop.resize((224,224)), dtype=np.float32)/255.0, axis=0), verbose=0)[0][0])
                        emo = max(emotion_pipe(crop), key=lambda x: x['score'])['label']
                        gen = max(gender_pipe(crop), key=lambda x: x['score'])['label']
                        
                        frame_results.append({'ID': face_id, 'Age': age, 'Emotion': emo.capitalize(), 'Gender': gen.capitalize()})
                        
                        # Draw Rectangle and ID Tag
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 3)
                        cv2.rectangle(frame, (x1, y1-30), (x1+60, y1), (255, 165, 0), -1)
                        cv2.putText(frame, f"ID:{face_id}", (x1+5, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    frames_data[f"Frame {idx} (Time: {idx/fps:.1f}s)"] = (frame, frame_results)
                
                st.session_state['processed_frames'] = frames_data
                cap.release()

        # --- VIEWING INTERFACE ---
        selection = st.selectbox("Select a frame to inspect:", list(st.session_state['processed_frames'].keys()))
        frame_img, frame_data = st.session_state['processed_frames'][selection]

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col2:
            st.markdown("### Analysis Report")
            if frame_data:
                st.table(pd.DataFrame(frame_data))
            else:
                st.info("No faces detected in this frame.")
