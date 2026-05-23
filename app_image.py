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

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Face Analysis Pro")

# --- GLOBAL SETTINGS ---
BASE_MODEL_DIR = os.path.join(os.getcwd(), "all_models")
SECRET_FOLDER_ID = st.secrets["drive_folder_id"]

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
st.title("👤 Face Analysis Dashboard")
st.markdown("Upload an image to analyze age, gender, and emotion automatically.")

# Initialize models silently
with st.spinner("Loading AI models..."):
    models = setup_environment(SECRET_FOLDER_ID)

if models:
    yolo, emotion_pipe, gender_pipe, age_model = models
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        
        # Processing
        results = yolo(frame, classes=[0], verbose=False)
        boxes = [list(map(int, box.xyxy[0])) for box in results[0].boxes]
        coords = get_clean_boxes(boxes, iou_threshold=0.3)
        
        results_list = []
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Updated loop with ID labeling
        for i, (x1, y1, x2, y2) in enumerate(coords):
            face_crop = pil_img.crop((x1, y1, x2, y2))
            
            # Predictions
            age = float(age_model.predict(np.expand_dims(np.array(face_crop.resize((224, 224)), dtype=np.float32) / 255.0, axis=0), verbose=0)[0][0])
            emotion = max(emotion_pipe(face_crop), key=lambda x: x['score'])['label']
            gender = max(gender_pipe(face_crop), key=lambda x: x['score'])['label']
            
            face_id = i + 1
            results_list.append({'ID': face_id, 'Age': int(age), 'Emotion': emotion.capitalize(), 'Gender': gender.capitalize()})
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 3)
            
            # Draw ID Label Background (The visual link)
            label = f"ID: {face_id}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), (255, 165, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

       # 3. Layout: Image on Left, Results Table on Right
        col1, col2 = st.columns([1.5, 1])
        with col1:
            # Displays the image with bounding boxes and ID tags drawn on it
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Analyzed Image", use_container_width=True)
            
        with col2:
            st.markdown("### Analysis Report")
            if results_list:
                df = pd.DataFrame(results_list)
                # This makes ID the primary column and displays all data clearly
                st.table(df) 
            else:
                st.warning("No faces detected in the image.")
