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

# Define BASE_MODEL_DIR relative to the app's current directory
BASE_MODEL_DIR = os.path.join(os.getcwd(), "all_models")

# --- ACCESS SECRETS ---
try:
    SECRET_FOLDER_ID = st.secrets["drive_folder_id"]
except KeyError:
    st.error("Debugger: 'drive_folder_id' not found in Streamlit secrets!")
    st.stop()

# Helper function for Box Cleanup
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
        with st.spinner("Downloading models from Drive..."):
            gdown.download_folder(id=drive_folder_id, output=BASE_MODEL_DIR, quiet=False)
    
    required_files = {
        "YOLO": os.path.join(BASE_MODEL_DIR, "yolo/yolov8n.pt"),
        "Age": os.path.join(BASE_MODEL_DIR, "age/best_model.h5"),
        "Emotion": os.path.join(BASE_MODEL_DIR, "emotion/pytorch_model.bin"),
        "Gender": os.path.join(BASE_MODEL_DIR, "gender/model.safetensors")
    }
    
    missing = [name for name, path in required_files.items() if not os.path.exists(path)]
    if missing:
        st.error(f"Debugger: Missing files: {', '.join(missing)}")
        return None

    try:
        yolo = YOLO(required_files["YOLO"])
        emotion_pipe = pipeline("image-classification", model=os.path.join(BASE_MODEL_DIR, "emotion"))
        gender_pipe = pipeline("image-classification", model=os.path.join(BASE_MODEL_DIR, "gender"))
        age_model = keras.models.load_model(required_files["Age"], compile=False)
        return yolo, emotion_pipe, gender_pipe, age_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# --- UI START ---
st.title("Face Analysis App")

if st.button("Initialize & Debug Models"):
    models = setup_environment(SECRET_FOLDER_ID)
    if models:
        st.success("System Ready: All models loaded!")
        st.session_state['models'] = models

if 'models' in st.session_state:
    yolo, emotion_pipe, gender_pipe, age_model = st.session_state['models']
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        
        results = yolo(frame, classes=[0], verbose=False)
        boxes = [list(map(int, box.xyxy[0])) for box in results[0].boxes]
        coords = get_clean_boxes(boxes, iou_threshold=0.3)
        
        results_list = []
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        for i, (x1, y1, x2, y2) in enumerate(coords):
            face_crop = pil_img.crop((x1, y1, x2, y2))
            
            resized_age = face_crop.resize((224, 224))
            age_array = np.expand_dims(np.array(resized_age, dtype=np.float32) / 255.0, axis=0)
            age = float(age_model.predict(age_array, verbose=0)[0][0])
            
            emotion = max(emotion_pipe(face_crop), key=lambda x: x['score'])['label']
            gender = max(gender_pipe(face_crop), key=lambda x: x['score'])['label']
            
            results_list.append({'ID': i+1, 'Age': int(age), 'Emotion': emotion, 'Gender': gender})
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"{gender}, {int(age)}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Analyzed Image", use_container_width=True)
        with col2:
            st.subheader("Analysis Results")
            if results_list:
                st.dataframe(pd.DataFrame(results_list), use_container_width=True, hide_index=True)
            else:
                st.warning("No faces detected.")
