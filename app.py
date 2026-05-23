import streamlit as st
import os
import gdown
from ultralytics import YOLO
from transformers import pipeline
from tensorflow import keras
import shutil 

# Define BASE_MODEL_DIR relative to the app's current directory
BASE_MODEL_DIR = os.path.join(os.getcwd(), "all_models")

# --- ACCESS SECRETS ---
# This looks for the key 'drive_folder_id' in your Streamlit Cloud Secrets
try:
    SECRET_FOLDER_ID = st.secrets["drive_folder_id"]
except KeyError:
    st.error("Debugger: 'drive_folder_id' not found in Streamlit secrets!")
    st.stop()



@st.cache_resource
def setup_environment(drive_folder_id):
    # If the folder exists but you want to ensure it's fresh/complete:
    if os.path.exists(BASE_MODEL_DIR):
        # OPTIONAL: Delete it to force a fresh download if you added new files
        # shutil.rmtree(BASE_MODEL_DIR) 
        pass 
    
    if not os.path.exists(BASE_MODEL_DIR):
        with st.spinner("Downloading models from Drive..."):
            gdown.download_folder(id=drive_folder_id, output=BASE_MODEL_DIR, quiet=False)
    
    # Validation check
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

    # Load Models
    try:
        yolo = YOLO(required_files["YOLO"])
        emotion_pipe = pipeline("image-classification", model=os.path.join(BASE_MODEL_DIR, "emotion"))
        gender_pipe = pipeline("image-classification", model=os.path.join(BASE_MODEL_DIR, "gender"))
        age_model = keras.models.load_model(required_files["Age"], compile=False)
        return yolo, emotion_pipe, gender_pipe, age_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# --- STREAMLIT UI ---
st.title("Face Analysis App")

if st.button("Initialize & Debug Models"):
    models = setup_environment(SECRET_FOLDER_ID)
    if models:
        st.success("System Ready: All models loaded!")
        st.session_state['models'] = models
