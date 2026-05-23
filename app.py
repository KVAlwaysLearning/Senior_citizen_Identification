import streamlit as st
import os
import gdown
from ultralytics import YOLO
from transformers import pipeline
from tensorflow import keras

# --- CONFIGURATION ---
BASE_MODEL_DIR = "/content/all_models" # Define the missing variable

# --- ACCESS SECRETS ---
try:
    SECRET_FOLDER_ID = st.secrets["1eGlOyj6Fl1gIT9mXp6Baor_wravdP2wV"]
except KeyError:
    st.error("Debugger: 'drive_folder_id' not found in Streamlit secrets!")
    st.stop()

@st.cache_resource
def setup_environment(drive_folder_id):
    """Downloads models from Drive and initializes them."""
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
