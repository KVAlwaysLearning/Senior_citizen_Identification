import streamlit as st
import os
from ultralytics import YOLO
from transformers import pipeline
import tensorflow as tf
from tensorflow import keras

# Define the base directory where your 'all_models' folder is located
# Adjust this path if your folder is in a different location in your Drive
BASE_MODEL_DIR = "/content/drive/MyDrive/my_project/all_models"

@st.cache_resource
def debug_and_load_models():
    """
    1. Validates the existence of all required files.
    2. Loads models into memory.
    """
    # Define required files and their specific paths
    required_files = {
        "YOLO": os.path.join(BASE_MODEL_DIR, "yolo/yolov8n.pt"),
        "Age": os.path.join(BASE_MODEL_DIR, "age/best_model.h5"),
        "Emotion": os.path.join(BASE_MODEL_DIR, "emotion/pytorch_model.bin"),
        "Gender": os.path.join(BASE_MODEL_DIR, "gender/model.safetensors")
    }

    # Debugger: Check if files exist
    for name, path in required_files.items():
        if not os.path.exists(path):
            st.error(f"DEBUGGER ERROR: {name} model file not found at {path}")
            return None
    
    st.success("Debugger: All model files located.")

    # Load Models
    try:
        yolo = YOLO(required_files["YOLO"])
        emotion_pipe = pipeline("image-classification", model=os.path.join(BASE_MODEL_DIR, "emotion"))
        gender_pipe = pipeline("image-classification", model=os.path.join(BASE_MODEL_DIR, "gender"))
        
        # Load custom Keras model
        age_model = keras.models.load_model(required_files["Age"], compile=False)
        
        return yolo, emotion_pipe, gender_pipe, age_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# --- STREAMLIT UI ---
st.title("Face Analysis App")

if st.button("Initialize & Debug Models"):
    models = debug_and_load_models()
    if models:
        st.write("Models successfully initialized and ready for inference!")
