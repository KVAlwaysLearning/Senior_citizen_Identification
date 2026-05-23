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

# ==============================================================================
# IMAGE ANALYSIS PIPELINE (Add this to your app.py)
# ==============================================================================

if 'models' in st.session_state:
    # Retrieve models from session state
    yolo, emotion_pipe, gender_pipe, age_model = st.session_state['models']
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        
        # 1. YOLO Detection
        results = yolo(frame, classes=[0], verbose=False)
        boxes = [list(map(int, box.xyxy[0])) for box in results[0].boxes]
        coords = get_clean_boxes(boxes, iou_threshold=0.3)
        
        # 2. Process Detections
        results_list = []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        for i, (x1, y1, x2, y2) in enumerate(coords):
            face_crop = pil_img.crop((x1, y1, x2, y2))
            
            # Age Estimation
            resized_age = face_crop.resize((224, 224))
            age_array = np.expand_dims(np.array(resized_age, dtype=np.float32) / 255.0, axis=0)
            age = float(age_model.predict(age_array, verbose=0)[0][0])
            
            # Emotion
            emo_results = emotion_pipe(face_crop)
            emotion = max(emo_results, key=lambda x: x['score'])['label']
            
            # Gender
            gen_results = gender_pipe(face_crop)
            gender = max(gen_results, key=lambda x: x['score'])['label']
            
            results_list.append({'ID': i+1, 'Age': int(age), 'Emotion': emotion, 'Gender': gender})
            
            # Draw overlay on image
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"{gender}, {int(age)}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 3. Layout: Image on Left, Results Table on Right
        col1, col2 = st.columns([2, 1]) 

        with col1:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Analyzed Image", use_container_width=True)

        with col2:
            st.subheader("Analysis Results")
            if results_list:
                df = pd.DataFrame(results_list)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning("No faces detected in the image.")
