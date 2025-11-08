import streamlit as st
import json
import os
import pandas as pd
from streamlit_autorefresh import st_autorefresh

# --- New Imports for Prediction ---
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# --- Configuration ---
LOGS_PATH = "logs/training_metrics.json"   # <-- This path is correct
MODEL_PATH = "models/global_model.h5"     # <-- This path is correct
REFRESH_INTERVAL = 2000  # 2 seconds in milliseconds
CLIENTS_REQUIRED = 2     # <-- Set to 2 as requested

# --- IMPORTANT: Model & Data Configuration ---
# You MUST change these to match your model's training!
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
CLASS_NAMES = ["No Diabetic Retinopathy", "Diabetic Retinopathy Present"]


# --- Data Loading Functions ---

# We don't cache load_data() to force a re-read on every refresh
def load_data():
    """Load the latest metrics from the JSON log file."""
    default_data = {"rounds": [], "connected_clients": 0, "status": "Initializing..."}
    if not os.path.exists(LOGS_PATH):
        return default_data
    
    try:
        with open(LOGS_PATH, 'r') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError, PermissionError) as e:
        print(f"Warning: Could not read log file (likely locked): {e}")
        return default_data

@st.cache_data(show_spinner=False)
def load_model_bytes():
    """Load the model file bytes for downloading."""
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                return f.read()
        except (IOError, PermissionError) as e:
            print(f"Warning: Could not read model file (likely locked): {e}")
            return None
    return None

# --- NEW: Functions for Prediction ---

@st.cache_resource(show_spinner="Loading global model for prediction...")
def load_keras_model():
    """Load the Keras model into memory for actual prediction."""
    if os.path.exists(MODEL_PATH):
        try:
            # Load the model using TensorFlow/Keras
            model = tf.keras.models.load_model(MODEL_PATH)
            return model
        except Exception as e:
            st.error(f"Error loading Keras model: {e}")
            return None
    return None

def preprocess_image(image_bytes):
    """
    Preprocess the uploaded image to match the model's input requirements.
    """
    try:
        # Open the image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB (if it's RGBA, for example)
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # <-- IMPORTANT! Resize to your model's expected input size
        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # <-- IMPORTANT! Normalize pixels (e.g., / 255.0)
        # Change this if your model expects a different normalization
        image_array = image_array / 255.0
        
        # Expand dimensions to create a "batch" of 1
        # Shape changes from (width, height, channels) to (1, width, height, channels)
        image_batch = np.expand_dims(image_array, axis=0)
        
        return image_batch
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def decode_prediction(prediction, class_names):
    """Convert the model's raw output into a human-readable class name."""
    # Example for binary or multi-class classification
    predicted_index = np.argmax(prediction[0])
    confidence = prediction[0][predicted_index] * 100
    
    # <-- IMPORTANT! Adjust this logic if your model output is different
    # (e.g., if it's binary sigmoid output, prediction[0][0] > 0.5)
    
    if predicted_index < len(class_names):
        return class_names[predicted_index], confidence
    else:
        return "Unknown", 0.0

# --- Main Dashboard ---
st.set_page_config(layout="wide", page_title="Federated DR Screening")
st_autorefresh(interval=REFRESH_INTERVAL, key="data_refresher")

# Load data and models
data = load_data()
model_bytes_for_download = load_model_bytes() # For the download button
model_for_prediction = load_keras_model()     # For the prediction feature

status = data.get("status", "Waiting...")
rounds = data.get("rounds", [])
last_round = rounds[-1] if rounds else None

# --- Header ---
col_title, col_metric = st.columns([3, 1])
with col_title:
    st.title("🏥 Federated DR Screening Dashboard")
    st.markdown(f"**Status:** `{status}`")
with col_metric:
    if last_round:
        st.metric(
            label=f"Latest Global Accuracy (Round {last_round['round']})",
            value=f"{last_round.get('global_accuracy', 0.0) * 100:.2f}%"
        )
    else:
        st.metric("Latest Global Accuracy", "---")

st.divider()

# --- Main Content ---

# Calculate connected clients
if "Waiting" in status:
    try:
        connected_clients = CLIENTS_REQUIRED - int(status.split(" ")[1])
    except: connected_clients = 0
elif "Round" in status or "complete" in status:
    connected_clients = CLIENTS_REQUIRED
else:
    connected_clients = 0

# --- Create Columns for Dashboard (SWAPPED) ---
col_info, col_chart = st.columns([1, 2])  # <-- Swapped order and ratio

with col_info:
    st.header("Session Info")
    
    with st.container(border=True):
        st.metric(
            label="Connected Hospitals",
            value=f"{connected_clients} / {CLIENTS_REQUIRED}"
        )
        st.markdown(
            "🔒 **Privacy:** Only model weights are shared. "
            "No patient data ever leaves the hospital."
        )

    # Place download button outside the container but still in the column
    if model_bytes_for_download:
        st.download_button(
            "Download Latest Global Model (.h5)", 
            model_bytes_for_download, 
            "global_model.h5",
            use_container_width=True
        )
    else:
        st.download_button(
            "Download Latest Global Model (.h5)", 
            b'', 
            "global_model.h5", 
            disabled=True,
            use_container_width=True
        )

with col_chart:
    st.header("Global Model Performance")
    if not rounds:
        st.info("Waiting for the first round to complete...")
    else:
        df = pd.DataFrame(rounds)
        df = df.rename(columns={"round": "Round", "global_accuracy": "Global Accuracy"})
        df["Global Accuracy"] = df["Global Accuracy"] * 100
        st.line_chart(df.set_index("Round")["Global Accuracy"], height=350)

# --- Raw Log (Full Width) ---
with st.expander("Show Raw JSON Log"):
    st.json(data)

st.divider()

# --- =============== NEW PREDICTION SECTION =============== ---

st.header("🔬 Predict on a New Scan")

if model_for_prediction is None:
    st.warning("Model is not yet available for prediction. Please wait for the first training round to complete.", icon="⏳")
else:
    # Use columns for a cleaner layout
    col_upload, col_result = st.columns(2)
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload a retinal scan (JPG, PNG)", 
            type=["jpg", "jpeg", "png"]
        )

    if uploaded_file is not None:
        # Read the file bytes
        image_bytes = uploaded_file.getvalue()
        
        # Display the uploaded image
        with col_upload:
            st.image(image_bytes, caption="Uploaded Scan", use_column_width=True)
        
        with col_result:
            with st.spinner("Analyzing scan..."):
                # Preprocess the image
                processed_batch = preprocess_image(image_bytes)
                
                if processed_batch is not None:
                    # Run prediction
                    try:
                        prediction = model_for_prediction.predict(processed_batch)
                        
                        # Decode the prediction
                        class_name, confidence = decode_prediction(prediction, CLASS_NAMES)
                        
                        # Display the result
                        st.subheader("Analysis Result")
                        if "Present" in class_name:
                             st.error(f"**Prediction:** {class_name}", icon="⚠️")
                        else:
                             st.success(f"**Prediction:** {class_name}", icon="✅")
                        
                        st.metric(label="Confidence", value=f"{confidence:.2f}%")
                        
                        # Optional: Show raw prediction output
                        with st.expander("Show Raw Prediction Output"):
                            st.write(prediction)

                    except Exception as e:
                        st.error(f"Error during prediction: {e}")