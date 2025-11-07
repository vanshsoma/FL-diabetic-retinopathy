import streamlit as st
import json
import os
import pandas as pd
from streamlit_autorefresh import st_autorefresh

# --- Configuration ---
LOGS_PATH = "logs/training_metrics.json"   # <-- This path is correct
MODEL_PATH = "models/global_model.h5"     # <-- This path is correct
REFRESH_INTERVAL = 2000  # 2 seconds in milliseconds
CLIENTS_REQUIRED = 4     # Make sure this matches your flower_server.py (set to 1 for test)

# Hospital names (must match the client_cid)
# This is now a "lookup" map
HOSPITAL_NAMES = {
    "hospital_1": "City General Hospital",
    "hospital_2": "Suburban Medical Center",
    "hospital_3": "Rural District Hospital",
    "hospital_4": "University Medical Research"
}

# --- Data Loading Functions ---

# --- THIS IS THE FIX ---
# We comment out the cache decorator. This forces Streamlit
# to re-read the file every 2 seconds, solving the race condition.
# @st.cache_data(show_spinner=False)
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
def load_model():
    """Load the model file for downloading."""
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                return f.read()
        except (IOError, PermissionError) as e:
            print(f"Warning: Could not read model file (likely locked): {e}")
            return None
    return None

# --- Main Dashboard ---
st.set_page_config(layout="wide", page_title="Federated DR Screening")
st_autorefresh(interval=REFRESH_INTERVAL, key="data_refresher")

# Load data
data = load_data()
model_data = load_model()
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

# --- Main Content Columns ---
col_hospitals, col_chart = st.columns([1, 1])

# --- Column 1: Hospital Status ---
with col_hospitals:
    st.header("Hospital Status")
    
    # Get client metrics from the *last* completed round
    client_metrics_dict = last_round.get("client_metrics", {}) if last_round else {}
    
    # Dynamically find which clients are active from the log
    active_client_ids = list(client_metrics_dict.keys())
    
    # Determine connected clients for the "Info" box
    if "Waiting" in status:
        try:
            connected_clients = CLIENTS_REQUIRED - int(status.split(" ")[1])
        except: connected_clients = 0
    elif "Round" in status or "complete" in status:
        connected_clients = CLIENTS_REQUIRED
    else:
        connected_clients = 0

    cols = st.columns(2)
    
    # Create a unified list of all clients to display
    all_client_keys = list(HOSPITAL_NAMES.keys())
    # Add any unknown clients (like our test client)
    for cid in active_client_ids:
        if cid not in all_client_keys:
            all_client_keys.append(cid)

    if not all_client_keys:
        st.info("Waiting for clients to connect...")

    for i, client_id in enumerate(all_client_keys):
        if i >= 4: break # Limit to 4 cards for this demo
        
        card_col = cols[i % 2]
        
        # Get the friendly name, or use the ID as a fallback
        name = HOSPITAL_NAMES.get(client_id, f"Client ({client_id[:6]}...)")
        metrics = client_metrics_dict.get(client_id)
        
        with card_col:
            if metrics:
                # Client is active and sent data
                st.success(f"**{name}** (Active)")
                c1, c2 = st.columns(2)
                c1.metric("Local Loss", f"{metrics.get('loss', 0):.3f}")
                c2.metric("Local Accuracy", f"{metrics.get('accuracy', 0) * 100:.1f}%")
            else:
                # Client is known but offline
                st.error(f"**{name}** (Offline)")
                c1, c2 = st.columns(2)
                c1.metric("Local Loss", "---")
                c2.metric("Local Accuracy", "---")

# --- Column 2: Chart and Info ---
with col_chart:
    st.header("Global Model Performance")
    
    if not rounds:
        st.info("Waiting for the first round to complete...")
    else:
        df = pd.DataFrame(rounds)
        df = df.rename(columns={"round": "Round", "global_accuracy": "Global Accuracy"})
        df["Global Accuracy"] = df["Global Accuracy"] * 100
        st.line_chart(df.set_index("Round")["Global Accuracy"], height=300)
    
    st.subheader("Info")
    st.markdown(f"""
    - **Connected Hospitals:** {connected_clients} / {CLIENTS_REQUIRED}
    - **Privacy:** 🔒 Only model weights are shared. No patient data ever leaves the hospital.
    """)
    
    if model_data:
        st.download_button("Download Latest Global Model (.h5)", model_data, "global_model.h5")
    else:
        st.download_button("Download Latest Global Model (.h5)", b'', "global_model.h5", disabled=True)

with st.expander("Show Raw JSON Log"):
    st.json(data)