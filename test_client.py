import flwr as fl
import tensorflow as tf
import numpy as np
import sys 
import os
import zipfile 
import pandas as pd
from sklearn.model_selection import train_test_split 

# --- CONFIGURATION CONSTANTS ---
NUM_CLASSES = 5
IMAGE_SHAPE = (224, 224) 
BATCH_SIZE = 32

# ----------------------------------------------------------------------
# 1. ACTUAL DATA LOADING FUNCTION (Handles Unzipping and Custom Load)
# ----------------------------------------------------------------------

def unzip_data(client_id):
    """
    Unzips the hospital data file into a local directory if not already unzipped.
    """
    ZIP_FILE_NAME = f"hospital_{client_id}_data.zip"
    ZIP_PATH = os.path.join(os.getcwd(), "data_zips", ZIP_FILE_NAME) 
    EXTRACT_DIR = os.path.join(os.getcwd(), "data_unzipped", f"hospital_{client_id}_data")
    
    print(f"[Client {client_id}] Checking data: {EXTRACT_DIR}")

    if os.path.isdir(EXTRACT_DIR) and any(os.listdir(EXTRACT_DIR)):
        print(f"[Client {client_id}] Data already unzipped. Skipping.")
        return EXTRACT_DIR

    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(f"Zip file not found: {ZIP_PATH}.")

    print(f"[Client {client_id}] Unzipping {ZIP_FILE_NAME}...")
    os.makedirs(os.path.dirname(EXTRACT_DIR), exist_ok=True)
    
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print(f"[Client {client_id}] Successfully extracted to {EXTRACT_DIR}")
        return EXTRACT_DIR
    except Exception as e:
        raise Exception(f"Failed to unzip {ZIP_FILE_NAME}: {e}")


def load_data(client_id):
    """
    Loads data by reading the Excel/CSV file for labels and matching them to image files.
    """
    HOSPITAL_DIR = unzip_data(client_id)
    
    data_root = os.path.join(HOSPITAL_DIR, f"hospital_{client_id}_data")
    if not os.path.isdir(data_root):
        data_root = HOSPITAL_DIR 

    # --- Step 1: Search for the Label File ---
    LABEL_BASE_NAME = "labels"
    LABEL_FILE = None
    
    for ext in ['.xlsx', '.csv', '']:
        path = os.path.join(data_root, LABEL_BASE_NAME + ext)
        if os.path.exists(path):
            LABEL_FILE = path
            break
            
    if not LABEL_FILE:
        raise FileNotFoundError(f"Label file not found. Searched in {data_root}")
        
    IMAGE_DIR = os.path.join(data_root, "images")

    # --- Step 2: Read Labels from Excel/CSV ---
    print(f"[Client {client_id}] Reading labels from {LABEL_FILE}")
    try:
        if LABEL_FILE.endswith('.xlsx'):
            labels_df = pd.read_excel(LABEL_FILE, engine='openpyxl')
        elif LABEL_FILE.endswith('.csv'):
            labels_df = pd.read_csv(LABEL_FILE)
        else:
            labels_df = pd.read_excel(LABEL_FILE, engine='openpyxl')
            
        labels_df.columns = ['image_name', 'level']
    except Exception as e:
        raise Exception(f"Failed to read label file: {LABEL_FILE}. Error: {e}")

    # --- Step 3: Load and Preprocess Images ---
    X_data = []
    y_labels = []
    
    print(f"[Client {client_id}] Starting image loading from {IMAGE_DIR}...")
    
    POSSIBLE_EXTENSIONS = ['.jpg', '.jpeg', '.png'] 

    for index, row in labels_df.iterrows():
        base_name = str(row['image_name']).replace('.jpg', '').replace('.jpeg', '').strip()
        img_level = int(row['level']) 
        image_loaded = False
        
        for ext in POSSIBLE_EXTENSIONS:
            img_name = base_name + ext
            img_path = os.path.join(IMAGE_DIR, img_name)

            if os.path.exists(img_path):
                try:
                    img = tf.keras.utils.load_img(img_path, target_size=IMAGE_SHAPE)
                    img_array = tf.keras.utils.img_to_array(img)
                    
                    X_data.append(img_array)
                    y_labels.append(img_level)
                    image_loaded = True
                    break 
                except Exception as e:
                    print(f"Skipping image {img_name} due to loading error: {e}")
                    image_loaded = True 
                    break
        
    if not X_data:
        raise Exception("No valid images were loaded. Check file paths and extensions.")
    
    X_data = np.array(X_data, dtype="float32")
    y_labels_int = np.array(y_labels, dtype="int")

    # --- Step 4: Split Data and Final Preprocessing ---
    X_train, X_test, y_train_int, y_test_int = train_test_split(
        X_data, y_labels_int, test_size=0.2, random_state=42, stratify=y_labels_int
    )

    X_train /= 255.0
    X_test /= 255.0
    
    y_train = tf.keras.utils.to_categorical(y_train_int, num_classes=NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test_int, num_classes=NUM_CLASSES)
    
    print(f"  Loaded Training Samples: {len(X_train)}")
    print(f"  Loaded Testing Samples: {len(X_test)}")
    
    return X_train, y_train, X_test, y_test

# ----------------------------------------------------------------------
# 2. MOBILELNETV2 MODEL DEFINITION (MUST match server)
# ----------------------------------------------------------------------
try:
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SHAPE + (3,), 
        include_top=False,
        weights=None  
    )
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax") 
    ])
    
    model.compile(optimizer="adam", 
                  loss="categorical_crossentropy", 
                  metrics=["accuracy"])
    print("MobileNetV2 client model defined and compiled.")
    
except Exception as e:
    print(f"Error initializing MobileNetV2 model: {e}")
    sys.exit(1)

# ----------------------------------------------------------------------
# 3. REAL FLOWER CLIENT IMPLEMENTATION (FLOWER LOGIC)
# ----------------------------------------------------------------------

class RealHospitalClient(fl.client.NumPyClient):
    
    def __init__(self, model, x_train, y_train, x_test, y_test, cid):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.cid = cid 
    
    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] Received global model. Starting local training...")
        self.set_parameters(parameters)
        
        history = self.model.fit(
            self.x_train, self.y_train, 
            epochs=1, # <-- Back to 1 epoch for speed
            batch_size=BATCH_SIZE, 
            verbose=0
        )
        
        updated_parameters = self.model.get_weights()
        num_examples = len(self.x_train)
        loss = history.history["loss"][-1]
        accuracy = history.history["accuracy"][-1]

        print(f"[Client {self.cid}] Training complete. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return updated_parameters, num_examples, {
             "loss": loss,
             "accuracy": accuracy
        }

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] Received model for evaluation. Starting local evaluation...")
        self.set_parameters(parameters)
        
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        metrics_dict = {"accuracy": accuracy, "loss": loss}
        
        print(f"[Client {self.cid}] Evaluation complete. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return loss, len(self.x_test), metrics_dict

# ----------------------------------------------------------------------
# 4. CLIENT EXECUTION (Handles CID 1 and 2)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # --- Step 1: Get Client ID from command-line ---
    if len(sys.argv) >= 2:
        CLIENT_ID = int(sys.argv[1])
    else:
        CLIENT_ID = 1  # default to Hospital 1

    # --- Step 2: Get Server IP from command-line ---
    if len(sys.argv) >= 3:
        SERVER_IP = sys.argv[2]
    else:
        SERVER_IP = "localhost"

    # --- Step 3: Restrict to Hospitals 1 & 2 only ---
    if CLIENT_ID not in [1, 2]:
        print(f"[ERROR] Invalid Client ID {CLIENT_ID}. This setup supports only Hospital 1 and 2.")
        sys.exit(1)

    # --- Step 4: Load local data for that hospital ---
    try:
        X_train, y_train, X_test, y_test = load_data(CLIENT_ID)
    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR during data loading/unzipping: {e}")
        sys.exit(1)

    # --- Step 5: Create client instance ---
    client_instance = RealHospitalClient(model, X_train, y_train, X_test, y_test, CLIENT_ID)

    print(f"Starting MobileNetV2 Client ID {CLIENT_ID}...")
    print(f"Connecting to Server at {SERVER_IP}:8080 ...")

    # --- Step 6: Connect to Federated Server ---
    try:
        fl.client.start_client(
            server_address=f"{SERVER_IP}:8080",
            client=client_instance.to_client()
        )
    except Exception as e:
        if "Connection refused" in str(e):
            print(f"[Client {CLIENT_ID}] ERROR: Server not available at {SERVER_IP}:8080. Is it running?")
        else:
            print(f"[Client {CLIENT_ID}] Connection error: {e}")