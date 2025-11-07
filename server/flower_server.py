import flwr as fl
import tensorflow as tf
import numpy as np
from flwr.server.strategy import FedAvg
from typing import Callable, Dict, Optional, Tuple, List
from flwr.common import Parameters, Scalar, FitRes, Metrics
import json # <-- NEW IMPORT

# --- CONFIGURATION CONSTANTS (Keep consistent with client) ---
NUM_CLASSES = 5
IMAGE_SHAPE = (224, 224, 3) 
LOG_FILE = "logs/training_metrics.json"

# ----------------------------------------------------------------------
# 1. MOBILELNETV2 MODEL DEFINITION (MUST match client)
# ----------------------------------------------------------------------

def get_model():
    """Defines and compiles the MobileNetV2 model architecture."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SHAPE,
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
    return model

# ----------------------------------------------------------------------
# 2. METRIC AGGREGATION & LOGGING (THE FIX FOR THE DASHBOARD)
# ----------------------------------------------------------------------

# Helper function to write metrics to the JSON log file
def save_metrics_to_json(round_number: int, metrics: Dict[str, Scalar]):
    # Load existing data or start new
    try:
        with open(LOG_FILE, "r") as f:
            log_data = json.load(f)
    except FileNotFoundError:
        log_data = []

    # Append new round data
    round_data = {"round": round_number, **metrics}
    log_data.append(round_data)

    # Write back to file
    with open(LOG_FILE, "w") as f:
        json.dump(log_data, f, indent=4)

def fit_metrics_aggregation_fn(
    metrics_list: List[Tuple[int, Metrics]]
) -> Metrics:
    """Aggregates fit metrics and logs them."""
    # Calculate weighted average for loss and accuracy
    total_examples = sum([num_examples for num_examples, _ in metrics_list])
    
    avg_loss = sum(
        [m["loss"] * num_examples for num_examples, m in metrics_list]
    ) / total_examples
    
    avg_accuracy = sum(
        [m["accuracy"] * num_examples for num_examples, m in metrics_list]
    ) / total_examples
    
    print(f"Server (Fit):   Loss {avg_loss:.4f}, Accuracy {avg_accuracy:.4f}")
    
    return {"loss": avg_loss, "accuracy": avg_accuracy}


def evaluate_metrics_aggregation_fn(
    metrics_list: List[Tuple[int, Metrics]]
) -> Metrics:
    """Aggegates evaluation metrics and logs them to JSON."""
    # Calculate weighted average for loss and accuracy
    total_examples = sum([num_examples for num_examples, _ in metrics_list])
    
    avg_loss = sum(
        [m["loss"] * num_examples for num_examples, m in metrics_list]
    ) / total_examples
    
    avg_accuracy = sum(
        [m["accuracy"] * num_examples for num_examples, m in metrics_list]
    ) / total_examples
    
    print(f"Server (Eval):  Loss {avg_loss:.4f}, Accuracy {avg_accuracy:.4f}")
    
    # **SAVE TO JSON FILE FOR DASHBOARD**
    # We use the round number from the server config (this is a bit of a hack)
    # A cleaner way would be to get the current round, but this works
    try:
        with open(LOG_FILE, "r") as f:
            log_data = json.load(f)
        current_round = len(log_data) + 1
    except FileNotFoundError:
        current_round = 1
        
    save_metrics_to_json(current_round, {
        "global_loss": avg_loss,
        "global_accuracy": avg_accuracy
    })
    
    return {"loss": avg_loss, "accuracy": avg_accuracy}


# ----------------------------------------------------------------------
# 3. SERVER STARTUP LOGIC
# ----------------------------------------------------------------------

def start_flower_server():
    initial_model = get_model()
    initial_parameters = fl.common.ndarrays_to_parameters(initial_model.get_weights())

    # Clear the log file at the start of a new session
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        print("Removed old log file.")

    # Define the strategy
    strategy = FedAvg(
        fraction_fit=1.0, 
        fraction_evaluate=1.0, 
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2, 
        initial_parameters=initial_parameters,
        
        # ** ADD THE METRIC HANDLERS **
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    # Start the server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=6), # 6 rounds
        strategy=strategy,
    )

if __name__ == "__main__":
    start_flower_server()