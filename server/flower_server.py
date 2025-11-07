import flwr as fl
import tensorflow as tf
import os
import json
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, Metrics
from typing import Dict, Optional, Tuple, List, Union

# Make sure models and logs directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

MODEL_PATH = "models/global_model.h5"
LOGS_PATH = "logs/training_metrics.json"

# --- CONFIGURATION ---
NUM_ROUNDS = 10         
CLIENTS_REQUIRED = 2   

# Maps the client's CID (which is a string) to the name the dashboard expects
CLIENT_ID_TO_NAME = {
    "1": "City General Hospital",
    "2": "Suburban Medical Center",
    "3": "Rural District Hospital",       # For future use
    "4": "University Medical Research"  # For future use
}

# --- MODEL DEFINITION ---
# This structure MUST match the client's model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None  
)
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(5, activation="softmax") # 5 classes
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# --- END OF MODEL DEFINITION ---

# Save the initial model
model.save(MODEL_PATH)

# Initialize logs file
if not os.path.exists(LOGS_PATH):
    with open(LOGS_PATH, 'w') as f:
        json.dump({"rounds": [], "connected_clients": 0, "status": "Waiting"}, f)

def update_log(data: Dict):
    """Helper function to read, update, and write to the JSON log file."""
    try:
        with open(LOGS_PATH, 'r+') as f:
            log_data = json.load(f)
            log_data.update(data)
            f.seek(0)
            f.truncate()
            json.dump(log_data, f, indent=4)
    except (IOError, json.JSONDecodeError):
        with open(LOGS_PATH, 'w') as f:
            json.dump(data, f, indent=4)

def get_initial_parameters(config):
    """Load the initial model parameters."""
    if os.path.exists(MODEL_PATH):
        model.load_weights(MODEL_PATH)
        print(f"Loaded initial weights from {MODEL_PATH}")
    else:
        print("No initial model found, using randomly initialized weights.")
        
    weights = model.get_weights()
    parameters = ndarrays_to_parameters(weights)
    return parameters

class SaveModelStrategy(FedAvg):
    """
    Custom strategy to save the global model and log metrics every round.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This will store the metrics from the FIT stage
        self.client_fit_metrics = {} 

    def on_server_round_start(self, server_round: int) -> Optional[Dict[str, fl.common.Scalar]]:
        """Called at the start of each round."""
        update_log({"status": f"Round {server_round}/{NUM_ROUNDS}", "current_round": server_round, "connected_clients": CLIENTS_REQUIRED})
        return super().on_server_round_start(server_round)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # --- THIS IS THE FIX ---
        # Log client FIT metrics using the mapped name
        self.client_fit_metrics[server_round] = {}
        for client_proxy, fit_res in results:
            client_cid = client_proxy.cid 
            client_name = CLIENT_ID_TO_NAME.get(client_cid, f"Client {client_cid}")
            
            metrics = fit_res.metrics
            self.client_fit_metrics[server_round][client_name] = metrics 
            print(f"[Round {server_round}] Client {client_name} ({client_cid}) fit metrics: {metrics}")
        # --- END OF FIX ---

        if aggregated_parameters is not None:
            aggregated_weights = parameters_to_ndarrays(aggregated_parameters)
            model.set_weights(aggregated_weights)
            model.save(MODEL_PATH)
            print(f"[Round {server_round}] Global model saved to {MODEL_PATH}")

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        
        if not results:
            print("No evaluation results received.")
            return None, {}

        total_examples = 0
        weighted_accuracy_sum = 0
        total_loss = 0
        
        # --- THIS IS THE FIX ---
        # Create a dictionary for this round's EVALUATION metrics
        client_eval_metrics = {}
        # --- END OF FIX ---

        for client_proxy, eval_res in results:
            num_examples = eval_res.num_examples
            loss = eval_res.loss
            metrics = eval_res.metrics
            
            accuracy = metrics.get("accuracy", 0.0) 
            
            total_examples += num_examples
            weighted_accuracy_sum += num_examples * accuracy
            total_loss += num_examples * loss
            
            # --- THIS IS THE FIX ---
            # Map the CID to the name and save the EVAL metrics
            client_cid = client_proxy.cid
            client_name = CLIENT_ID_TO_NAME.get(client_cid, f"Client {client_cid}")
            client_eval_metrics[client_name] = {"loss": loss, "accuracy": accuracy}
            print(f"[Round {server_round}] Client {client_name} ({client_cid}) eval metrics: {metrics}")
            # --- END OF FIX ---

        loss_aggregated = total_loss / total_examples if total_examples > 0 else 0
        global_accuracy = weighted_accuracy_sum / total_examples if total_examples > 0 else 0
        
        metrics_aggregated = {"accuracy": global_accuracy}
        
        print(f"[Round {server_round}] Global accuracy: {global_accuracy:.4f}")

        # --- LOG TO JSON ---
        try:
            with open(LOGS_PATH, 'r') as f:
                log_data = json.load(f)
            
            # The client_metrics dict now has the correct names
            log_data["rounds"].append({
                "round": server_round,
                "global_accuracy": global_accuracy,
                "global_loss": loss_aggregated,
                # --- THIS IS THE FIX ---
                # Save the EVAL metrics, which the dashboard is looking for
                "client_metrics": client_eval_metrics 
            })
            
            with open(LOGS_PATH, 'w') as f:
                json.dump(log_data, f, indent=4)
                
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error writing to log file: {e}")

        return loss_aggregated, metrics_aggregated

# Define the strategy
strategy = SaveModelStrategy(
    initial_parameters=get_initial_parameters(None),
    min_fit_clients=CLIENTS_REQUIRED,
    min_evaluate_clients=CLIENTS_REQUIRED,
    min_available_clients=CLIENTS_REQUIRED,
)

# Start the server
print(f"Starting Flower server on 0.0.0.0:8080, waiting for {CLIENTS_REQUIRED} clients...")
update_log({"status": f"Waiting for {CLIENTS_REQUIRED} clients..."})

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)

print("Training complete. Server shutting down.")
update_log({"status": "Training complete."})