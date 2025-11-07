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
NUM_ROUNDS = 10
CLIENTS_REQUIRED = 1  # Set to 1 for testing

# Initialize a simple model to get its weights
# This structure MUST match the client's model
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None  # We start from scratch
)
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(5, activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

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
        # If file is corrupted or empty, re-initialize
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
        self.client_metrics = {} 

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
        
        # Aggregate weights
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # Log client metrics
        self.client_metrics[server_round] = {}
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid # Get the client's unique ID
            metrics = fit_res.metrics
            self.client_metrics[server_round][client_id] = metrics
            print(f"[Round {server_round}] Client {client_id} metrics: {metrics}")

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
        
        # --- THIS IS THE FIX ---
        # Don't use super().aggregate_evaluate(), it's unreliable for custom keys.
        # Manually calculate the weighted average of "accuracy" from the results.
        
        if not results:
            print("No evaluation results received.")
            return None, {}

        total_examples = 0
        weighted_accuracy_sum = 0
        total_loss = 0

        for client_proxy, eval_res in results:
            num_examples = eval_res.num_examples
            loss = eval_res.loss
            metrics = eval_res.metrics
            
            # The key MUST match what the client sends in evaluate()
            accuracy = metrics.get("accuracy", 0.0) 
            
            total_examples += num_examples
            weighted_accuracy_sum += num_examples * accuracy
            total_loss += num_examples * loss

        # Calculate final metrics
        loss_aggregated = total_loss / total_examples if total_examples > 0 else 0
        global_accuracy = weighted_accuracy_sum / total_examples if total_examples > 0 else 0
        
        metrics_aggregated = {"accuracy": global_accuracy}
        # --- END OF FIX ---
        
        print(f"[Round {server_round}] Global accuracy: {global_accuracy:.4f}")

        try:
            with open(LOGS_PATH, 'r') as f:
                log_data = json.load(f)
            
            log_data["rounds"].append({
                "round": server_round,
                "global_accuracy": global_accuracy, # This will now be correct
                "client_metrics": self.client_metrics.get(server_round, {})
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