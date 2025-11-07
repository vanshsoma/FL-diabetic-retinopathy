import flwr as fl
import tensorflow as tf
import numpy as np

# 1. Define the same model as the server
# This structure MUST match the server's model
try:
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
    print("Fake client model initialized.")
except Exception as e:
    print(f"Error initializing model: {e}")
    print("This might be a TensorFlow/Keras issue.")
    exit()

# 2. Define the Flower client
class FakeHospitalClient(fl.client.NumPyClient):
    
    def get_parameters(self, config):
        """Get the model's current weights."""
        print("[Client] Sending initial model parameters.")
        return model.get_weights()

    def fit(self, parameters, config):
        """
        'Train' the model. We'll load the server's weights
        and return fake metrics for the 'fit' (client_metrics) log.
        """
        print("[Client] Received global model. Faking training...")
        
        # Load the server's weights
        model.set_weights(parameters)
        
        # Fake training results
        loss = np.random.rand()
        accuracy = 0.75 + np.random.rand() * 0.1 # 0.75-0.85
        
        print("[Client] 'Training' complete. Sending weights and metrics.")
        
        # Return new weights, number of examples, and fake metrics
        # These keys MUST match what the server logs in `aggregate_fit`
        return model.get_weights(), 100, {
            "loss": loss,
            "accuracy": accuracy
        }

    def evaluate(self, parameters, config):
        """
        'Evaluate' the model. We'll return fake metrics for the 'evaluate' (global_accuracy) log.
        """
        print("[Client] Received model for evaluation. Faking evaluation...")
        
        # Load the server's weights
        model.set_weights(parameters)
        
        # Fake an evaluation
        loss = 0.5  # This is the "global_loss"
        accuracy = 0.82  # This is the "global_accuracy"
        
        # This key MUST match what the server logs in `aggregate_evaluate`
        metrics_dict = {"accuracy": accuracy}
        
        print(f"[Client] 'Evaluation' complete. Sending metrics: {metrics_dict}")
        return loss, 10, metrics_dict

# 3. Start the client
print("Starting fake hospital client...")
try:
    fl.client.start_client(
        server_address="localhost:8080",
        client=FakeHospitalClient().to_client() # Use .to_client() to fix deprecation warning
        # Removed client_cid="hospital_1" because your flwr version doesn't support it
    )
    print("Fake client disconnected.")
except Exception as e:
    print(f"Client connection error: {e}")
    print("Is the server (start_demo.bat) running in another terminal?")