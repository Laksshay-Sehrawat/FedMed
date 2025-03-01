import flwr as fl
import logging
from flwr.server.strategy import FedAvg
import numpy as np
from typing import List, Tuple, Dict, Optional
from flwr.common import Parameters, Scalar
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrustBasedStrategy(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client_trust_scores = {}
        self.round_metrics = {}
        self.initial_parameters = None  # Add parameter cache
        logger.info("TrustBasedStrategy initialized with:")
        logger.info(f"min_fit_clients={self.min_fit_clients}")
        logger.info(f"min_evaluate_clients={self.min_evaluate_clients}")

    def initialize_parameters(self, client_manager=None):
        """Initialize parameters with EXACT model structure"""
        dummy_model = create_dummy_model((128, 128, 3))
        return fl.common.ndarrays_to_parameters(dummy_model.get_weights())

    def on_fit_config_fn(self, rnd: int):
        # logger.info(f"Configuring round {rnd} for training")
        return super().on_fit_config_fn(rnd)

    def on_evaluate_config_fn(self, rnd: int):
        # logger.info(f"Configuring round {rnd} for evaluation")
        return super().on_evaluate_config_fn(rnd)

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using trust scores."""
        if not results:
            return None, {}
        
        # Extract weights and trust scores
        weights_results = [(fit_res.parameters, fit_res.metrics) for _, fit_res in results]
        trust_weights = [metrics.get("trust_score", 0.0) for _, metrics in weights_results]

        # Normalize weights
        trust_weights = np.array(trust_weights, dtype=np.float32)
        trust_weights = trust_weights / np.sum(trust_weights)

        # Get parameters from first client to check structure
        parameters = weights_results[0][0]
        
        # Convert parameters to numpy arrays if they aren't already
        weights_list = []
        for params, _ in weights_results:
            client_weights = []
            for tensor in params.tensors:
                # Ensure tensor is a numpy array of float32
                if isinstance(tensor, bytes):
                    tensor = np.frombuffer(tensor, dtype=np.float32)
                elif not isinstance(tensor, np.ndarray):
                    tensor = np.array(tensor, dtype=np.float32)
                client_weights.append(tensor)
            weights_list.append(client_weights)

        # Calculate weighted average of model updates
        weighted_params = [
            [np.multiply(w, tensor) for tensor in p] 
            for p, w in zip(weights_list, trust_weights)
        ]
        
        # Aggregate parameters
        aggregated_params = [
            np.sum(param_list, axis=0) 
            for param_list in zip(*weighted_params)
        ]

        # Calculate aggregated metrics
        aggregated_metrics = {}
        metrics_keys = ["accuracy", "loss", "precision", "recall", "f1_score", "trust_score"]
        for key in metrics_keys:
            values = [metrics.get(key, 0.0) for _, metrics in weights_results]
            if values:
                aggregated_metrics[key] = float(np.average(values, weights=trust_weights))

        # Convert back to Parameters and return with metrics
        return Parameters(tensors=aggregated_params, tensor_type="numpy.ndarray"), aggregated_metrics

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results with trust score consideration."""
        if not results:
            return None, {}
        
        # Calculate weighted metrics using trust scores
        weighted_metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
        total_weight = 0.0
        
        for client_proxy, eval_res in results:
            client_id = str(client_proxy.cid)
            trust_score = self.client_trust_scores.get(client_id, {}).get('trust_score', 0.0)
            
            # Skip clients with low trust scores
            if trust_score == 0.0:
                continue
                
            total_weight += trust_score
            for metric in weighted_metrics:
                weighted_metrics[metric] += trust_score * eval_res.metrics.get(metric, 0.0)
        
        if total_weight > 0:
            # Normalize metrics
            for metric in weighted_metrics:
                weighted_metrics[metric] /= total_weight
            
            logger.info(f"\nRound {rnd} Global Evaluation Results:")
            logger.info(f"Weighted Loss: {weighted_metrics['loss']:.4f}")
            logger.info(f"Weighted Accuracy: {weighted_metrics['accuracy']:.4f}")
            logger.info(f"Weighted Precision: {weighted_metrics['precision']:.4f}")
            logger.info(f"Weighted Recall: {weighted_metrics['recall']:.4f}")
            logger.info(f"Weighted F1 Score: {weighted_metrics['f1_score']:.4f}")
            
            return weighted_metrics['loss'], weighted_metrics
            
        return None, {}

def create_dummy_model(input_shape):
    """Create ResNet-based model matching client architecture"""
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        input_shape=input_shape,
        weights=None
    )
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.build(input_shape=(None, *input_shape))
    return model

def main():
    logger.info("🚀 Starting Flower server with Trust-Based Aggregation...")
    logger.info("Waiting for clients to connect...")

    strategy = TrustBasedStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        on_evaluate_config_fn=lambda rnd: {"round": rnd},
        accept_failures=False
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=6),
        grpc_max_message_length=1000 * 1024 * 1024,  # 100MB message limit
        strategy=strategy,
        client_manager=fl.server.SimpleClientManager()
    )

if __name__ == "__main__":
    main()