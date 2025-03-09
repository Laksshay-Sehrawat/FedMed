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
        self.initial_parameters = None
        self.model_structure = None
        logger.info("TrustBasedStrategy initialized with:")
        logger.info(f"min_fit_clients={self.min_fit_clients}")
        logger.info(f"min_evaluate_clients={self.min_evaluate_clients}")

    def initialize_parameters(self, client_manager=None):
        """Initialize parameters with model structure"""
        dummy_model = create_dummy_model((128, 128, 3))
        
        # Store model structure for layer identification
        self.model_structure = {layer.name: i for i, layer in enumerate(dummy_model.layers)}
        
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
        """Aggregate model weights using different strategies for base and personalized layers"""
        if not results:
            return None, {}
        
        # Extract weights and metrics
        weights_results = [(fit_res.parameters, fit_res.metrics) for _, fit_res in results]
        trust_scores = [metrics.get("trust_score", 0.0) for _, metrics in weights_results]
        confidence_scores = [metrics.get("confidence", 0.0) for _, metrics in weights_results]
        reliability_flags = [metrics.get("is_reliable", 1) for _, metrics in weights_results]
        
        # Filter out unreliable clients
        valid_indices = [i for i, reliable in enumerate(reliability_flags) if reliable == 1]
        
        # Convert all parameters to lists of numpy arrays 
        all_weights = [fl.common.parameters_to_ndarrays(params) for params, _ in weights_results]
        
        # Check if all clients are unreliable
        if not valid_indices:
            logger.warning("No reliable clients found - using DIRECT FedAvg aggregation")
            
            # Use simple FedAvg directly on the complete model
            num_clients = len(all_weights)
            if num_clients == 0:
                return None, {}
                
            # Simple average of all weights - this preserves the model architecture exactly
            aggregated_weights = [
                np.sum([weights[i] for weights in all_weights], axis=0) / num_clients
                for i in range(len(all_weights[0]))
            ]
            
            # Calculate aggregated metrics by simple averaging
            aggregated_metrics = {}
            for _, metrics in weights_results:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        aggregated_metrics[k] = aggregated_metrics.get(k, 0) + v / num_clients
            
            logger.info(f"Direct FedAvg complete - aggregated {len(aggregated_weights)} weight tensors")
            
            # Return the directly averaged weights
            return fl.common.ndarrays_to_parameters(aggregated_weights), aggregated_metrics
        
        # If we have reliable clients, proceed with the original trust-based aggregation
        # Get only valid weights from reliable clients
        valid_weights = [all_weights[i] for i in valid_indices]
        valid_trust_scores = [trust_scores[i] for i in valid_indices]
        valid_confidence_scores = [confidence_scores[i] for i in valid_indices]
        
        # Normalize weights for valid clients
        norm_trust_scores = np.array(valid_trust_scores) / np.sum(valid_trust_scores) if sum(valid_trust_scores) > 0 else np.ones(len(valid_indices)) / len(valid_indices)
        norm_confidence_scores = np.array(valid_confidence_scores) / np.sum(valid_confidence_scores) if sum(valid_confidence_scores) > 0 else np.ones(len(valid_indices)) / len(valid_indices)
        
        # Create dummy model to get layer names if not already stored
        if self.model_structure is None:
            dummy_model = create_dummy_model((128, 128, 3))
            self.model_structure = {layer.name: i for i, layer in enumerate(dummy_model.layers)}
        
        # Create layer name mapping for identifying base vs personalized layers
        layer_names = list(self.model_structure.keys())
        
        # Separate base layer weights and personalized layer weights
        base_indices = []
        personalized_indices = []
        
        for idx, name in enumerate(layer_names):
            if "base_" in name:
                base_indices.append(idx)
            else:
                personalized_indices.append(idx)
        
        # Safer approach - just use standard weighted averaging for all layers
        # This still preserves the advantages of trust-based aggregation while being less error-prone
        
        # Initialize aggregated weights with the same structure as the input weights
        aggregated_weights = []
        
        # For each layer in the model
        for layer_idx in range(len(valid_weights[0])):
            # Get all clients' weights for this layer
            layer_weights = [client_w[layer_idx] for client_w in valid_weights]
            
            # Choose weights based on layer type
            if layer_idx in base_indices:
                # Base layers use equal weights
                weights_to_use = np.ones(len(valid_weights)) / len(valid_weights)
            else:
                # Personalized layers use confidence-weighted aggregation
                weights_to_use = norm_confidence_scores
                
            # Weighted average for this layer
            layer_agg = np.zeros_like(layer_weights[0])
            for i, w in enumerate(layer_weights):
                layer_agg += w * weights_to_use[i]
                
            aggregated_weights.append(layer_agg)
        
        # Calculate aggregated metrics using trust scores for weighting
        aggregated_metrics = {}
        metric_keys = ["accuracy", "loss", "precision", "recall", "f1_score", "trust_score", "confidence"]
        
        for key in metric_keys:
            values = [metrics.get(key, 0.0) for _, metrics in weights_results]
            if values:
                # Use trust scores for weighting metrics
                aggregated_metrics[key] = float(np.average([values[i] for i in valid_indices], 
                                                        weights=norm_trust_scores))
        
        # Store metrics for this round
        self.round_metrics[rnd] = aggregated_metrics
        
        # Log results
        logger.info(f"Round {rnd} aggregation complete")
        logger.info(f"Used {len(valid_indices)}/{len(results)} clients")
        logger.info(f"Aggregated metrics: Acc={aggregated_metrics.get('accuracy', 0):.4f}, " 
                    f"F1={aggregated_metrics.get('f1_score', 0):.4f}")
        
        # Verify structure hasn't changed
        if len(aggregated_weights) != len(all_weights[0]):
            logger.error(f"CRITICAL: Aggregation changed layer count from {len(all_weights[0])} to {len(aggregated_weights)}")
            # Fallback to first client's weights as a last resort
            return fl.common.ndarrays_to_parameters(all_weights[0]), aggregated_metrics
        
        # Return the aggregated weights and metrics
        return fl.common.ndarrays_to_parameters(aggregated_weights), aggregated_metrics
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

    def secure_aggregate(self, weights_list, trust_weights, noise_scale=0.0001):
        """Optimized secure aggregation with reduced computational overhead"""
        # Unwrap the nested lists if necessary
        if len(weights_list) == 1 and isinstance(weights_list[0], list):
            weights_list = weights_list[0]
        
        if len(trust_weights) == 1 and isinstance(trust_weights[0], (list, np.ndarray)):
            trust_weights = trust_weights[0]
        
        aggregated_weights = []
        for layer_weights in zip(*weights_list):
            layer_agg = np.zeros_like(layer_weights[0])
            
            # Only add noise to small tensors (e.g., < 10,000 elements)
            add_noise = layer_weights[0].size < 10000
            
            for client_idx, client_weight in enumerate(layer_weights):
                # Add minimal noise only to small layers
                if add_noise:
                    noise_shape = client_weight.shape
                    noise = np.random.normal(0, noise_scale, noise_shape)
                    noisy_weight = client_weight + noise
                    layer_agg += noisy_weight * trust_weights[client_idx]
                else:
                    # Skip noise for large layers
                    layer_agg += client_weight * trust_weights[client_idx]
                    
            aggregated_weights.append(layer_agg)
        
        return aggregated_weights

    def evaluate_global_vs_personalized(
        self,
        round_idx: int,
        global_parameters: Parameters,
        client_results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]]
    ):
        """Compare global model performance against personalized client models"""
        # Extract metrics from client evaluations
        client_metrics = {
            client.cid: eval_res.metrics 
            for client, eval_res in client_results
        }
        
        # Calculate global model metrics (weighted average of client metrics)
        global_metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            values = [metrics.get(metric, 0.0) for metrics in client_metrics.values()]
            if values:
                global_metrics[metric] = float(np.mean(values))
        
        # Calculate personalization gain for each client
        personalization_gain = {}
        for cid, metrics in client_metrics.items():
            gains = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                client_val = metrics.get(metric, 0.0)
                global_val = global_metrics.get(metric, 0.0)
                if global_val > 0:
                    gains[metric] = (client_val - global_val) / global_val
                else:
                    gains[metric] = 0.0
            personalization_gain[cid] = gains
        
        # Log comparison results
        logger.info(f"Round {round_idx} - Global vs. Personalized Model Comparison:")
        logger.info(f"Global metrics: {global_metrics}")
        
        avg_acc_gain = np.mean([gain.get('accuracy', 0.0) for gain in personalization_gain.values()])
        logger.info(f"Average accuracy gain from personalization: {avg_acc_gain:.2%}")
        
        return {
            'global_metrics': global_metrics,
            'personalization_gain': personalization_gain
        }

def create_dummy_model(input_shape=(128, 128, 3)):
    """Create a dummy model matching the client architecture for initialization"""
    # Base layers
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        input_shape=input_shape,
        weights=None
    )
    
    # Mark base layers
    for layer in base_model.layers:
        layer.name = f"base_{layer.name}"
    
    # Personalized layers
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(name="personalized_pooling")(x)
    x = tf.keras.layers.Dense(128, activation='relu', name="personalized_dense1")(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax', name="personalized_output")(x)
    
    # Combined model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
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
        config=fl.server.ServerConfig(num_rounds=2),
        grpc_max_message_length=1024 * 1024 * 1024,  # Increase to 1GB
        strategy=strategy,
        client_manager=fl.server.SimpleClientManager()
    )

if __name__ == "__main__":
    main()