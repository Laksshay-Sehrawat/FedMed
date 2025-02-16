import flwr as fl
import logging
from flwr.server.strategy import FedAvg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LoggingStrategy(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_fit_config_fn(self, rnd: int):
        # logger.info(f"Configuring round {rnd} for training")
        return super().on_fit_config_fn(rnd)

    def on_evaluate_config_fn(self, rnd: int):
        # logger.info(f"Configuring round {rnd} for evaluation")
        return super().on_evaluate_config_fn(rnd)

    def aggregate_fit(self, rnd, results, failures):
        # logger.info(f"Aggregating fit results for round {rnd}")
        return super().aggregate_fit(rnd, results, failures)

    def aggregate_evaluate(self, rnd, results, failures):
        # logger.info(f"Aggregating evaluation results for round {rnd}")
        if results:
            # logger.info(f"Received evaluation results: {results}")
            loss = sum([res.loss for _, res in results]) / len(results)
            accuracy = sum([res.metrics["accuracy"] for _, res in results]) / len(results)
            # logger.info(f"Round {rnd} evaluation results: loss={loss}, accuracy={accuracy}")
            return loss, {"accuracy": accuracy}
        else:
            logger.warning(f"Round {rnd} evaluation returned no results")
            return None, {}

def main():
    logger.info("🚀 Starting Flower server...")
    logger.info("Waiting for clients to connect...")

    strategy = LoggingStrategy(
        fraction_fit=1.0,  # Use 100% of available clients for training
        fraction_evaluate=1.0,  # Use 100% of available clients for evaluation
        min_fit_clients=1,  # Minimum number of clients to participate in training
        min_evaluate_clients=1,  # Minimum number of clients to participate in evaluation
        min_available_clients=2,  # Minimum number of clients that need to be connected to the server
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        grpc_max_message_length=100 * 1024 * 1024,  # 100 MB
        strategy=strategy
    )

if __name__ == "__main__":
    main()