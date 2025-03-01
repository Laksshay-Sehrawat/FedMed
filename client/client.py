from flwr.client import NumPyClient
from tensorflow.keras.callbacks import EarlyStopping
import flwr as fl
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm  # Import tqdm for progress bars
import logging
from PIL import Image  # Ensure Pillow is imported
import sys
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import os
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for trust score calculation
TEST_BATCH_SIZE = 32
W_ACC = 0.3   # Weight for accuracy
W_F1 = 0.3    # Weight for F1 score
W_CONF = 0.2  # Weight for confidence
W_LOSS = 0.2  # Weight for loss
TRUST_THRESHOLD = 0.2  # 20% threshold for trust score deviation

assert W_ACC + W_F1 + W_CONF + W_LOSS == 1.0, "Weights must sum to 1"

# Load your dataset here
def get_client_data(data_path):
    train_datagen = ImageDataGenerator(
        rescale=1./255,        # Normalize pixel values
        rotation_range=20,     # Random rotation
        width_shift_range=0.2, # Horizontal shift
        height_shift_range=0.2,# Vertical shift
        shear_range=0.2,       # Shear transformation
        zoom_range=0.2,        # Random zoom
        horizontal_flip=True,  # Horizontal flip
        fill_mode='nearest'    # Fill missing pixels
    )
    train_generator = train_datagen.flow_from_directory(
        f'{data_path}/train',  # Use smaller dataset
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
    )
    val_datagen = ImageDataGenerator(rescale=1.0/255.0)
    val_generator = val_datagen.flow_from_directory(
        f'{data_path}/valid',  # Use smaller dataset
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_generator = test_datagen.flow_from_directory(
        f'{data_path}/test',  # Use smaller dataset
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )
    return train_generator, val_generator, test_generator

# Define the model with base and personalized layers
def create_model():
    """ResNet-based model matching server architecture"""
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        input_shape=(128, 128, 3),
        weights=None
    )
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def calculate_confidence(predictions):
    """Calculate prediction confidence as max probability."""
    return np.mean(np.max(predictions, axis=1))

def calculate_trust_score(model, test_generator, previous_metrics=None):
    """
    Calculate trust score with confidence and historical comparison.
    Args:
        model: Trained Keras model
        test_generator: Test data generator
        previous_metrics: Previous round metrics for comparison
    Returns:
        trust_score: Float value representing trust score
        metrics: Dictionary containing individual metrics
    """
    # Get predictions for all test data
    num_samples = len(test_generator.filenames)
    steps = np.ceil(num_samples / TEST_BATCH_SIZE)
    
    # Convert steps to integer if it's a float
    steps = int(steps)
    predictions = model.predict(test_generator, steps=steps)
    y_true = test_generator.classes
    y_pred = np.argmax(predictions, axis=1)
    
    # Calculate comprehensive metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    loss = model.evaluate(test_generator, steps=steps, verbose=0)[0]
    confidence = calculate_confidence(predictions)
    
    # Calculate trust score using the formula with confidence
    trust_score = (W_ACC * acc) + (W_F1 * f1) + (W_CONF * confidence) + (W_LOSS * (1 / (1 + loss)))
    
    # Check for significant performance drop if previous metrics exist
    is_reliable = True
    if previous_metrics:
        prev_trust = previous_metrics.get('trust_score', 0)
        if prev_trust > 0:
            drop = (prev_trust - trust_score) / prev_trust
            is_reliable = drop <= TRUST_THRESHOLD
    
    # Convert numpy values to native Python types
    metrics = {
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'loss': float(loss),
        'confidence': float(confidence),
        'trust_score': float(trust_score),
        'is_reliable': bool(is_reliable)
    }
    
    return float(trust_score), metrics

class Client(fl.client.NumPyClient):
    def __init__(self, data_path):
        self.model = create_model()
        self.expected_weights_count = len(self.model.get_weights())  # Store expected count
        try:
            # Verify data loading before connecting
            assert os.path.exists(data_path), f"Path {data_path} does not exist"
            self.train_generator, self.val_generator, self.test_generator = get_client_data(data_path)
            logger.info(f"Data validation passed for {data_path}")
        except Exception as e:
            logger.error(f"CRITICAL DATA ERROR: {str(e)}")
            sys.exit(1)  # Exit immediately on data issues
        self.previous_metrics = {"trust_score": 1.0}

    def _validate_parameters(self, parameters):
        """Validate parameter structure before setting weights"""
        if len(parameters) != self.expected_weights_count:
            logger.error(f"Parameter mismatch! Expected {self.expected_weights_count} layers, got {len(parameters)}")
            logger.debug("Parameter shapes received:")
            for i, p in enumerate(parameters):
                logger.debug(f"Layer {i}: {p.shape if hasattr(p, 'shape') else 'Invalid'}")
            raise ValueError("Server-client model architecture mismatch")

    def get_parameters(self, config):
        """Get model parameters, separating base and personalized layers."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Train the model with both base and personalized layers."""
        try:
            self._validate_parameters(parameters)
            self.model.set_weights(parameters)
        except ValueError as e:
            logger.error("FATAL ARCHITECTURE MISMATCH:")
            logger.error("Client model layers:")
            for i, layer in enumerate(self.model.layers):
                logger.error(f"Layer {i}: {layer.name} - {layer.output_shape}")
            raise

        steps_per_epoch = len(self.train_generator)
        epochs = 5

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )

        # Train the model
        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=self.val_generator,
            callbacks=[early_stopping],
            verbose=1
        )

        # Calculate trust score and metrics
        trust_score, metrics = calculate_trust_score(self.model, self.test_generator, self.previous_metrics)
        
        # Log detailed metrics
        logger.info("\nClient Training Results:")
        logger.info(f"Trust Score: {trust_score:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"Confidence: {metrics['confidence']:.4f}")
        logger.info(f"Is Reliable: {metrics['is_reliable']}")
        
        # Store metrics for next round
        self.previous_metrics = metrics

        # Ensure all metrics are native Python types
        metrics = {k: float(v) if isinstance(v, (np.floating, np.integer)) else bool(v) if isinstance(v, np.bool_) else v 
                  for k, v in metrics.items()}

        return self.model.get_weights(), len(self.train_generator), metrics

    def evaluate(self, parameters, config):
        """Evaluate the model with current parameters."""
        try:
            self._validate_parameters(parameters)
            self.model.set_weights(parameters)
        except ValueError as e:
            logger.error("Evaluation failed due to parameter mismatch")
            logger.error("Verify server and client model architectures match exactly")
            raise
        
        # Evaluate and get comprehensive metrics
        trust_score, metrics = calculate_trust_score(self.model, self.val_generator)
        
        # Ensure all metrics are native Python types
        metrics = {k: float(v) if isinstance(v, (np.floating, np.integer)) else bool(v) if isinstance(v, np.bool_) else v 
                  for k, v in metrics.items()}
        
        return float(metrics['loss']), len(self.val_generator), metrics

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python client.py <data_path>")
        sys.exit(1)
    data_path = sys.argv[1]
    client = Client(data_path)
    
    max_retries = 3
    retry_delay = 10  # seconds
    
    for attempt in range(max_retries):
        try:
            fl.client.start_numpy_client(
                server_address="0.0.0.0:8080",
                client=client,
                grpc_max_message_length=1000 * 1024 * 1024,
                root_certificates=None
            )
            break  # Success - exit loop
        except ConnectionRefusedError:
            if attempt < max_retries - 1:
                logger.warning(f"Connection attempt {attempt+1}/{max_retries} failed. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                logger.error("Maximum connection retries reached. Check server status and network.")
                raise
        except Exception as e:
            logger.error(f"Critical error: {str(e)}")
            raise