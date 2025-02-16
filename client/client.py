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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

# Define the model
def create_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False  # Freeze the base model

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),  # Add a dense layer for feature extraction
        tf.keras.layers.Dropout(0.5),  # Dropout for regularization
        tf.keras.layers.Dense(4, activation='softmax')  # Final layer for 4 classes
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

class Client(NumPyClient):
    def __init__(self, data_path):
        self.model = create_model()
        self.train_generator, self.val_generator, self.test_generator = get_client_data(data_path)

    def get_parameters(self, config):  # Added config parameter
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        steps_per_epoch = len(self.train_generator)  # Use full dataset
        epochs = 5  # Set the number of epochs for client-side training

        early_stopping = EarlyStopping(
            monitor='val_loss',  
            patience=3,          
            restore_best_weights=True
        )

        # Train the model for multiple epochs
        self.model.fit(
            self.train_generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=self.val_generator,
            callbacks=[early_stopping],
            verbose=1  # Verbosity level (can be adjusted)
        )

        return self.model.get_weights(), len(self.train_generator), {}


    def evaluate(self, parameters, config):
        # Set model weights
        self.model.set_weights(parameters)

        # Evaluate on the entire validation set
        steps = len(self.val_generator)  # Use full validation set

        # Evaluate the model
        loss, accuracy = self.model.evaluate(
            self.val_generator,
            steps=steps,
            verbose=1  # Show evaluation progress
        )

        # Return results
        return float(loss), int(len(self.val_generator) * self.val_generator.batch_size), {"accuracy": float(accuracy)}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python client.py <data_path>")
        sys.exit(1)
    data_path = sys.argv[1]
    fl.client.start_numpy_client(server_address="localhost:8080", client=Client(data_path))