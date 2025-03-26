import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("models/training.log"), logging.StreamHandler()],
)


class KenoModelTrainer:
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

    def load_data(self):
        """Load processed training data."""
        X = np.load(self.models_dir / "X.npy")
        y = np.load(self.models_dir / "y.npy")
        return X, y

    def train_random_forest(self, X, y):
        """Train a Random Forest model."""
        logging.info("Training Random Forest model...")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = rf_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        logging.info(f"Random Forest MSE: {mse:.4f}")
        logging.info(f"Random Forest MAE: {mae:.4f}")

        # Save the model
        joblib.dump(rf_model, self.models_dir / "random_forest.joblib")
        return rf_model

    def build_neural_network(self, input_shape):
        """Build a neural network model."""
        model = models.Sequential(
            [
                layers.Dense(256, activation="relu", input_shape=input_shape),
                layers.Dropout(0.3),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(64, activation="relu"),
                layers.Dense(20, activation="linear"),
            ]
        )

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        return model

    def train_neural_network(self, X, y):
        """Train a neural network model."""
        logging.info("Training Neural Network model...")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build and train the model
        model = self.build_neural_network((X.shape[1],))

        # Early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        # Train the model
        history = model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1,
        )

        # Evaluate the model
        test_loss, test_mae = model.evaluate(X_test, y_test)
        logging.info(f"Neural Network Test Loss: {test_loss:.4f}")
        logging.info(f"Neural Network Test MAE: {test_mae:.4f}")

        # Save the model
        model.save(self.models_dir / "neural_network.h5")
        return model, history

    def train_models(self):
        """Train all models."""
        try:
            # Load data
            X, y = self.load_data()

            # Train Random Forest
            rf_model = self.train_random_forest(X, y)

            # Train Neural Network
            nn_model, history = self.train_neural_network(X, y)

            logging.info("All models trained successfully")

            return rf_model, nn_model

        except Exception as e:
            logging.error(f"Error training models: {str(e)}")
            raise


def main():
    trainer = KenoModelTrainer()
    rf_model, nn_model = trainer.train_models()
    logging.info("Model training completed successfully")


if __name__ == "__main__":
    main()
