import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from data_processor import KenoDataProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("models/prediction.log"), logging.StreamHandler()],
)


class KenoPredictor:
    def __init__(self):
        self.models_dir = Path("models")
        self.data_processor = KenoDataProcessor()

    def load_models(self):
        """Load trained models."""
        try:
            # Load Random Forest model
            rf_model = joblib.load(self.models_dir / "random_forest.joblib")

            # Load Neural Network model
            nn_model = tf.keras.models.load_model(self.models_dir / "neural_network.h5")

            # Load scaler
            scaler = joblib.load(self.models_dir / "scaler.joblib")

            return rf_model, nn_model, scaler

        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            raise

    def prepare_input_data(self, latest_data, scaler):
        """Prepare input data for prediction."""
        # Extract features from the latest data
        X = self.data_processor.extract_features(latest_data)

        # Scale the features
        X_scaled = scaler.transform(X)

        return X_scaled

    def post_process_predictions(self, predictions):
        """Post-process predictions to ensure valid Keno numbers."""
        # Round predictions to nearest integer
        predictions = np.round(predictions).astype(int)

        # Ensure numbers are within valid range (1-80)
        predictions = np.clip(predictions, 1, 80)

        # Remove duplicates and ensure exactly 20 numbers
        unique_numbers = np.unique(predictions)
        if len(unique_numbers) < 20:
            # Add random numbers if we have less than 20 unique numbers
            remaining = 20 - len(unique_numbers)
            additional = np.random.choice(
                np.setdiff1d(np.arange(1, 81), unique_numbers), size=remaining, replace=False
            )
            predictions = np.concatenate([unique_numbers, additional])
        elif len(unique_numbers) > 20:
            # Take only 20 numbers if we have more
            predictions = unique_numbers[:20]

        # Sort the numbers
        predictions = np.sort(predictions)

        return predictions

    def generate_predictions(self, latest_data):
        """Generate predictions using both models."""
        try:
            # Load models and scaler
            rf_model, nn_model, scaler = self.load_models()

            # Prepare input data
            X_scaled = self.prepare_input_data(latest_data, scaler)

            # Get predictions from both models
            rf_predictions = rf_model.predict(X_scaled)
            nn_predictions = nn_model.predict(X_scaled)

            # Combine predictions (weighted average)
            combined_predictions = 0.5 * rf_predictions + 0.5 * nn_predictions

            # Post-process predictions
            final_predictions = self.post_process_predictions(combined_predictions)

            # Create prediction report
            report = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "predictions": final_predictions.tolist(),
                "model_confidence": {
                    "random_forest": float(rf_model.score(X_scaled, rf_predictions)),
                    "neural_network": float(
                        nn_model.evaluate(X_scaled, nn_predictions, verbose=0)[0]
                    ),
                },
            }

            # Save predictions
            output_file = (
                self.models_dir / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            pd.DataFrame(report).to_json(output_file, orient="records")

            logging.info(f"Predictions saved to {output_file}")
            return report

        except Exception as e:
            logging.error(f"Error generating predictions: {str(e)}")
            raise


def main():
    predictor = KenoPredictor()

    # Load the latest data
    latest_data = predictor.data_processor.load_data()

    # Generate predictions
    predictions = predictor.generate_predictions(latest_data)

    # Print predictions
    print("\nKeno Number Predictions:")
    print("----------------------")
    print(f"Generated at: {predictions['timestamp']}")
    print("\nPredicted Numbers:")
    print(predictions["predictions"])
    print("\nModel Confidence Scores:")
    print(f"Random Forest: {predictions['model_confidence']['random_forest']:.4f}")
    print(f"Neural Network: {predictions['model_confidence']['neural_network']:.4f}")


if __name__ == "__main__":
    main()
