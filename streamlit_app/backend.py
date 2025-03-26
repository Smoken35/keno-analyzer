"""
Backend module for the Keno prediction Streamlit app.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from keno_analyzer.prediction import PredictionEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PredictionBackend:
    """Backend class for handling predictions."""

    def __init__(self):
        """Initialize the prediction backend."""
        self.engine = PredictionEngine()
        self.data: Optional[List[List[int]]] = None
        self.strategy_map = {
            "Pattern-Based": "PatternBased",
            "Rule-Based": "RuleBased",
            "Cluster-Based": "ClusterBased",
        }

    def process_csv(self, file) -> Tuple[bool, str]:
        """Process uploaded CSV file.

        Args:
            file: Uploaded CSV file object

        Returns:
            Tuple of (success, message)
        """
        try:
            # Read CSV file
            df = pd.read_csv(file)

            # Validate columns
            required_columns = ["draw_date", "numbers", "draw_number"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return False, f"Missing required columns: {', '.join(missing_columns)}"

            # Convert numbers column to list of integers
            if isinstance(df["numbers"].iloc[0], str):
                # If numbers are comma-separated string
                self.data = [[int(num.strip()) for num in row.split(",")] for row in df["numbers"]]
            else:
                # If numbers are already in list format
                self.data = df["numbers"].tolist()

            # Validate data
            for i, draw in enumerate(self.data):
                if not all(isinstance(num, int) and 1 <= num <= 80 for num in draw):
                    return False, f"Invalid numbers in draw {i+1}"

            logger.info(f"Successfully processed CSV with {len(self.data)} draws")
            return True, f"Successfully loaded {len(self.data)} draws"

        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            return False, f"Error processing CSV: {str(e)}"

    def generate_prediction(
        self, strategy: str, pick_size: int
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Generate prediction using selected strategy.

        Args:
            strategy: Name of the strategy to use
            pick_size: Number of numbers to predict

        Returns:
            Tuple of (success, message, prediction_results)
        """
        try:
            if not self.data:
                return False, "No data loaded", None

            # Map strategy name to engine strategy
            if strategy not in self.strategy_map:
                return False, f"Unknown strategy: {strategy}", None

            engine_strategy = self.strategy_map[strategy]

            # Set active strategy and fit
            self.engine.set_active_strategy(engine_strategy)
            self.engine.fit(self.data)

            # Generate prediction
            prediction_results = self.engine.predict(draw_index=len(self.data), num_picks=pick_size)

            # Calculate hit rates
            hit_rates = self._calculate_hit_rates(pick_size)
            prediction_results["hit_rates"] = hit_rates

            logger.info(f"Generated prediction using {strategy}")
            return True, "Successfully generated prediction", prediction_results

        except Exception as e:
            logger.error(f"Error generating prediction: {str(e)}")
            return False, f"Error generating prediction: {str(e)}", None

    def _calculate_hit_rates(self, pick_size: int) -> Dict[str, float]:
        """Calculate hit rates for each strategy.

        Args:
            pick_size: Number of picks to use

        Returns:
            Dictionary of strategy names and their hit rates
        """
        hit_rates = {}

        for display_name, engine_name in self.strategy_map.items():
            try:
                # Evaluate strategy
                evaluation = self.engine.evaluate_strategy(
                    strategy_name=engine_name,
                    historical_data=self.data,
                    start_index=len(self.data) - 100,  # Use last 100 draws
                )

                # Calculate hit rate percentage
                hit_rates[display_name] = evaluation["hit_rate"] * 100

            except Exception as e:
                logger.error(f"Error calculating hit rate for {display_name}: {str(e)}")
                hit_rates[display_name] = 0.0

        return hit_rates
