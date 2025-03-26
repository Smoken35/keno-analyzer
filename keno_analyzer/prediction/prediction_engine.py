"""
Prediction engine for managing different Keno prediction strategies.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .strategies.base_strategy import BasePredictionStrategy
from .strategies.cluster_strategy import ClusterBasedStrategy
from .strategies.pattern_strategy import PatternBasedStrategy
from .strategies.rule_strategy import RuleBasedStrategy

logger = logging.getLogger(__name__)


class PredictionEngine:
    """Engine for managing and executing Keno prediction strategies."""

    def __init__(self):
        """Initialize the prediction engine."""
        self.strategies: Dict[str, BasePredictionStrategy] = {}
        self.active_strategy: Optional[BasePredictionStrategy] = None

        # Register available strategies
        self.register_strategy(PatternBasedStrategy)
        self.register_strategy(RuleBasedStrategy)
        self.register_strategy(ClusterBasedStrategy)

    def register_strategy(self, strategy_class: Type[BasePredictionStrategy]) -> None:
        """Register a new prediction strategy.

        Args:
            strategy_class: Strategy class to register
        """
        strategy = strategy_class()
        self.strategies[strategy.name] = strategy
        logger.info(f"Registered strategy: {strategy.name}")

    def set_active_strategy(self, strategy_name: str) -> None:
        """Set the active prediction strategy.

        Args:
            strategy_name: Name of the strategy to activate

        Raises:
            ValueError: If strategy not found
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy not found: {strategy_name}")

        self.active_strategy = self.strategies[strategy_name]
        logger.info(f"Set active strategy: {strategy_name}")

    def get_available_strategies(self) -> List[Dict[str, Any]]:
        """Get information about available strategies.

        Returns:
            List of strategy information dictionaries
        """
        return [strategy.get_strategy_info() for strategy in self.strategies.values()]

    def fit(self, historical_data: List[List[int]]) -> None:
        """Fit the active strategy to historical data.

        Args:
            historical_data: List of historical Keno draws

        Raises:
            ValueError: If no active strategy
        """
        if not self.active_strategy:
            raise ValueError("No active strategy selected")

        logger.info(f"Fitting strategy {self.active_strategy.name} on {len(historical_data)} draws")
        self.active_strategy.fit(historical_data)

    def predict(self, draw_index: int, num_picks: int = 20) -> Dict[str, Any]:
        """Generate predictions using the active strategy.

        Args:
            draw_index: Index of the draw to predict
            num_picks: Number of numbers to predict

        Returns:
            Dictionary containing:
                - prediction: List of predicted numbers
                - confidence: Confidence score
                - strategy_info: Strategy information

        Raises:
            ValueError: If no active strategy
        """
        if not self.active_strategy:
            raise ValueError("No active strategy selected")

        prediction = self.active_strategy.predict(draw_index, num_picks)
        confidence = self.active_strategy.get_confidence(prediction)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "strategy_info": self.active_strategy.get_strategy_info(),
        }

    def evaluate_strategy(
        self,
        strategy_name: str,
        historical_data: List[List[int]],
        start_index: int = 0,
        end_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate a strategy's performance on historical data.

        Args:
            strategy_name: Name of the strategy to evaluate
            historical_data: List of historical Keno draws
            start_index: Starting index for evaluation
            end_index: Ending index for evaluation (exclusive)

        Returns:
            Dictionary containing evaluation metrics:
                - total_predictions: Number of predictions made
                - average_hits: Average number of correct predictions
                - average_confidence: Average confidence score
                - hit_rate: Percentage of predictions with at least 10 hits
                - detailed_results: List of prediction results
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy not found: {strategy_name}")

        strategy = self.strategies[strategy_name]
        end_index = end_index or len(historical_data)

        # Fit strategy on data before start_index
        strategy.fit(historical_data[:start_index])

        results = []
        total_hits = 0
        total_confidence = 0

        for i in range(start_index, end_index):
            # Generate prediction
            prediction = strategy.predict(i)
            confidence = strategy.get_confidence(prediction)

            # Calculate hits
            actual = historical_data[i]
            hits = len(set(prediction) & set(actual))

            results.append(
                {
                    "draw_index": i,
                    "prediction": prediction,
                    "actual": actual,
                    "hits": hits,
                    "confidence": confidence,
                }
            )

            total_hits += hits
            total_confidence += confidence

        num_predictions = len(results)
        hit_rate = sum(1 for r in results if r["hits"] >= 10) / num_predictions

        return {
            "total_predictions": num_predictions,
            "average_hits": total_hits / num_predictions,
            "average_confidence": total_confidence / num_predictions,
            "hit_rate": hit_rate,
            "detailed_results": results,
        }
