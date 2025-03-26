"""
Pattern-based prediction strategy using frequent patterns.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Set

import numpy as np

from .base_strategy import BasePredictionStrategy

logger = logging.getLogger(__name__)


class PatternBasedStrategy(BasePredictionStrategy):
    """Strategy that predicts numbers based on frequent patterns."""

    def __init__(self, name: str = "PatternBased", config: Dict[str, Any] = None):
        """Initialize the pattern-based strategy.

        Args:
            name: Name of the strategy
            config: Optional configuration dictionary with keys:
                - min_support: Minimum support threshold (default: 0.1)
                - max_pattern_size: Maximum pattern size (default: 10)
                - top_n_patterns: Number of top patterns to consider (default: 100)
                - pattern_weight_decay: Weight decay factor for pattern age (default: 0.95)
        """
        default_config = {
            "min_support": 0.1,
            "max_pattern_size": 10,
            "top_n_patterns": 100,
            "pattern_weight_decay": 0.95,
        }
        super().__init__(name, {**default_config, **(config or {})})

        self.patterns = []
        self.pattern_weights = {}
        self.number_frequencies = defaultdict(float)
        self.recent_draws = []

    def _validate_config(self) -> None:
        """Validate the strategy configuration."""
        if not 0 < self.config["min_support"] <= 1:
            raise ValueError("min_support must be between 0 and 1")
        if not 1 <= self.config["max_pattern_size"] <= 20:
            raise ValueError("max_pattern_size must be between 1 and 20")
        if not 1 <= self.config["top_n_patterns"] <= 1000:
            raise ValueError("top_n_patterns must be between 1 and 1000")
        if not 0 < self.config["pattern_weight_decay"] <= 1:
            raise ValueError("pattern_weight_decay must be between 0 and 1")

    def fit(self, historical_data: List[List[int]]) -> None:
        """Fit the strategy to historical data.

        Args:
            historical_data: List of historical Keno draws
        """
        logger.info(f"Fitting pattern strategy on {len(historical_data)} historical draws")

        # Store recent draws for pattern matching
        self.recent_draws = historical_data[-100:]  # Keep last 100 draws

        # Calculate number frequencies
        self._calculate_number_frequencies(historical_data)

        # Extract and weight patterns
        self._extract_patterns(historical_data)

        logger.info(f"Extracted {len(self.patterns)} patterns")

    def _calculate_number_frequencies(self, historical_data: List[List[int]]) -> None:
        """Calculate weighted frequencies for each number."""
        decay = self.config["pattern_weight_decay"]
        total_weight = 0

        for i, draw in enumerate(reversed(historical_data)):
            weight = decay**i
            total_weight += weight
            for num in draw:
                self.number_frequencies[num] += weight

        # Normalize frequencies
        for num in self.number_frequencies:
            self.number_frequencies[num] /= total_weight

    def _extract_patterns(self, historical_data: List[List[int]]) -> None:
        """Extract and weight patterns from historical data."""
        pattern_counts = defaultdict(int)
        pattern_weights = defaultdict(float)
        decay = self.config["pattern_weight_decay"]

        # Count patterns with decay weights
        for i, draw in enumerate(reversed(historical_data)):
            weight = decay**i
            for size in range(2, self.config["max_pattern_size"] + 1):
                for j in range(len(draw) - size + 1):
                    pattern = tuple(sorted(draw[j : j + size]))
                    pattern_counts[pattern] += 1
                    pattern_weights[pattern] += weight

        # Convert to list and sort by weighted support
        total_draws = len(historical_data)
        self.patterns = []
        for pattern, count in pattern_counts.items():
            support = count / total_draws
            if support >= self.config["min_support"]:
                self.patterns.append(
                    {
                        "numbers": list(pattern),
                        "support": support,
                        "weight": pattern_weights[pattern],
                    }
                )

        # Sort patterns by weighted support
        self.patterns.sort(key=lambda x: x["weight"], reverse=True)
        self.patterns = self.patterns[: self.config["top_n_patterns"]]

    def predict(self, draw_index: int, num_picks: int = 20) -> List[int]:
        """Generate predictions based on patterns.

        Args:
            draw_index: Index of the draw to predict
            num_picks: Number of numbers to predict

        Returns:
            List of predicted numbers
        """
        # Initialize prediction scores
        number_scores = defaultdict(float)

        # Score numbers based on patterns
        for pattern in self.patterns:
            weight = pattern["weight"]
            for num in pattern["numbers"]:
                number_scores[num] += weight

        # Add base frequency scores
        for num in range(1, 81):
            number_scores[num] += self.number_frequencies[num]

        # Select top numbers
        prediction = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)[:num_picks]

        prediction = [num for num, _ in prediction]
        self._validate_prediction(prediction, num_picks)

        return prediction

    def get_confidence(self, prediction: List[int]) -> float:
        """Get confidence score for a prediction.

        Args:
            prediction: List of predicted numbers

        Returns:
            Confidence score between 0 and 1
        """
        # Calculate average pattern support for prediction
        pattern_supports = []
        for pattern in self.patterns:
            if all(num in prediction for num in pattern["numbers"]):
                pattern_supports.append(pattern["support"])

        if not pattern_supports:
            return 0.0

        # Weight confidence by pattern support and size
        total_weight = 0
        weighted_sum = 0
        for pattern in self.patterns:
            if all(num in prediction for num in pattern["numbers"]):
                weight = pattern["weight"] * len(pattern["numbers"])
                weighted_sum += pattern["support"] * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0
