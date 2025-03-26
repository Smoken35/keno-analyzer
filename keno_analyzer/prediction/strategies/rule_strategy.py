"""
Rule-based prediction strategy using association rules.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from .base_strategy import BasePredictionStrategy

logger = logging.getLogger(__name__)


class RuleBasedStrategy(BasePredictionStrategy):
    """Strategy that predicts numbers based on association rules."""

    def __init__(self, name: str = "RuleBased", config: Dict[str, Any] = None):
        """Initialize the rule-based strategy.

        Args:
            name: Name of the strategy
            config: Optional configuration dictionary with keys:
                - min_lift: Minimum lift threshold (default: 1.5)
                - min_confidence: Minimum confidence threshold (default: 0.3)
                - top_n_rules: Number of top rules to consider (default: 100)
                - rule_weight_decay: Weight decay factor for rule age (default: 0.95)
                - antecedent_weight: Weight for antecedent numbers (default: 0.6)
                - consequent_weight: Weight for consequent numbers (default: 0.4)
        """
        default_config = {
            "min_lift": 1.5,
            "min_confidence": 0.3,
            "top_n_rules": 100,
            "rule_weight_decay": 0.95,
            "antecedent_weight": 0.6,
            "consequent_weight": 0.4,
        }
        super().__init__(name, {**default_config, **(config or {})})

        self.rules = []
        self.number_frequencies = defaultdict(float)
        self.recent_draws = []

    def _validate_config(self) -> None:
        """Validate the strategy configuration."""
        if not 1 <= self.config["min_lift"] <= 10:
            raise ValueError("min_lift must be between 1 and 10")
        if not 0 < self.config["min_confidence"] <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        if not 1 <= self.config["top_n_rules"] <= 1000:
            raise ValueError("top_n_rules must be between 1 and 1000")
        if not 0 < self.config["rule_weight_decay"] <= 1:
            raise ValueError("rule_weight_decay must be between 0 and 1")
        if not 0 <= self.config["antecedent_weight"] <= 1:
            raise ValueError("antecedent_weight must be between 0 and 1")
        if not 0 <= self.config["consequent_weight"] <= 1:
            raise ValueError("consequent_weight must be between 0 and 1")
        if not np.isclose(self.config["antecedent_weight"] + self.config["consequent_weight"], 1.0):
            raise ValueError("antecedent_weight + consequent_weight must equal 1.0")

    def fit(self, historical_data: List[List[int]]) -> None:
        """Fit the strategy to historical data.

        Args:
            historical_data: List of historical Keno draws
        """
        logger.info(f"Fitting rule strategy on {len(historical_data)} historical draws")

        # Store recent draws for rule matching
        self.recent_draws = historical_data[-100:]  # Keep last 100 draws

        # Calculate number frequencies
        self._calculate_number_frequencies(historical_data)

        # Extract and weight rules
        self._extract_rules(historical_data)

        logger.info(f"Extracted {len(self.rules)} rules")

    def _calculate_number_frequencies(self, historical_data: List[List[int]]) -> None:
        """Calculate weighted frequencies for each number."""
        decay = self.config["rule_weight_decay"]
        total_weight = 0

        for i, draw in enumerate(reversed(historical_data)):
            weight = decay**i
            total_weight += weight
            for num in draw:
                self.number_frequencies[num] += weight

        # Normalize frequencies
        for num in self.number_frequencies:
            self.number_frequencies[num] /= total_weight

    def _extract_rules(self, historical_data: List[List[int]]) -> None:
        """Extract and weight association rules from historical data."""
        # Count item frequencies
        item_counts = defaultdict(int)
        pair_counts = defaultdict(int)
        total_draws = len(historical_data)

        for draw in historical_data:
            # Count single items
            for num in draw:
                item_counts[num] += 1

            # Count pairs
            for i in range(len(draw)):
                for j in range(i + 1, len(draw)):
                    pair = tuple(sorted([draw[i], draw[j]]))
                    pair_counts[pair] += 1

        # Calculate support and confidence
        rules = []
        for pair, count in pair_counts.items():
            support = count / total_draws
            item1, item2 = pair

            # Calculate confidence in both directions
            conf1 = count / item_counts[item1]
            conf2 = count / item_counts[item2]

            # Calculate lift
            lift1 = conf1 / (item_counts[item2] / total_draws)
            lift2 = conf2 / (item_counts[item1] / total_draws)

            # Create rules if they meet thresholds
            if lift1 >= self.config["min_lift"] and conf1 >= self.config["min_confidence"]:
                rules.append(
                    {
                        "antecedent": [item1],
                        "consequent": [item2],
                        "support": support,
                        "confidence": conf1,
                        "lift": lift1,
                    }
                )

            if lift2 >= self.config["min_lift"] and conf2 >= self.config["min_confidence"]:
                rules.append(
                    {
                        "antecedent": [item2],
                        "consequent": [item1],
                        "support": support,
                        "confidence": conf2,
                        "lift": lift2,
                    }
                )

        # Sort rules by lift and confidence
        self.rules = sorted(rules, key=lambda x: (x["lift"], x["confidence"]), reverse=True)[
            : self.config["top_n_rules"]
        ]

    def predict(self, draw_index: int, num_picks: int = 20) -> List[int]:
        """Generate predictions based on association rules.

        Args:
            draw_index: Index of the draw to predict
            num_picks: Number of numbers to predict

        Returns:
            List of predicted numbers
        """
        # Initialize prediction scores
        number_scores = defaultdict(float)

        # Score numbers based on rules
        for rule in self.rules:
            weight = rule["lift"] * rule["confidence"]

            # Score antecedent numbers
            for num in rule["antecedent"]:
                number_scores[num] += weight * self.config["antecedent_weight"]

            # Score consequent numbers
            for num in rule["consequent"]:
                number_scores[num] += weight * self.config["consequent_weight"]

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
        # Calculate rule-based confidence
        rule_confidences = []
        for rule in self.rules:
            # Check if prediction contains antecedent
            if all(num in prediction for num in rule["antecedent"]):
                # Check if prediction contains consequent
                consequent_hits = sum(1 for num in rule["consequent"] if num in prediction)
                if consequent_hits > 0:
                    rule_confidences.append(
                        rule["confidence"] * (consequent_hits / len(rule["consequent"]))
                    )

        if not rule_confidences:
            return 0.0

        # Weight confidence by rule lift
        total_weight = 0
        weighted_sum = 0
        for rule in self.rules:
            if all(num in prediction for num in rule["antecedent"]):
                consequent_hits = sum(1 for num in rule["consequent"] if num in prediction)
                if consequent_hits > 0:
                    weight = rule["lift"] * consequent_hits
                    weighted_sum += rule["confidence"] * weight
                    total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0
