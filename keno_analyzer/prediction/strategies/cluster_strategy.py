"""
Cluster-based prediction strategy using number clusters.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from .base_strategy import BasePredictionStrategy

logger = logging.getLogger(__name__)


class ClusterBasedStrategy(BasePredictionStrategy):
    """Strategy that predicts numbers based on number clusters."""

    def __init__(self, name: str = "ClusterBased", config: Dict[str, Any] = None):
        """Initialize the cluster-based strategy.

        Args:
            name: Name of the strategy
            config: Optional configuration dictionary with keys:
                - top_n_clusters: Number of top clusters to consider (default: 20)
                - cluster_weight_decay: Weight decay factor for cluster age (default: 0.95)
                - min_cluster_size: Minimum size for valid clusters (default: 3)
                - min_cluster_quality: Minimum quality score for clusters (default: 0.3)
                - cohesion_weight: Weight for cluster cohesion in scoring (default: 0.4)
                - frequency_weight: Weight for number frequency in scoring (default: 0.6)
        """
        default_config = {
            "top_n_clusters": 20,
            "cluster_weight_decay": 0.95,
            "min_cluster_size": 3,
            "min_cluster_quality": 0.3,
            "cohesion_weight": 0.4,
            "frequency_weight": 0.6,
        }
        super().__init__(name, {**default_config, **(config or {})})

        self.clusters = []
        self.number_frequencies = defaultdict(float)
        self.number_cluster_scores = defaultdict(float)
        self.recent_draws = []

    def _validate_config(self) -> None:
        """Validate the strategy configuration."""
        if not 1 <= self.config["top_n_clusters"] <= 100:
            raise ValueError("top_n_clusters must be between 1 and 100")
        if not 0 < self.config["cluster_weight_decay"] <= 1:
            raise ValueError("cluster_weight_decay must be between 0 and 1")
        if not 2 <= self.config["min_cluster_size"] <= 20:
            raise ValueError("min_cluster_size must be between 2 and 20")
        if not 0 <= self.config["min_cluster_quality"] <= 1:
            raise ValueError("min_cluster_quality must be between 0 and 1")
        if not 0 <= self.config["cohesion_weight"] <= 1:
            raise ValueError("cohesion_weight must be between 0 and 1")
        if not 0 <= self.config["frequency_weight"] <= 1:
            raise ValueError("frequency_weight must be between 0 and 1")
        if not np.isclose(self.config["cohesion_weight"] + self.config["frequency_weight"], 1.0):
            raise ValueError("cohesion_weight + frequency_weight must equal 1.0")

    def fit(self, historical_data: List[List[int]]) -> None:
        """Fit the strategy to historical data.

        Args:
            historical_data: List of historical Keno draws
        """
        logger.info(f"Fitting cluster strategy on {len(historical_data)} historical draws")

        # Store recent draws for cluster matching
        self.recent_draws = historical_data[-100:]  # Keep last 100 draws

        # Calculate number frequencies
        self._calculate_number_frequencies(historical_data)

        # Extract and weight clusters
        self._extract_clusters(historical_data)

        # Calculate number scores based on clusters
        self._calculate_number_scores()

        logger.info(f"Extracted {len(self.clusters)} clusters")

    def _calculate_number_frequencies(self, historical_data: List[List[int]]) -> None:
        """Calculate weighted frequencies for each number."""
        decay = self.config["cluster_weight_decay"]
        total_weight = 0

        for i, draw in enumerate(reversed(historical_data)):
            weight = decay**i
            total_weight += weight
            for num in draw:
                self.number_frequencies[num] += weight

        # Normalize frequencies
        for num in self.number_frequencies:
            self.number_frequencies[num] /= total_weight

    def _calculate_cluster_cohesion(
        self, cluster: Set[int], historical_data: List[List[int]]
    ) -> float:
        """Calculate cohesion score for a cluster.

        Args:
            cluster: Set of numbers in the cluster
            historical_data: Historical draw data

        Returns:
            Cohesion score between 0 and 1
        """
        if len(cluster) < 2:
            return 0.0

        # Count how often cluster numbers appear together
        together_count = 0
        total_draws = len(historical_data)

        for draw in historical_data:
            cluster_numbers_in_draw = sum(1 for n in cluster if n in draw)
            if cluster_numbers_in_draw >= 2:
                together_count += 1

        return together_count / total_draws

    def _extract_clusters(self, historical_data: List[List[int]]) -> None:
        """Extract and weight number clusters from historical data."""
        # Find frequent number pairs
        pair_counts = defaultdict(int)
        total_draws = len(historical_data)

        for draw in historical_data:
            for i in range(len(draw)):
                for j in range(i + 1, len(draw)):
                    pair = tuple(sorted([draw[i], draw[j]]))
                    pair_counts[pair] += 1

        # Build clusters from frequent pairs
        clusters = []
        for pair, count in pair_counts.items():
            if count / total_draws >= self.config["min_cluster_quality"]:
                cluster = set(pair)
                cohesion = self._calculate_cluster_cohesion(cluster, historical_data)

                if cohesion >= self.config["min_cluster_quality"]:
                    clusters.append(
                        {
                            "numbers": cluster,
                            "size": len(cluster),
                            "support": count / total_draws,
                            "cohesion": cohesion,
                            "quality_score": (count / total_draws) * cohesion,
                        }
                    )

        # Sort clusters by quality score
        self.clusters = sorted(clusters, key=lambda x: x["quality_score"], reverse=True)[
            : self.config["top_n_clusters"]
        ]

    def _calculate_number_scores(self) -> None:
        """Calculate final scores for each number based on clusters and frequency."""
        for num in range(1, 81):
            # Calculate cluster-based score
            cluster_score = 0.0
            for cluster in self.clusters:
                if num in cluster["numbers"]:
                    cluster_score += cluster["quality_score"] * cluster["cohesion"]

            # Combine with frequency score
            freq_score = self.number_frequencies[num]
            self.number_cluster_scores[num] = (
                self.config["cohesion_weight"] * cluster_score
                + self.config["frequency_weight"] * freq_score
            )

    def predict(self, draw_index: int, num_picks: int = 20) -> List[int]:
        """Generate predictions based on cluster analysis.

        Args:
            draw_index: Index of the draw to predict
            num_picks: Number of numbers to predict

        Returns:
            List of predicted numbers
        """
        # Select top numbers based on combined scores
        prediction = sorted(self.number_cluster_scores.items(), key=lambda x: x[1], reverse=True)[
            :num_picks
        ]

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
        # Calculate cluster-based confidence
        cluster_confidences = []
        for cluster in self.clusters:
            # Count how many predicted numbers are in this cluster
            cluster_hits = sum(1 for n in prediction if n in cluster["numbers"])
            if cluster_hits >= 2:  # At least 2 numbers from cluster
                cluster_confidences.append(
                    cluster["quality_score"] * (cluster_hits / len(cluster["numbers"]))
                )

        if not cluster_confidences:
            return 0.0

        # Weight confidence by cluster quality
        total_weight = 0
        weighted_sum = 0
        for cluster in self.clusters:
            cluster_hits = sum(1 for n in prediction if n in cluster["numbers"])
            if cluster_hits >= 2:
                weight = cluster["quality_score"] * cluster_hits
                weighted_sum += cluster["cohesion"] * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0
