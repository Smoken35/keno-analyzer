#!/usr/bin/env python3
"""
Cluster Builder - Groups Keno numbers into co-occurrence clusters using graph-based clustering.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np
from scipy.spatial.distance import jaccard
from sklearn.cluster import SpectralClustering

logger = logging.getLogger(__name__)


@dataclass
class NumberCluster:
    """Data class for storing number cluster information."""

    numbers: List[int]
    cohesion: float
    coverage: float
    frequency: float
    timestamp: str


class ClusterBuilder:
    """Implements graph-based clustering for Keno numbers."""

    def __init__(self, min_cohesion: float = 0.3, min_coverage: float = 0.2):
        """
        Initialize the cluster builder.

        Args:
            min_cohesion: Minimum cluster cohesion threshold
            min_coverage: Minimum cluster coverage threshold
        """
        self.min_cohesion = min_cohesion
        self.min_coverage = min_coverage
        self.clusters: List[NumberCluster] = []
        self.transaction_count = 0
        self.graph = None

    def _create_cooccurrence_matrix(self, transactions: List[Set[int]]) -> np.ndarray:
        """
        Create co-occurrence matrix from transactions.

        Args:
            transactions: List of transaction sets

        Returns:
            Numpy array of co-occurrence frequencies
        """
        matrix = np.zeros((80, 80))

        for transaction in transactions:
            # Convert transaction to list for easier indexing
            trans_list = sorted(list(transaction))

            # Update co-occurrence counts
            for i in range(len(trans_list)):
                for j in range(i + 1, len(trans_list)):
                    matrix[trans_list[i] - 1][trans_list[j] - 1] += 1
                    matrix[trans_list[j] - 1][trans_list[i] - 1] += 1

        return matrix

    def _create_similarity_matrix(self, cooccurrence_matrix: np.ndarray) -> np.ndarray:
        """
        Create similarity matrix using Jaccard similarity.

        Args:
            cooccurrence_matrix: Co-occurrence frequency matrix

        Returns:
            Numpy array of similarity scores
        """
        similarity = np.zeros((80, 80))

        for i in range(80):
            for j in range(i + 1, 80):
                # Calculate Jaccard similarity
                intersection = cooccurrence_matrix[i][j]
                union = cooccurrence_matrix[i][i] + cooccurrence_matrix[j][j] - intersection
                similarity[i][j] = intersection / union if union > 0 else 0
                similarity[j][i] = similarity[i][j]

        return similarity

    def _build_graph(self, similarity_matrix: np.ndarray):
        """
        Build NetworkX graph from similarity matrix.

        Args:
            similarity_matrix: Matrix of similarity scores
        """
        self.graph = nx.Graph()

        # Add edges with similarity weights
        for i in range(80):
            for j in range(i + 1, 80):
                if similarity_matrix[i][j] > 0:
                    self.graph.add_edge(i + 1, j + 1, weight=similarity_matrix[i][j])

    def _calculate_cluster_metrics(
        self, cluster: Set[int], transactions: List[Set[int]]
    ) -> Tuple[float, float, float]:
        """
        Calculate cluster cohesion, coverage, and frequency.

        Args:
            cluster: Set of numbers in cluster
            transactions: List of transaction sets

        Returns:
            Tuple of (cohesion, coverage, frequency)
        """
        # Calculate frequency
        cluster_count = sum(1 for t in transactions if any(n in t for n in cluster))
        frequency = cluster_count / self.transaction_count

        # Calculate coverage (how often all numbers appear together)
        full_matches = sum(1 for t in transactions if cluster.issubset(t))
        coverage = full_matches / self.transaction_count

        # Calculate cohesion (average similarity between numbers)
        similarities = []
        for i, n1 in enumerate(cluster):
            for n2 in list(cluster)[i + 1 :]:
                if self.graph.has_edge(n1, n2):
                    similarities.append(self.graph[n1][n2]["weight"])

        cohesion = np.mean(similarities) if similarities else 0

        return cohesion, coverage, frequency

    def build_clusters(
        self, transactions: List[Set[int]], n_clusters: int = 10
    ) -> List[NumberCluster]:
        """
        Build number clusters using graph-based clustering.

        Args:
            transactions: List of transaction sets
            n_clusters: Number of clusters to create

        Returns:
            List of number clusters
        """
        logger.info("Starting cluster building...")

        self.transaction_count = len(transactions)

        # Create co-occurrence and similarity matrices
        cooccurrence_matrix = self._create_cooccurrence_matrix(transactions)
        similarity_matrix = self._create_similarity_matrix(cooccurrence_matrix)

        # Build graph
        self._build_graph(similarity_matrix)

        # Perform spectral clustering
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="discretize",
            random_state=42,
        )

        # Fit and predict clusters
        labels = clustering.fit_predict(similarity_matrix)

        # Create clusters from labels
        self.clusters = []
        for cluster_id in range(n_clusters):
            cluster_numbers = [i + 1 for i, label in enumerate(labels) if label == cluster_id]

            if cluster_numbers:  # Skip empty clusters
                cluster_set = set(cluster_numbers)
                cohesion, coverage, frequency = self._calculate_cluster_metrics(
                    cluster_set, transactions
                )

                # Only keep clusters meeting thresholds
                if cohesion >= self.min_cohesion and coverage >= self.min_coverage:
                    self.clusters.append(
                        NumberCluster(
                            numbers=sorted(cluster_numbers),
                            cohesion=cohesion,
                            coverage=coverage,
                            frequency=frequency,
                            timestamp=datetime.now().isoformat(),
                        )
                    )

        # Sort clusters by frequency and cohesion
        self.clusters.sort(key=lambda x: (x.frequency, x.cohesion), reverse=True)

        logger.info(f"Built {len(self.clusters)} clusters")
        return self.clusters

    def _calculate_cluster_quality(self, cluster: NumberCluster) -> float:
        """
        Calculate a quality score for a cluster.

        Args:
            cluster: Number cluster

        Returns:
            Quality score (0-1)
        """
        # Weighted combination of metrics
        weights = {"cohesion": 0.4, "coverage": 0.3, "frequency": 0.3}

        return (
            weights["cohesion"] * cluster.cohesion
            + weights["coverage"] * cluster.coverage
            + weights["frequency"] * cluster.frequency
        )

    def rank_clusters(self) -> List[Dict]:
        """
        Rank clusters by quality score.

        Returns:
            List of ranked clusters with quality scores
        """
        ranked_clusters = []

        for cluster in self.clusters:
            quality_score = self._calculate_cluster_quality(cluster)
            ranked_clusters.append(
                {
                    "numbers": cluster.numbers,
                    "cohesion": cluster.cohesion,
                    "coverage": cluster.coverage,
                    "frequency": cluster.frequency,
                    "quality_score": quality_score,
                    "timestamp": cluster.timestamp,
                }
            )

        # Sort by quality score
        ranked_clusters.sort(key=lambda x: x["quality_score"], reverse=True)

        return ranked_clusters

    def save_results(self, output_file: str):
        """
        Save cluster results to JSON file.

        Args:
            output_file: Path to save results
        """
        import json
        from pathlib import Path

        # Rank clusters and prepare output
        ranked_clusters = self.rank_clusters()
        results = {
            "clusters": ranked_clusters,
            "metadata": {
                "total_clusters": len(ranked_clusters),
                "min_cohesion": self.min_cohesion,
                "min_coverage": self.min_coverage,
                "timestamp": datetime.now().isoformat(),
            },
        }

        # Create output directory if needed
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to file
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_file}")

    def get_top_clusters(self, n: int = 5) -> List[Dict]:
        """
        Get top N clusters by quality score.

        Args:
            n: Number of clusters to return

        Returns:
            List of top N clusters
        """
        ranked_clusters = self.rank_clusters()
        return ranked_clusters[:n]

    def analyze_cluster_stability(
        self, cluster: NumberCluster, transactions: List[Set[int]], window_size: int = 100
    ) -> Dict:
        """
        Analyze cluster stability over time.

        Args:
            cluster: Number cluster to analyze
            transactions: List of transaction sets
            window_size: Size of sliding window

        Returns:
            Dictionary containing stability analysis
        """
        cluster_set = set(cluster.numbers)
        stability_scores = []

        # Calculate stability in sliding windows
        for i in range(0, len(transactions) - window_size + 1):
            window = transactions[i : i + window_size]
            cohesion, coverage, frequency = self._calculate_cluster_metrics(cluster_set, window)
            stability_scores.append(
                {
                    "window_start": i,
                    "cohesion": cohesion,
                    "coverage": coverage,
                    "frequency": frequency,
                }
            )

        # Calculate stability metrics
        cohesion_scores = [s["cohesion"] for s in stability_scores]
        coverage_scores = [s["coverage"] for s in stability_scores]
        frequency_scores = [s["frequency"] for s in stability_scores]

        return {
            "stability_scores": stability_scores,
            "cohesion_mean": np.mean(cohesion_scores),
            "cohesion_std": np.std(cohesion_scores),
            "coverage_mean": np.mean(coverage_scores),
            "coverage_std": np.std(coverage_scores),
            "frequency_mean": np.mean(frequency_scores),
            "frequency_std": np.std(frequency_scores),
        }
