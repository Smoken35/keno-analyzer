"""
Pattern visualization module for Keno backtesting system.
"""

import itertools
import json
import logging
import os
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from prometheus_client import Counter, Gauge, Histogram
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("keno_data/pattern_visualization.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Prometheus metrics
VISUALIZATION_TIME = Histogram(
    "pattern_visualization_duration_seconds",
    "Time spent generating visualizations",
    ["visualization_type"],
)

PATTERN_COUNTER = Counter(
    "pattern_analysis_count", "Number of pattern analyses performed", ["pattern_type"]
)

ALERT_COUNTER = Counter("pattern_alert_count", "Number of pattern alerts generated", ["alert_type"])


class PatternVisualizer:
    """Visualization class for Keno pattern analysis."""

    def __init__(self, results_dir: str = "backtest_results"):
        """Initialize the pattern visualizer.

        Args:
            results_dir: Directory to store visualization results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        # Create directory for visualizations
        self.viz_dir = os.path.join(results_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)

        # Alert receiver endpoint
        self.alert_endpoint = "http://localhost:3456/alert"

    def send_alert(self, severity: str, description: str, alert_type: str) -> None:
        """Send an alert to the alert receiver.

        Args:
            severity: Alert severity (critical, warning, info)
            description: Alert description
            alert_type: Type of alert
        """
        try:
            alert_data = {
                "alerts": [
                    {
                        "status": "firing",
                        "labels": {"severity": severity, "alertname": alert_type},
                        "annotations": {"description": description},
                        "startsAt": datetime.utcnow().isoformat(),
                    }
                ]
            }

            response = requests.post(self.alert_endpoint, json=alert_data)
            if response.status_code == 200:
                ALERT_COUNTER.labels(alert_type=alert_type).inc()
                logger.info(f"Alert sent successfully: {description}")
            else:
                logger.error(f"Failed to send alert: {response.status_code}")
        except Exception as e:
            logger.error(f"Error sending alert: {e}")

    def visualize_pattern_analysis(self, results: Dict) -> None:
        """Generate visualizations for pattern analysis results.

        Args:
            results: Pattern analysis results
        """
        try:
            with VISUALIZATION_TIME.labels(visualization_type="pattern_analysis").time():
                # 1. Number frequency heatmap
                plt.figure(figsize=(12, 8))

                # Create grid for 80 numbers (8x10)
                frequency_grid = np.zeros((8, 10))

                # Fill grid with frequency data
                for num_str, data in results["number_frequency"].items():
                    num = int(num_str)
                    row = (num - 1) // 10
                    col = (num - 1) % 10

                    # Use deviation from expected (1.25%)
                    deviation = data["percentage"] - data["expected_percentage"]
                    frequency_grid[row, col] = deviation

                # Check for significant deviations
                significant_deviation = np.any(np.abs(frequency_grid) > 0.3)
                if significant_deviation:
                    self.send_alert(
                        severity="warning",
                        description="Significant number frequency deviation detected",
                        alert_type="FrequencyDeviation",
                    )

                # Create heatmap
                ax = sns.heatmap(
                    frequency_grid,
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    center=0,
                    vmin=-0.5,
                    vmax=0.5,
                )

                plt.title("Number Frequency Deviation from Expected (%)")

                # Add number labels
                for row in range(8):
                    for col in range(10):
                        num = row * 10 + col + 1
                        plt.text(
                            col + 0.5, row + 0.3, str(num), ha="center", fontsize=8, color="black"
                        )

                # Save plot
                plt.tight_layout()
                freq_path = os.path.join(self.viz_dir, "number_frequency_heatmap.png")
                plt.savefig(freq_path)
                plt.close()

                PATTERN_COUNTER.labels(pattern_type="frequency_heatmap").inc()

                # 2. Pair frequency network (top 50 pairs)
                try:
                    import networkx as nx

                    plt.figure(figsize=(14, 14))

                    # Create graph
                    G = nx.Graph()

                    # Add nodes (all numbers)
                    for num in range(1, 81):
                        G.add_node(num)

                    # Add edges for significant pairs
                    max_edges = 50
                    edge_count = 0

                    for pair_str, data in results["pair_frequency"].items():
                        if edge_count >= max_edges:
                            break

                        # Convert string representation to tuple
                        pair = eval(pair_str)

                        # Add edge with weight based on significance ratio
                        G.add_edge(pair[0], pair[1], weight=data["ratio"])
                        edge_count += 1

                    # Check for highly significant pairs
                    significant_pairs = [
                        pair
                        for pair, data in results["pair_frequency"].items()
                        if data["ratio"] > 2.0
                    ]
                    if significant_pairs:
                        self.send_alert(
                            severity="warning",
                            description=f"Highly significant number pairs detected: {significant_pairs[:3]}",
                            alert_type="SignificantPairs",
                        )

                    # Calculate node sizes based on frequency
                    node_sizes = []
                    for node in G.nodes():
                        freq = (
                            results["number_frequency"].get(str(node), {}).get("percentage", 1.25)
                        )
                        # Scale size by frequency
                        node_sizes.append(100 + 500 * (freq / 1.25))

                    # Calculate edge widths based on significance
                    edge_widths = []
                    for u, v, data in G.edges(data=True):
                        edge_widths.append(data["weight"] * 2)

                    # Layout
                    pos = nx.spring_layout(G, seed=42)

                    # Draw network
                    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.7)
                    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5)
                    nx.draw_networkx_labels(G, pos, font_size=8)

                    plt.title("Top Significant Number Pairs Network")
                    plt.axis("off")

                    # Save plot
                    pairs_path = os.path.join(self.viz_dir, "pair_frequency_network.png")
                    plt.savefig(pairs_path)
                    plt.close()

                    PATTERN_COUNTER.labels(pattern_type="pair_network").inc()
                except ImportError:
                    logger.warning("NetworkX library not available for pair network visualization")

                # 3. Grid bias visualization
                plt.figure(figsize=(12, 8))

                # Create subplots for row, column, and quadrant bias
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

                # Row bias
                row_values = [
                    results["grid_patterns"]["row_bias"][row]["bias_ratio"] for row in range(8)
                ]
                sns.barplot(x=list(range(1, 9)), y=row_values, ax=ax1)
                ax1.axhline(y=1, color="r", linestyle="--")
                ax1.set_title("Row Bias Ratio")
                ax1.set_xlabel("Row")
                ax1.set_ylabel("Bias Ratio")

                # Column bias
                col_values = [
                    results["grid_patterns"]["col_bias"][col]["bias_ratio"] for col in range(10)
                ]
                sns.barplot(x=list(range(1, 11)), y=col_values, ax=ax2)
                ax2.axhline(y=1, color="r", linestyle="--")
                ax2.set_title("Column Bias Ratio")
                ax2.set_xlabel("Column")
                ax2.set_ylabel("Bias Ratio")

                # Quadrant bias
                quadrant_values = [
                    results["grid_patterns"]["quadrant_bias"][q]["bias_ratio"] for q in range(4)
                ]
                sns.barplot(
                    x=["Q1 (TL)", "Q2 (TR)", "Q3 (BL)", "Q4 (BR)"], y=quadrant_values, ax=ax3
                )
                ax3.axhline(y=1, color="r", linestyle="--")
                ax3.set_title("Quadrant Bias Ratio")
                ax3.set_xlabel("Quadrant")
                ax3.set_ylabel("Bias Ratio")

                # Check for significant grid bias
                significant_bias = any(
                    abs(v - 1) > 0.2 for v in row_values + col_values + quadrant_values
                )
                if significant_bias:
                    self.send_alert(
                        severity="warning",
                        description="Significant grid bias detected",
                        alert_type="GridBias",
                    )

                plt.tight_layout()
                grid_path = os.path.join(self.viz_dir, "grid_bias.png")
                plt.savefig(grid_path)
                plt.close()

                PATTERN_COUNTER.labels(pattern_type="grid_bias").inc()

                # 4. Hot-cold cycle visualization
                if results["hot_cold_cycles"]:
                    plt.figure(figsize=(14, 8))

                    # Extract data
                    window_indices = [cycle["window_start"] for cycle in results["hot_cold_cycles"]]
                    hot_counts = [cycle["hot_count"] for cycle in results["hot_cold_cycles"]]
                    cold_counts = [cycle["cold_count"] for cycle in results["hot_cold_cycles"]]

                    # Check for extreme hot-cold cycles
                    if max(hot_counts) > 40 or max(cold_counts) > 40:
                        self.send_alert(
                            severity="warning",
                            description="Extreme hot-cold cycle detected",
                            alert_type="HotColdCycle",
                        )

                    # Create stacked area chart
                    plt.stackplot(
                        window_indices,
                        [hot_counts, cold_counts],
                        labels=["Hot Numbers", "Cold Numbers"],
                        alpha=0.7,
                        colors=["crimson", "steelblue"],
                    )

                    plt.title("Hot and Cold Number Trends Over Time")
                    plt.xlabel("Window Start Index")
                    plt.ylabel("Number Count")
                    plt.legend()
                    plt.grid(True, linestyle="--", alpha=0.7)

                    # Save plot
                    hot_cold_path = os.path.join(self.viz_dir, "hot_cold_cycles.png")
                    plt.savefig(hot_cold_path)
                    plt.close()

                    PATTERN_COUNTER.labels(pattern_type="hot_cold_cycles").inc()

                # 5. Sequential pattern visualization
                plt.figure(figsize=(12, 8))

                # Get top sequential patterns
                top_sequential = sorted(
                    [
                        (int(num), data["follow_probability"])
                        for num, data in results["sequential_patterns"].items()
                        if float(data["follow_probability"]) > 0.1
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                )[:20]

                if top_sequential:
                    # Extract data
                    numbers, probabilities = zip(*top_sequential)

                    # Check for highly significant sequential patterns
                    if max(probabilities) > 0.4:
                        self.send_alert(
                            severity="warning",
                            description=f"Highly significant sequential pattern detected: {numbers[0]} ({probabilities[0]:.2%})",
                            alert_type="SequentialPattern",
                        )

                    # Create bar chart
                    plt.bar(range(len(numbers)), probabilities)
                    plt.xticks(range(len(numbers)), numbers, rotation=45)

                    plt.title("Top Numbers with Consecutive Draw Probability")
                    plt.xlabel("Number")
                    plt.ylabel("Probability of Appearing in Consecutive Draws")
                    plt.axhline(y=0.25, color="r", linestyle="--", label="Expected (25%)")
                    plt.legend()
                    plt.grid(True, linestyle="--", alpha=0.7)

                    # Save plot
                    sequential_path = os.path.join(self.viz_dir, "sequential_patterns.png")
                    plt.savefig(sequential_path)
                    plt.close()

                    PATTERN_COUNTER.labels(pattern_type="sequential_patterns").inc()

                logger.info(f"Saved pattern analysis visualizations to {self.viz_dir}")
        except Exception as e:
            logger.error(f"Error creating pattern analysis visualizations: {e}")
            self.send_alert(
                severity="critical",
                description=f"Error in pattern visualization: {str(e)}",
                alert_type="VisualizationError",
            )

    def visualize_bootstrap_results(self, results: Dict) -> None:
        """Generate visualizations for bootstrap analysis results.

        Args:
            results: Bootstrap analysis results
        """
        try:
            with VISUALIZATION_TIME.labels(visualization_type="bootstrap_analysis").time():
                # 1. ROI distribution comparison
                plt.figure(figsize=(14, 10))

                # Prepare data
                strategies = list(results["bootstrap_distributions"].keys())

                # Limit to top strategies for clarity
                strategies_by_mean = sorted(
                    strategies,
                    key=lambda s: results["stability_metrics"][s]["roi_mean"],
                    reverse=True,
                )

                top_strategies = strategies_by_mean[:6]  # Show top 6

                # Create subplot for each top strategy
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()

                for i, strategy in enumerate(top_strategies):
                    roi_values = results["bootstrap_distributions"][strategy]["roi_values"]

                    ax = axes[i]
                    sns.histplot(roi_values, kde=True, ax=ax)

                    # Add vertical line at mean
                    mean_roi = results["stability_metrics"][strategy]["roi_mean"]
                    ax.axvline(mean_roi, color="r", linestyle="--")

                    # Add zero line
                    ax.axvline(0, color="k", linestyle="-", alpha=0.5)

                    ax.set_title(strategy)
                    ax.set_xlabel("ROI (%)")
                    ax.set_ylabel("Frequency")

                    # Add statistics as text
                    stats_text = f"Mean: {mean_roi:.2f}%\nStd: {results['stability_metrics'][strategy]['roi_std']:.2f}%"
                    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment="top")

                    # Check for negative ROI
                    if mean_roi < 0:
                        self.send_alert(
                            severity="warning",
                            description=f"Strategy {strategy} has negative ROI: {mean_roi:.2f}%",
                            alert_type="NegativeROI",
                        )

                plt.suptitle("Bootstrap ROI Distributions")
                plt.tight_layout()

                # Save plot
                roi_dist_path = os.path.join(self.viz_dir, "bootstrap_roi_distributions.png")
                plt.savefig(roi_dist_path)
                plt.close()

                # 2. Stability comparison
                plt.figure(figsize=(12, 8))

                # Prepare data
                strategies = list(results["stability_metrics"].keys())
                roi_means = [results["stability_metrics"][s]["roi_mean"] for s in strategies]
                roi_stds = [results["stability_metrics"][s]["roi_std"] for s in strategies]

                # Create scatter plot (mean vs std)
                plt.scatter(roi_means, roi_stds)

                # Add labels for each point
                for i, strategy in enumerate(strategies):
                    plt.text(roi_means[i], roi_stds[i], strategy, fontsize=8)

                    # Check for high volatility
                    if roi_stds[i] > 10:
                        self.send_alert(
                            severity="warning",
                            description=f"Strategy {strategy} shows high volatility: {roi_stds[i]:.2f}%",
                            alert_type="HighVolatility",
                        )

                # Add reference line for 0% ROI
                plt.axvline(x=0, color="r", linestyle="--", alpha=0.5)

                plt.title("ROI Stability Comparison")
                plt.xlabel("Mean ROI (%)")
                plt.ylabel("ROI Standard Deviation (%)")
                plt.grid(True, linestyle="--", alpha=0.7)

                # Save plot
                stability_path = os.path.join(self.viz_dir, "roi_stability_comparison.png")
                plt.savefig(stability_path)
                plt.close()

                PATTERN_COUNTER.labels(pattern_type="bootstrap_analysis").inc()
                logger.info(f"Saved bootstrap visualizations to {self.viz_dir}")
        except Exception as e:
            logger.error(f"Error creating bootstrap visualizations: {e}")
            self.send_alert(
                severity="critical",
                description=f"Error in bootstrap visualization: {str(e)}",
                alert_type="VisualizationError",
            )

    def visualize_walk_forward_results(self, results: Dict) -> None:
        """Generate visualizations for walk-forward analysis results.

        Args:
            results: Walk-forward analysis results
        """
        try:
            with VISUALIZATION_TIME.labels(visualization_type="walk_forward_analysis").time():
                # 1. ROI over time for each method
                plt.figure(figsize=(14, 8))

                # Plot ROI for each method
                for method in results["method_performance"]:
                    indices = [
                        entry["window_index"] for entry in results["method_performance"][method]
                    ]
                    roi_values = [entry["roi"] for entry in results["method_performance"][method]]

                    plt.plot(indices, roi_values, label=method)

                    # Check for performance degradation
                    if len(roi_values) > 10:
                        recent_roi = roi_values[-5:]
                        if all(r < 0 for r in recent_roi):
                            self.send_alert(
                                severity="warning",
                                description=f"Method {method} shows recent performance degradation",
                                alert_type="PerformanceDegradation",
                            )

                # Add adaptive ensemble if available
                if (
                    "adaptive_ensemble" in results
                    and "window_results" in results["adaptive_ensemble"]
                ):
                    indices = [
                        entry["window_index"]
                        for entry in results["adaptive_ensemble"]["window_results"]
                    ]
                    roi_values = [
                        entry["roi"] for entry in results["adaptive_ensemble"]["window_results"]
                    ]

                    plt.plot(
                        indices, roi_values, label="adaptive_ensemble", linewidth=2, linestyle="--"
                    )

                # Add zero line
                plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)

                plt.title("ROI Over Time")
                plt.xlabel("Window Index")
                plt.ylabel("ROI (%)")
                plt.legend()
                plt.grid(True, linestyle="--", alpha=0.7)

                # Save plot
                plt.tight_layout()
                roi_time_path = os.path.join(self.viz_dir, "roi_over_time.png")
                plt.savefig(roi_time_path)
                plt.close()

                # 2. Method performance comparison
                plt.figure(figsize=(14, 8))

                # Prepare data
                methods = list(results["overall_results"].keys())
                mean_roi = [results["overall_results"][m]["mean_roi"] for m in methods]
                positive_percentage = [
                    results["overall_results"][m]["positive_percentage"] for m in methods
                ]

                # Add adaptive ensemble if available
                if "adaptive_ensemble" in results:
                    methods.append("adaptive_ensemble")
                    mean_roi.append(results["adaptive_ensemble"]["mean_roi"])
                    positive_percentage.append(results["adaptive_ensemble"]["positive_percentage"])

                # Sort by mean ROI
                sorted_indices = np.argsort(mean_roi)
                methods = [methods[i] for i in sorted_indices]
                mean_roi = [mean_roi[i] for i in sorted_indices]
                positive_percentage = [positive_percentage[i] for i in sorted_indices]

                # Create bar chart
                width = 0.35
                x = np.arange(len(methods))

                fig, ax1 = plt.subplots(figsize=(14, 8))

                # Plot mean ROI
                bars1 = ax1.bar(x - width / 2, mean_roi, width, label="Mean ROI (%)")
                ax1.set_ylabel("Mean ROI (%)")
                ax1.axhline(y=0, color="r", linestyle="--", alpha=0.5)

                # Create second y-axis for positive percentage
                ax2 = ax1.twinx()
                bars2 = ax2.bar(
                    x + width / 2,
                    positive_percentage,
                    width,
                    color="orange",
                    label="Positive Windows (%)",
                )
                ax2.set_ylabel("Positive Windows (%)")

                # Add labels
                ax1.set_xticks(x)
                ax1.set_xticklabels(methods, rotation=45, ha="right")
                ax1.set_title("Method Performance Comparison")

                # Add legend
                ax1.legend(loc="upper left")
                ax2.legend(loc="upper right")

                # Save plot
                plt.tight_layout()
                method_comparison_path = os.path.join(
                    self.viz_dir, "method_performance_comparison.png"
                )
                plt.savefig(method_comparison_path)
                plt.close()

                # 3. Adaptive weights evolution
                if results["adaptive_weights"]:
                    plt.figure(figsize=(14, 8))

                    # Plot weight evolution for each method
                    methods = list(results["adaptive_weights"][0]["weights"].keys())

                    for method in methods:
                        indices = [entry["window_index"] for entry in results["adaptive_weights"]]
                        weights = [
                            entry["weights"][method] for entry in results["adaptive_weights"]
                        ]

                        plt.plot(indices, weights, label=method)

                        # Check for weight instability
                        if len(weights) > 10:
                            recent_weights = weights[-5:]
                            if max(recent_weights) - min(recent_weights) > 0.5:
                                self.send_alert(
                                    severity="warning",
                                    description=f"Method {method} shows weight instability",
                                    alert_type="WeightInstability",
                                )

                    plt.title("Adaptive Weights Evolution")
                    plt.xlabel("Window Index")
                    plt.ylabel("Weight")
                    plt.legend()
                    plt.grid(True, linestyle="--", alpha=0.7)

                    # Save plot
                    plt.tight_layout()
                    weights_path = os.path.join(self.viz_dir, "adaptive_weights_evolution.png")
                    plt.savefig(weights_path)
                    plt.close()

                PATTERN_COUNTER.labels(pattern_type="walk_forward_analysis").inc()
                logger.info(f"Saved walk-forward visualizations to {self.viz_dir}")
        except Exception as e:
            logger.error(f"Error creating walk-forward visualizations: {e}")
            self.send_alert(
                severity="critical",
                description=f"Error in walk-forward visualization: {str(e)}",
                alert_type="VisualizationError",
            )
