"""
Advanced visualization functions for Keno strategy comparison.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

# Custom color scheme for strategies
STRATEGY_COLORS = {
    "PatternBased": "#FF6B6B",  # Coral Red
    "RuleBased": "#4ECDC4",  # Turquoise
    "ClusterBased": "#45B7D1",  # Sky Blue
}


def plot_time_series(
    results: Dict[str, Dict[str, Any]], output_dir: str, window_size: int = 50
) -> str:
    """Plot rolling averages of hits and confidence over time for each strategy.

    Args:
        results: Dictionary mapping strategy names to their evaluation results
        output_dir: Directory to save the plot
        window_size: Size of the rolling window for averaging

    Returns:
        Path to the saved plot file
    """
    output_path = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_path / f"time_series_{timestamp}.png"

    # Set style
    plt.style.use("seaborn")
    plt.figure(figsize=(15, 10))

    # Create subplots
    gs = plt.GridSpec(2, 1, height_ratios=[1, 1])

    # 1. Rolling Average Hits
    ax1 = plt.subplot(gs[0])
    for strategy, res in results.items():
        hits = [r["hits"] for r in res["detailed_results"]]
        rolling_avg = np.convolve(hits, np.ones(window_size) / window_size, mode="valid")
        ax1.plot(rolling_avg, label=strategy, color=STRATEGY_COLORS.get(strategy, "gray"))

    ax1.set_title(f"Rolling Average Hits (Window Size: {window_size})")
    ax1.set_xlabel("Draw Index")
    ax1.set_ylabel("Average Hits")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Rolling Average Confidence
    ax2 = plt.subplot(gs[1])
    for strategy, res in results.items():
        confidences = [r["confidence"] for r in res["detailed_results"]]
        rolling_avg = np.convolve(confidences, np.ones(window_size) / window_size, mode="valid")
        ax2.plot(rolling_avg, label=strategy, color=STRATEGY_COLORS.get(strategy, "gray"))

    ax2.set_title(f"Rolling Average Confidence (Window Size: {window_size})")
    ax2.set_xlabel("Draw Index")
    ax2.set_ylabel("Average Confidence")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved time series plot to: {plot_path}")
    return str(plot_path)


def plot_hit_distribution(results: Dict[str, Dict[str, Any]], output_dir: str) -> str:
    """Plot boxplots showing the distribution of hits for each strategy.

    Args:
        results: Dictionary mapping strategy names to their evaluation results
        output_dir: Directory to save the plot

    Returns:
        Path to the saved plot file
    """
    output_path = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_path / f"hit_distribution_{timestamp}.png"

    # Prepare data for boxplot
    data = []
    labels = []
    for strategy, res in results.items():
        hits = [r["hits"] for r in res["detailed_results"]]
        data.append(hits)
        labels.append(strategy)

    # Set style
    plt.style.use("seaborn")
    plt.figure(figsize=(12, 6))

    # Create boxplot
    bp = plt.boxplot(data, labels=labels, patch_artist=True)

    # Customize colors
    for patch, strategy in zip(bp["boxes"], labels):
        patch.set_facecolor(STRATEGY_COLORS.get(strategy, "gray"))

    plt.title("Distribution of Hits by Strategy")
    plt.ylabel("Number of Hits")
    plt.grid(True, alpha=0.3)

    # Rotate labels for better readability
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved hit distribution plot to: {plot_path}")
    return str(plot_path)


def plot_strategy_overlap(results: Dict[str, Dict[str, Any]], output_dir: str) -> str:
    """Plot a heatmap showing the overlap between strategy predictions.

    Args:
        results: Dictionary mapping strategy names to their evaluation results
        output_dir: Directory to save the plot

    Returns:
        Path to the saved plot file
    """
    output_path = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_path / f"strategy_overlap_{timestamp}.png"

    strategies = list(results.keys())
    n_strategies = len(strategies)
    overlap_matrix = np.zeros((n_strategies, n_strategies))

    # Calculate overlap between strategies
    for i, strat1 in enumerate(strategies):
        for j, strat2 in enumerate(strategies):
            if i == j:
                overlap_matrix[i, j] = 1.0
                continue

            # Calculate average overlap in predictions
            total_overlap = 0
            num_predictions = len(results[strat1]["detailed_results"])

            for k in range(num_predictions):
                pred1 = set(results[strat1]["detailed_results"][k]["prediction"])
                pred2 = set(results[strat2]["detailed_results"][k]["prediction"])
                overlap = len(pred1 & pred2) / len(pred1)
                total_overlap += overlap

            overlap_matrix[i, j] = total_overlap / num_predictions

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        overlap_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=strategies,
        yticklabels=strategies,
    )

    plt.title("Strategy Prediction Overlap")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved strategy overlap heatmap to: {plot_path}")
    return str(plot_path)


def plot_confidence_vs_hits(results: Dict[str, Dict[str, Any]], output_dir: str) -> str:
    """Plot confidence scores against actual hits for each strategy.

    Args:
        results: Dictionary mapping strategy names to their evaluation results
        output_dir: Directory to save the plot

    Returns:
        Path to the saved plot file
    """
    output_path = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_path / f"confidence_vs_hits_{timestamp}.png"

    # Set style
    plt.style.use("seaborn")
    plt.figure(figsize=(12, 6))

    # Plot scatter points for each strategy
    for strategy, res in results.items():
        hits = [r["hits"] for r in res["detailed_results"]]
        confidences = [r["confidence"] for r in res["detailed_results"]]

        # Add jitter to hits for better visualization
        jittered_hits = np.array(hits) + np.random.normal(0, 0.1, len(hits))

        plt.scatter(
            confidences,
            jittered_hits,
            alpha=0.5,
            label=strategy,
            color=STRATEGY_COLORS.get(strategy, "gray"),
        )

    plt.title("Confidence vs Actual Hits")
    plt.xlabel("Prediction Confidence")
    plt.ylabel("Number of Hits")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add diagonal line for reference
    plt.plot([0, 1], [0, 20], "k--", alpha=0.3, label="Perfect Calibration")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved confidence vs hits plot to: {plot_path}")
    return str(plot_path)


def generate_all_visualizations(
    results: Dict[str, Dict[str, Any]], output_dir: str
) -> Dict[str, str]:
    """Generate all advanced visualizations for strategy comparison.

    Args:
        results: Dictionary mapping strategy names to their evaluation results
        output_dir: Directory to save the plots

    Returns:
        Dictionary mapping visualization names to their file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate all visualizations
    visualization_paths = {
        "time_series": plot_time_series(results, output_dir),
        "hit_distribution": plot_hit_distribution(results, output_dir),
        "strategy_overlap": plot_strategy_overlap(results, output_dir),
        "confidence_vs_hits": plot_confidence_vs_hits(results, output_dir),
    }

    return visualization_paths
