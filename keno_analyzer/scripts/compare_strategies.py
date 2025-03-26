"""
Script for comparing different Keno prediction strategies.
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from keno_analyzer.prediction.prediction_engine import PredictionEngine
from keno_analyzer.scripts.advanced_visualizations import generate_all_visualizations
from keno_analyzer.scripts.interactive_report import generate_interactive_report

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_all_strategies(
    data: List[List[int]], start_index: int = 50, end_index: int = None
) -> Dict[str, Dict[str, Any]]:
    """Evaluate all available strategies on historical data.

    Args:
        data: List of historical Keno draws
        start_index: Starting index for evaluation
        end_index: Ending index for evaluation (exclusive)

    Returns:
        Dictionary mapping strategy names to their evaluation results
    """
    engine = PredictionEngine()
    strategy_names = [s["name"] for s in engine.get_available_strategies()]
    end_index = end_index or len(data)
    results = {}

    for strategy_name in tqdm(strategy_names, desc="Evaluating strategies"):
        logger.info(f"Evaluating: {strategy_name}")
        result = engine.evaluate_strategy(
            strategy_name, data, start_index=start_index, end_index=end_index
        )
        results[strategy_name] = result

        # Log summary metrics
        logger.info(f"Results for {strategy_name}:")
        logger.info(f"  Average hits: {result['average_hits']:.2f}")
        logger.info(f"  Average confidence: {result['average_confidence']:.3f}")
        logger.info(f"  Hit rate (10+ hits): {result['hit_rate']*100:.2f}%")

    return results


def plot_strategy_comparison(results: Dict[str, Dict[str, Any]], output_dir: str) -> str:
    """Generate comparison plots for all strategies.

    Args:
        results: Dictionary mapping strategy names to their evaluation results
        output_dir: Directory to save the plots

    Returns:
        Path to the saved plot file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    strategies = list(results.keys())
    avg_hits = [results[s]["average_hits"] for s in strategies]
    avg_conf = [results[s]["average_confidence"] for s in strategies]
    hit_rate = [results[s]["hit_rate"] * 100 for s in strategies]

    # Set style
    plt.style.use("seaborn")
    sns.set_palette("husl")

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))

    # 1. Average Hits
    plt.subplot(1, 3, 1)
    sns.barplot(x=strategies, y=avg_hits)
    plt.title("Average Hits per Draw")
    plt.ylim(0, 20)
    plt.xticks(rotation=45)

    # 2. Average Confidence
    plt.subplot(1, 3, 2)
    sns.barplot(x=strategies, y=avg_conf)
    plt.title("Average Confidence")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)

    # 3. Hit Rate
    plt.subplot(1, 3, 3)
    sns.barplot(x=strategies, y=hit_rate)
    plt.title("Hit Rate (10+ Hits)")
    plt.ylim(0, 100)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plot_path = output_path / f"strategy_comparison_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved comparison plots to: {plot_path}")
    return str(plot_path)


def save_comparison_csv(results: Dict[str, Dict[str, Any]], output_dir: str) -> str:
    """Save comparison results to CSV.

    Args:
        results: Dictionary mapping strategy names to their evaluation results
        output_dir: Directory to save the CSV file

    Returns:
        Path to the saved CSV file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_path / f"strategy_comparison_summary_{timestamp}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Strategy",
                "Total Predictions",
                "Average Hits",
                "Average Confidence",
                "Hit Rate (%)",
                "Best Performance",
                "Worst Performance",
            ]
        )

        for name, res in results.items():
            hits = [r["hits"] for r in res["detailed_results"]]
            writer.writerow(
                [
                    name,
                    res["total_predictions"],
                    f"{res['average_hits']:.2f}",
                    f"{res['average_confidence']:.3f}",
                    f"{res['hit_rate'] * 100:.2f}",
                    f"{max(hits)} hits",
                    f"{min(hits)} hits",
                ]
            )

    logger.info(f"Saved comparison summary to: {csv_path}")
    return str(csv_path)


def main():
    """Main function to run the strategy comparison."""
    # Configuration
    input_file = "data/keno_draws.json"  # Replace with your actual draw file
    output_dir = "results/strategy_comparison"
    start_index = 1000  # Start evaluation from this index

    # Load data
    logger.info(f"Loading data from: {input_file}")
    with open(input_file) as f:
        data = json.load(f)

    if len(data) < start_index + 100:
        logger.error(f"Insufficient data: need at least {start_index + 100} draws")
        return

    # Evaluate strategies
    results = evaluate_all_strategies(data, start_index=start_index)

    # Generate basic comparison plot
    basic_plot_path = plot_strategy_comparison(results, output_dir)

    # Generate advanced visualizations
    advanced_plot_paths = generate_all_visualizations(results, output_dir)

    # Combine all plot paths
    all_plot_paths = {"basic_comparison": basic_plot_path, **advanced_plot_paths}

    # Save CSV summary
    csv_path = save_comparison_csv(results, output_dir)

    # Generate interactive HTML report
    html_path = generate_interactive_report(results, all_plot_paths, csv_path, output_dir)

    logger.info("\n✅ Strategy comparison complete!")
    logger.info(f"📊 Results saved to:")
    logger.info(f"  - Basic Comparison Plot: {basic_plot_path}")
    for name, path in advanced_plot_paths.items():
        logger.info(f"  - {name.replace('_', ' ').title()}: {path}")
    logger.info(f"  - CSV Summary: {csv_path}")
    logger.info(f"  - Interactive HTML Report: {html_path}")


if __name__ == "__main__":
    main()
