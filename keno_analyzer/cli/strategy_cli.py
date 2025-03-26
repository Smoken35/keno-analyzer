#!/usr/bin/env python3
"""
CLI tool for testing and evaluating Keno prediction strategies.
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import seaborn as sns

from keno_analyzer.prediction.prediction_engine import PredictionEngine


def load_draws(file_path: str) -> List[List[int]]:
    """Load Keno draw data from a CSV or JSON file."""
    path = Path(file_path)
    if not path.exists():
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    if path.suffix == ".json":
        with open(path) as f:
            return json.load(f)
    elif path.suffix == ".csv":
        draws = []
        with open(path) as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    draw = [int(n.strip()) for n in row if n.strip().isdigit()]
                    if draw:
                        draws.append(draw)
                except ValueError:
                    continue
        return draws
    else:
        print("[ERROR] Unsupported file format. Use .json or .csv")
        sys.exit(1)


def save_evaluation_results(results: Dict[str, Any], output_dir: str) -> None:
    """Save evaluation results to JSON and CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = output_dir / f"evaluation_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Saved detailed results to: {json_path}")

    # Save CSV
    csv_path = output_dir / f"evaluation_results_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Draw Index", "Prediction", "Actual", "Hits", "Confidence"])
        for r in results["detailed_results"]:
            writer.writerow(
                [
                    r["draw_index"],
                    ",".join(map(str, r["prediction"])),
                    ",".join(map(str, r["actual"])),
                    r["hits"],
                    f"{r['confidence']:.3f}",
                ]
            )
    print(f"[INFO] Saved CSV results to: {csv_path}")


def plot_evaluation_results(results: Dict[str, Any], output_dir: str) -> None:
    """Generate visualization plots for evaluation results."""
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set style
    plt.style.use("seaborn")
    sns.set_palette("husl")

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)

    # 1. Hit Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    hits = [r["hits"] for r in results["detailed_results"]]
    sns.histplot(hits, bins=20, ax=ax1)
    ax1.set_title("Hit Distribution")
    ax1.set_xlabel("Number of Hits")
    ax1.set_ylabel("Count")

    # 2. Confidence Trend
    ax2 = fig.add_subplot(gs[0, 1])
    confidences = [r["confidence"] for r in results["detailed_results"]]
    ax2.plot(confidences)
    ax2.set_title("Confidence Trend")
    ax2.set_xlabel("Draw Index")
    ax2.set_ylabel("Confidence")

    # 3. Hit Rate Over Time
    ax3 = fig.add_subplot(gs[1, 0])
    window_size = 50
    hit_rates = []
    for i in range(0, len(results["detailed_results"]), window_size):
        window = results["detailed_results"][i : i + window_size]
        hit_rate = sum(1 for r in window if r["hits"] >= 10) / len(window)
        hit_rates.append(hit_rate)
    ax3.plot(hit_rates)
    ax3.set_title(f"Hit Rate (10+ hits) Over Time (Window: {window_size})")
    ax3.set_xlabel("Window Index")
    ax3.set_ylabel("Hit Rate")

    # 4. Hits vs Confidence
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(hits, confidences, alpha=0.5)
    ax4.set_title("Hits vs Confidence")
    ax4.set_xlabel("Number of Hits")
    ax4.set_ylabel("Confidence")

    # Save plot
    plt.tight_layout()
    plot_path = output_dir / f"evaluation_plots_{timestamp}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"[INFO] Saved visualization plots to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Keno Prediction Strategy CLI")
    parser.add_argument("--file", required=True, help="Path to historical draw data (JSON or CSV)")
    parser.add_argument(
        "--strategy",
        choices=["pattern", "rule", "cluster"],
        default="pattern",
        help="Strategy to use",
    )
    parser.add_argument(
        "--draw-index", type=int, help="Draw index to predict (must be < len(data))"
    )
    parser.add_argument(
        "--num-picks", type=int, default=20, help="Number of numbers to predict (default: 20)"
    )
    parser.add_argument("--evaluate", action="store_true", help="Evaluate strategy over full range")
    parser.add_argument("--start-index", type=int, default=50, help="Start index for evaluation")
    parser.add_argument("--summary", action="store_true", help="Show summary metrics")
    parser.add_argument("--details", action="store_true", help="Show detailed predictions")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization plots")
    parser.add_argument("--export", action="store_true", help="Export results to JSON/CSV")
    parser.add_argument(
        "--output-dir", default="evaluation_results", help="Directory for output files"
    )

    args = parser.parse_args()
    data = load_draws(args.file)

    if len(data) < 100:
        print(f"[ERROR] Need at least 100 draws to use the strategies (found: {len(data)})")
        sys.exit(1)

    engine = PredictionEngine()
    strategy_map = {"pattern": "PatternBased", "rule": "RuleBased", "cluster": "ClusterBased"}
    strategy_name = strategy_map[args.strategy]
    engine.set_active_strategy(strategy_name)

    if args.evaluate:
        end_index = len(data)
        print(f"\n[Evaluating Strategy: {strategy_name}]")
        result = engine.evaluate_strategy(
            strategy_name, data, start_index=args.start_index, end_index=end_index
        )

        if args.summary:
            print("\n[Evaluation Summary]")
            print(f"Strategy: {strategy_name}")
            print(f"Draws evaluated: {result['total_predictions']}")
            print(f"Average Hits: {result['average_hits']:.2f}")
            print(f"Average Confidence: {result['average_confidence']:.3f}")
            print(f"Hit Rate (10+ hits): {result['hit_rate']*100:.2f}%")

        if args.details:
            print("\n[Detailed Results]")
            for r in result["detailed_results"]:
                print(
                    f"Draw {r['draw_index']} | Hits: {r['hits']} | Confidence: {r['confidence']:.3f}"
                )
                print(f"Prediction: {r['prediction']}")
                print(f"Actual:     {r['actual']}")
                print("-" * 60)

        if args.export:
            save_evaluation_results(result, args.output_dir)

        if args.visualize:
            plot_evaluation_results(result, args.output_dir)

    elif args.draw_index is not None:
        engine.fit(data[: args.draw_index])
        result = engine.predict(args.draw_index, num_picks=args.num_picks)

        actual = data[args.draw_index]
        hits = len(set(result["prediction"]) & set(actual))

        print(f"\n[Prediction for Draw Index {args.draw_index}]")
        print(f"Strategy: {result['strategy_info']['name']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Actual:     {actual}")
        print(f"Hits:       {hits}")
        print(f"Confidence: {result['confidence']:.3f}")

    else:
        print("[ERROR] You must specify either --draw-index or --evaluate.")
        sys.exit(1)


if __name__ == "__main__":
    main()
