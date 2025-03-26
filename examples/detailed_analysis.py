"""
Comprehensive Keno data analysis script with visualizations.
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from keno.analysis.analyzer import KenoAnalyzer


def plot_frequency_heatmap(frequencies, save_path):
    """Plot frequency heatmap of numbers."""
    numbers = np.zeros((8, 10))
    for num, freq in frequencies.items():
        row = (num - 1) // 10
        col = (num - 1) % 10
        numbers[row][col] = freq

    plt.figure(figsize=(15, 10))
    sns.heatmap(numbers, annot=True, fmt=".0f", cmap="YlOrRd")
    plt.title("Keno Number Frequency Heatmap")
    plt.savefig(save_path)
    plt.close()


def plot_cyclic_patterns(cycles, save_path):
    """Plot cyclic pattern probabilities."""
    plt.figure(figsize=(12, 6))
    plt.bar(cycles.keys(), cycles.values())
    plt.title("Cyclic Pattern Analysis")
    plt.xlabel("Cycle Length")
    plt.ylabel("Probability")
    plt.savefig(save_path)
    plt.close()


def evaluate_prediction_methods(analyzer, pick_sizes=[5, 10, 15, 20], num_simulations=1000):
    """Evaluate different prediction methods."""
    methods = ["frequency", "patterns", "markov", "due"]
    results = []

    # Set up standard payout table
    payout_table = {
        5: {5: 500, 4: 15, 3: 2, 2: 0, 1: 0, 0: 0},
        10: {10: 2000, 9: 200, 8: 50, 7: 10, 6: 2, 5: 1, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0},
        15: {
            15: 10000,
            14: 1500,
            13: 500,
            12: 100,
            11: 25,
            10: 10,
            9: 5,
            8: 2,
            7: 1,
            6: 0,
            5: 0,
            4: 0,
            3: 0,
            2: 0,
            1: 0,
            0: 0,
        },
        20: {
            20: 100000,
            19: 10000,
            18: 2500,
            17: 500,
            16: 100,
            15: 50,
            14: 20,
            13: 10,
            12: 5,
            11: 2,
            10: 1,
            9: 0,
            8: 0,
            7: 0,
            6: 0,
            5: 0,
            4: 0,
            3: 0,
            2: 0,
            1: 0,
            0: 0,
        },
    }
    analyzer.set_payout_table(payout_table)

    for pick_size in pick_sizes:
        for method in methods:
            sim_result = analyzer.simulate_strategy(
                method, pick_size, num_simulations=num_simulations
            )
            results.append(
                {
                    "method": method,
                    "pick_size": pick_size,
                    "roi": sim_result["roi_percent"],
                    "total_return": sim_result["total_return"],
                    "matches_dist": sim_result["match_distribution"],
                }
            )

    return pd.DataFrame(results)


def main():
    # Create output directory for plots
    output_dir = "analysis_output"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize analyzer and load data
    analyzer = KenoAnalyzer("historical")
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "KenoPastYears")
    analyzer.load_csv_files(data_dir)

    # 1. Frequency Analysis
    print("\n=== Frequency Analysis ===")
    frequency = analyzer.analyze_frequency()
    plot_frequency_heatmap(frequency, os.path.join(output_dir, "frequency_heatmap.png"))

    # Print top and bottom 10 numbers
    print("\nMost Common Numbers:")
    for num, freq in sorted(frequency.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"Number {num:2d}: {freq:6d} times ({freq/len(analyzer.data)*100:5.2f}%)")

    print("\nLeast Common Numbers:")
    for num, freq in sorted(frequency.items(), key=lambda x: x[1])[:10]:
        print(f"Number {num:2d}: {freq:6d} times ({freq/len(analyzer.data)*100:5.2f}%)")

    # 2. Pattern Analysis
    print("\n=== Pattern Analysis ===")
    windows = [10, 30, 100]
    for window in windows:
        patterns = analyzer.analyze_patterns(window=window, pick_size=20)
        print(f"\nLast {window} draws:")
        print(f"Hot numbers: {patterns['hot_numbers']}")
        print(f"Cold numbers: {patterns['cold_numbers']}")

    # 3. Cyclic Patterns
    print("\n=== Cyclic Pattern Analysis ===")
    cycles = analyzer.analyze_cyclic_patterns()
    plot_cyclic_patterns(cycles, os.path.join(output_dir, "cyclic_patterns.png"))

    # 4. Strategy Evaluation
    print("\n=== Strategy Evaluation ===")
    results_df = evaluate_prediction_methods(analyzer)

    # Plot strategy comparison
    plt.figure(figsize=(12, 6))
    for pick_size in results_df["pick_size"].unique():
        pick_data = results_df[results_df["pick_size"] == pick_size]
        plt.plot(pick_data["method"], pick_data["roi"], marker="o", label=f"Pick {pick_size}")

    plt.title("ROI by Prediction Method and Pick Size")
    plt.xlabel("Method")
    plt.ylabel("ROI (%)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "strategy_comparison.png"))
    plt.close()

    # Print strategy evaluation results
    print("\nStrategy Evaluation Results:")
    print(results_df.to_string(index=False))

    # 5. Due Numbers Analysis
    print("\n=== Due Numbers Analysis ===")
    due_numbers = analyzer.analyze_due_numbers()
    print("\nTop 20 Due Numbers:")
    for num, score in due_numbers[:20]:
        print(f"Number {num:2d}: Due score {score:.4f}")

    # 6. Make Predictions
    print("\n=== Predictions for Next Draw ===")
    methods = ["frequency", "patterns", "markov", "due"]
    pick_sizes = [5, 10, 15, 20]

    for method in methods:
        print(f"\n{method.capitalize()} method predictions:")
        for size in pick_sizes:
            prediction = analyzer.predict_next_draw(method, size)
            print(f"Pick {size}: {prediction}")


if __name__ == "__main__":
    main()
