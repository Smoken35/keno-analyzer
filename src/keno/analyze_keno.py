"""
Keno data analysis script.
"""

import logging
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .data.fetcher import KenoDataFetcher
from .visualization.pattern_visualizer import PatternVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("keno_data/analysis.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def analyze_number_frequency(df: pd.DataFrame) -> Dict:
    """Analyze frequency of each number.

    Args:
        df: DataFrame containing Keno results

    Returns:
        Dictionary containing frequency analysis
    """
    # Get all numbers from all draws
    all_numbers = []
    for i in range(1, 21):
        all_numbers.extend(df[f"NUMBER DRAWN {i}"].tolist())

    # Count frequencies
    number_counts = Counter(all_numbers)
    total_draws = len(df)

    # Calculate percentages
    results = {}
    for num in range(1, 81):
        count = number_counts.get(num, 0)
        percentage = (count / total_draws) * 100
        expected_percentage = 1.25  # Each number should appear 1.25% of the time
        results[str(num)] = {
            "count": count,
            "percentage": percentage,
            "expected_percentage": expected_percentage,
            "deviation": percentage - expected_percentage,
        }

    return results


def analyze_pair_frequency(df: pd.DataFrame) -> Dict:
    """Analyze frequency of number pairs.

    Args:
        df: DataFrame containing Keno results

    Returns:
        Dictionary containing pair frequency analysis
    """
    pair_counts = Counter()
    total_draws = len(df)

    # Count pairs in each draw
    for _, row in df.iterrows():
        numbers = [row[f"NUMBER DRAWN {i}"] for i in range(1, 21)]
        for i, num1 in enumerate(numbers):
            for num2 in numbers[i + 1 :]:
                pair = tuple(sorted([num1, num2]))
                pair_counts[pair] += 1

    # Calculate frequencies and ratios
    results = {}
    expected_pair_freq = (20 * 19 / 2) / (80 * 79 / 2) * 100  # Expected pair frequency

    for pair, count in pair_counts.items():
        percentage = (count / total_draws) * 100
        ratio = percentage / expected_pair_freq
        results[str(pair)] = {
            "count": count,
            "percentage": percentage,
            "expected_percentage": expected_pair_freq,
            "ratio": ratio,
        }

    return results


def analyze_grid_patterns(df: pd.DataFrame) -> Dict:
    """Analyze patterns in the Keno grid.

    Args:
        df: DataFrame containing Keno results

    Returns:
        Dictionary containing grid pattern analysis
    """
    # Initialize counters
    row_counts = defaultdict(int)
    col_counts = defaultdict(int)
    quadrant_counts = defaultdict(int)
    total_draws = len(df)

    # Count occurrences in each position
    for _, row in df.iterrows():
        for i in range(1, 21):
            num = row[f"NUMBER DRAWN {i}"]
            row_idx = (num - 1) // 10
            col_idx = (num - 1) % 10
            quadrant_idx = (row_idx // 4) * 2 + (col_idx // 5)

            row_counts[row_idx] += 1
            col_counts[col_idx] += 1
            quadrant_counts[quadrant_idx] += 1

    # Calculate bias ratios
    expected_row_freq = 20 / 8  # Expected numbers per row
    expected_col_freq = 20 / 10  # Expected numbers per column
    expected_quadrant_freq = 20 / 4  # Expected numbers per quadrant

    results = {
        "row_bias": {
            str(row): {
                "count": count,
                "expected_count": expected_row_freq * total_draws,
                "bias_ratio": count / (expected_row_freq * total_draws),
            }
            for row, count in row_counts.items()
        },
        "col_bias": {
            str(col): {
                "count": count,
                "expected_count": expected_col_freq * total_draws,
                "bias_ratio": count / (expected_col_freq * total_draws),
            }
            for col, count in col_counts.items()
        },
        "quadrant_bias": {
            str(q): {
                "count": count,
                "expected_count": expected_quadrant_freq * total_draws,
                "bias_ratio": count / (expected_quadrant_freq * total_draws),
            }
            for q, count in quadrant_counts.items()
        },
    }

    return results


def analyze_hot_cold_cycles(df: pd.DataFrame, window_size: int = 10) -> List[Dict]:
    """Analyze hot and cold number cycles.

    Args:
        df: DataFrame containing Keno results
        window_size: Size of sliding window for analysis

    Returns:
        List of dictionaries containing hot-cold cycle analysis
    """
    results = []

    # Calculate frequency in each window
    for i in range(0, len(df) - window_size + 1):
        window = df.iloc[i : i + window_size]
        number_counts = Counter()

        # Count numbers in window
        for _, row in window.iterrows():
            for j in range(1, 21):
                number_counts[row[f"NUMBER DRAWN {j}"]] += 1

        # Identify hot and cold numbers
        hot_threshold = window_size * 0.4  # Appears in 40% of draws
        cold_threshold = window_size * 0.1  # Appears in 10% of draws

        hot_numbers = [num for num, count in number_counts.items() if count >= hot_threshold]
        cold_numbers = [num for num, count in number_counts.items() if count <= cold_threshold]

        results.append(
            {
                "window_start": i,
                "window_end": i + window_size - 1,
                "hot_count": len(hot_numbers),
                "cold_count": len(cold_numbers),
                "hot_numbers": hot_numbers,
                "cold_numbers": cold_numbers,
            }
        )

    return results


def analyze_sequential_patterns(df: pd.DataFrame) -> Dict:
    """Analyze patterns of numbers appearing in consecutive draws.

    Args:
        df: DataFrame containing Keno results

    Returns:
        Dictionary containing sequential pattern analysis
    """
    results = {}

    # Analyze each number
    for num in range(1, 81):
        consecutive_count = 0
        total_consecutive = 0
        total_draws = len(df)

        # Check each draw
        for i in range(1, total_draws):
            prev_draw = set(df.iloc[i - 1][[f"NUMBER DRAWN {j}" for j in range(1, 21)]])
            curr_draw = set(df.iloc[i][[f"NUMBER DRAWN {j}" for j in range(1, 21)]])

            if num in prev_draw and num in curr_draw:
                consecutive_count += 1
            total_consecutive += 1

        # Calculate probability
        follow_probability = consecutive_count / total_consecutive if total_consecutive > 0 else 0
        expected_probability = 0.25  # Expected probability of appearing in consecutive draws

        results[str(num)] = {
            "consecutive_count": consecutive_count,
            "total_consecutive": total_consecutive,
            "follow_probability": follow_probability,
            "expected_probability": expected_probability,
            "ratio": follow_probability / expected_probability if expected_probability > 0 else 0,
        }

    return results


def main():
    """Main function to run the analysis."""
    try:
        # Initialize data fetcher
        data_dir = "keno_data"
        os.makedirs(data_dir, exist_ok=True)
        fetcher = KenoDataFetcher(data_dir)

        # Update historical data
        logger.info("Updating historical data...")
        if not fetcher.update_historical_data():
            logger.error("Failed to update historical data")
            return

        # Get recent results
        logger.info("Fetching recent results...")
        df = fetcher.get_recent_results(days=100)
        if df.empty:
            logger.error("No data available for analysis")
            return

        # Perform analysis
        logger.info("Performing pattern analysis...")
        pattern_results = {
            "number_frequency": analyze_number_frequency(df),
            "pair_frequency": analyze_pair_frequency(df),
            "grid_patterns": analyze_grid_patterns(df),
            "hot_cold_cycles": analyze_hot_cold_cycles(df),
            "sequential_patterns": analyze_sequential_patterns(df),
        }

        # Initialize visualizer
        visualizer = PatternVisualizer(results_dir="backtest_results")

        # Generate visualizations
        logger.info("Generating visualizations...")
        visualizer.visualize_pattern_analysis(pattern_results)

        logger.info("Analysis completed successfully")

    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")


if __name__ == "__main__":
    main()
