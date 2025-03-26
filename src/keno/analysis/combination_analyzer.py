#!/usr/bin/env python3
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class CombinationAnalyzer:
    def __init__(self):
        self.setup_logging()
        self.data = self.load_data()
        self.combination_stats = {}
        self.setup_output_dir()

    def setup_logging(self):
        """Set up logging configuration."""
        log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "keno_data")
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(log_dir, "combination_analysis.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def setup_output_dir(self):
        """Set up directory for analysis outputs."""
        self.output_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "keno_data", "analysis"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load and prepare the historical data."""
        try:
            base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "keno_data")
            all_time_file = os.path.join(base_dir, "keno_results_all.csv")

            if not os.path.exists(all_time_file) or os.path.getsize(all_time_file) == 0:
                print("No historical data found. Please process some data first.")
                return pd.DataFrame()

            df = pd.read_csv(all_time_file)
            if isinstance(df["winning_numbers"].iloc[0], str):
                df["winning_numbers"] = df["winning_numbers"].apply(lambda x: eval(x))
            df["draw_time"] = pd.to_datetime(df["draw_time"])
            return df
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            print(f"Error loading data: {str(e)}")
            return pd.DataFrame()

    def analyze_combinations(self, size: int = 3) -> List[Dict[str, Any]]:
        """Analyze all possible combinations of a given size."""
        try:
            # Get all numbers that have appeared in the last 100 draws
            recent_numbers = set()
            for numbers in self.data.tail(100)["winning_numbers"]:
                recent_numbers.update(numbers)

            # Generate all possible combinations
            all_combinations = list(combinations(sorted(recent_numbers), size))

            # Analyze each combination
            combination_stats = []
            for combo in all_combinations:
                stats = self._analyze_single_combination(combo)
                if stats["frequency"] > 0:  # Only include combinations that have appeared
                    combination_stats.append(stats)

            # Sort by frequency and return top combinations
            return sorted(combination_stats, key=lambda x: x["frequency"], reverse=True)

        except Exception as e:
            logging.error(f"Error analyzing combinations: {str(e)}")
            return []

    def _analyze_single_combination(self, combo: Tuple[int, ...]) -> Dict[str, Any]:
        """Analyze a single combination of numbers."""
        combo_set = set(combo)
        total_draws = len(self.data)
        matches = []

        # Find all draws where this combination appears
        for _, row in self.data.iterrows():
            draw_numbers = set(row["winning_numbers"])
            if combo_set.issubset(draw_numbers):
                matches.append(
                    {
                        "draw_number": row["draw_number"],
                        "draw_time": row["draw_time"],
                        "additional_numbers": list(draw_numbers - combo_set),
                    }
                )

        # Calculate statistics
        frequency = (len(matches) / total_draws) * 100

        # Calculate average additional numbers
        avg_additional = np.mean([len(m["additional_numbers"]) for m in matches]) if matches else 0

        # Calculate recent performance
        recent_matches = [m for m in matches if self._is_recent(m["draw_time"])]
        recent_frequency = (len(recent_matches) / min(100, total_draws)) * 100

        # Calculate trend
        if len(matches) >= 2:
            first_seen = pd.to_datetime(matches[0]["draw_time"])
            last_seen = pd.to_datetime(matches[-1]["draw_time"])
            days_between = (last_seen - first_seen).days
            trend = len(matches) / max(1, days_between) * 30  # Appearances per month
        else:
            trend = 0

        return {
            "combination": list(combo),
            "frequency": frequency,
            "recent_frequency": recent_frequency,
            "total_matches": len(matches),
            "recent_matches": len(recent_matches),
            "avg_additional_numbers": avg_additional,
            "last_seen": matches[-1]["draw_time"] if matches else None,
            "trend": trend,
            "matches": matches,
        }

    def _is_recent(self, draw_time) -> bool:
        """Check if a draw time is recent (within last 30 days)."""
        try:
            if isinstance(draw_time, str):
                draw_date = pd.to_datetime(draw_time)
            else:
                draw_date = draw_time
            thirty_days_ago = datetime.now() - timedelta(days=30)
            return draw_date >= thirty_days_ago
        except:
            return False

    def analyze_number_relationships(self) -> Dict[str, Any]:
        """Analyze relationships between numbers."""
        try:
            relationships = defaultdict(
                lambda: {"appears_with": defaultdict(int), "total_appearances": 0}
            )

            # Count co-occurrences
            for numbers in self.data["winning_numbers"]:
                for i, num1 in enumerate(numbers):
                    relationships[num1]["total_appearances"] += 1
                    for num2 in numbers[i + 1 :]:
                        relationships[num1]["appears_with"][num2] += 1
                        relationships[num2]["appears_with"][num1] += 1

            # Convert to list and sort by strength
            relationship_list = []
            for num1, data in relationships.items():
                for num2, count in data["appears_with"].items():
                    strength = (count / data["total_appearances"]) * 100
                    relationship_list.append(
                        {
                            "number1": num1,
                            "number2": num2,
                            "strength": strength,
                            "co_occurrences": count,
                            "total_appearances": data["total_appearances"],
                        }
                    )

            return sorted(relationship_list, key=lambda x: x["strength"], reverse=True)

        except Exception as e:
            logging.error(f"Error analyzing number relationships: {str(e)}")
            return []

    def find_winning_patterns(self, min_size: int = 3, max_size: int = 5) -> List[Dict[str, Any]]:
        """Find patterns that have historically led to wins."""
        try:
            patterns = []

            # Analyze patterns of different sizes
            for size in range(min_size, max_size + 1):
                combinations = self.analyze_combinations(size)
                for combo in combinations:
                    if combo["frequency"] > 0:
                        patterns.append(
                            {
                                "size": size,
                                "combination": combo["combination"],
                                "frequency": combo["frequency"],
                                "recent_frequency": combo["recent_frequency"],
                                "avg_additional_numbers": combo["avg_additional_numbers"],
                                "last_seen": combo["last_seen"],
                                "trend": combo.get("trend", 0),
                            }
                        )

            # Sort by weighted score (frequency, recent performance, and trend)
            return sorted(
                patterns,
                key=lambda x: (
                    x["frequency"] * 0.4 + x["recent_frequency"] * 0.4 + x["trend"] * 0.2
                ),
                reverse=True,
            )

        except Exception as e:
            logging.error(f"Error finding winning patterns: {str(e)}")
            return []

    def plot_number_frequency(self):
        """Plot the frequency of each number."""
        try:
            # Count frequency of each number
            number_counts = defaultdict(int)
            total_draws = len(self.data)

            for numbers in self.data["winning_numbers"]:
                for num in numbers:
                    number_counts[num] += 1

            # Convert to percentage
            frequencies = {num: (count / total_draws) * 100 for num, count in number_counts.items()}

            # Create plot
            plt.figure(figsize=(15, 6))
            plt.bar(frequencies.keys(), frequencies.values())
            plt.title("Number Frequency Distribution")
            plt.xlabel("Number")
            plt.ylabel("Frequency (%)")
            plt.grid(True, alpha=0.3)

            # Save plot
            plt.savefig(os.path.join(self.output_dir, "number_frequency.png"))
            plt.close()

        except Exception as e:
            logging.error(f"Error plotting number frequency: {str(e)}")

    def plot_relationship_heatmap(self):
        """Plot a heatmap of number relationships."""
        try:
            # Get relationships
            relationships = self.analyze_number_relationships()

            # Create matrix
            numbers = sorted(
                list(
                    set(
                        [r["number1"] for r in relationships]
                        + [r["number2"] for r in relationships]
                    )
                )
            )
            matrix = np.zeros((len(numbers), len(numbers)))

            # Fill matrix
            for rel in relationships:
                i = numbers.index(rel["number1"])
                j = numbers.index(rel["number2"])
                matrix[i][j] = rel["strength"]
                matrix[j][i] = rel["strength"]  # Mirror the matrix

            # Create heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(matrix, xticklabels=numbers, yticklabels=numbers, cmap="YlOrRd")
            plt.title("Number Relationship Strength")

            # Save plot
            plt.savefig(os.path.join(self.output_dir, "relationship_heatmap.png"))
            plt.close()

        except Exception as e:
            logging.error(f"Error plotting relationship heatmap: {str(e)}")

    def print_analysis(self, combination_size: int = 3):
        """Print the analysis results."""
        print(f"\n{'='*20} Keno Combination Analysis {'='*20}")
        print(f"Analyzing combinations of size {combination_size}")
        print(f"Total historical draws analyzed: {len(self.data)}")

        # Generate visualizations
        self.plot_number_frequency()
        self.plot_relationship_heatmap()
        print(f"\nVisualizations saved in: {self.output_dir}")

        print("\nTop Combinations:")
        top_combinations = self.analyze_combinations(combination_size)[:10]
        for i, combo in enumerate(top_combinations, 1):
            print(f"\n{i}. Combination: {combo['combination']}")
            print(f"   Frequency: {combo['frequency']:.1f}%")
            print(f"   Recent Frequency: {combo['recent_frequency']:.1f}%")
            print(f"   Total Matches: {combo['total_matches']}")
            print(f"   Recent Matches: {combo['recent_matches']}")
            print(f"   Average Additional Numbers: {combo['avg_additional_numbers']:.1f}")
            print(f"   Last Seen: {combo['last_seen']}")
            print(f"   Monthly Trend: {combo['trend']:.2f} appearances/month")

        print("\nStrong Number Relationships:")
        relationships = self.analyze_number_relationships()[:10]
        for i, rel in enumerate(relationships, 1):
            print(f"\n{i}. {rel['number1']} ↔ {rel['number2']}")
            print(f"   Strength: {rel['strength']:.1f}%")
            print(f"   Co-occurrences: {rel['co_occurrences']}")
            print(f"   Total Appearances: {rel['total_appearances']}")

        print("\nWinning Patterns:")
        patterns = self.find_winning_patterns()[:10]
        for i, pattern in enumerate(patterns, 1):
            print(f"\n{i}. Size {pattern['size']}: {pattern['combination']}")
            print(f"   Frequency: {pattern['frequency']:.1f}%")
            print(f"   Recent Frequency: {pattern['recent_frequency']:.1f}%")
            print(f"   Average Additional Numbers: {pattern['avg_additional_numbers']:.1f}")
            print(f"   Last Seen: {pattern['last_seen']}")
            print(f"   Monthly Trend: {pattern['trend']:.2f} appearances/month")

        print("\nRecommendations:")
        print("1. Best Combinations by Size:")
        for size in [3, 4, 5]:
            combos = self.analyze_combinations(size)
            if combos:
                best = combos[0]
                print(
                    f"   Size {size}: {best['combination']} (Frequency: {best['frequency']:.1f}%)"
                )

        print("\n2. Hot Numbers (High Recent Frequency):")
        number_counts = defaultdict(int)
        recent_draws = self.data.tail(min(100, len(self.data)))
        for numbers in recent_draws["winning_numbers"]:
            for num in numbers:
                number_counts[num] += 1
        hot_numbers = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"   {', '.join([str(n[0]) for n in hot_numbers])}")

        print("\n3. Best Pattern Strategy:")
        if patterns:
            best_pattern = patterns[0]
            print(f"   Use pattern of size {best_pattern['size']}: {best_pattern['combination']}")
            print(f"   Build around these numbers for highest probability")

        print(f"\n{'='*60}\n")

    def binomial_probability(self, n: int, k: int, p: float) -> float:
        """
        Calculate the probability of getting k successes in n trials.

        Args:
            n: Number of trials
            k: Number of successes
            p: Probability of success on each trial

        Returns:
            Probability of k successes
        """
        return stats.binom.pmf(k, n, p)


def main():
    analyzer = CombinationAnalyzer()

    if analyzer.data.empty:
        print("No data available. Please process some data first.")
        return

    # Analyze different combination sizes
    for size in [3, 4, 5]:
        analyzer.print_analysis(size)


if __name__ == "__main__":
    main()
