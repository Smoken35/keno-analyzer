#!/usr/bin/env python3
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from keno.analysis.combination_analyzer import CombinationAnalyzer
from keno.analysis.gap_analyzer import KenoGapAnalyzer


class SuperEnsemblePredictor:
    def __init__(self):
        self.setup_logging()
        self.setup_output_dir()
        self.combination_analyzer = CombinationAnalyzer()
        self.gap_analyzer = KenoGapAnalyzer()

    def setup_logging(self):
        """Set up logging configuration."""
        log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "keno_data")
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(log_dir, "super_ensemble.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def setup_output_dir(self):
        """Set up directory for analysis outputs."""
        self.output_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "keno_data", "analysis"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def get_traditional_predictions(self, pick_size: int = 10) -> Dict[str, List[int]]:
        """Get predictions from traditional methods."""
        try:
            predictions = {}

            # Get predictions for different combination sizes
            for size in [3, 4, 5]:
                combos = self.combination_analyzer.analyze_combinations(size)
                if combos:
                    predictions[f"combination_{size}"] = combos[0]["combination"]

            # Get predictions from number relationships
            relationships = self.combination_analyzer.analyze_number_relationships()
            if relationships:
                relationship_numbers = []
                for rel in relationships[:5]:
                    relationship_numbers.extend([rel["number1"], rel["number2"]])
                predictions["relationships"] = relationship_numbers

            # Get predictions from winning patterns
            patterns = self.combination_analyzer.find_winning_patterns()
            if patterns:
                pattern_numbers = []
                for pattern in patterns[:3]:
                    pattern_numbers.extend(pattern["combination"])
                predictions["patterns"] = pattern_numbers

            return predictions

        except Exception as e:
            logging.error(f"Error getting traditional predictions: {str(e)}")
            return {}

    def get_gap_predictions(self, pick_size: int = 10) -> Dict[str, List[int]]:
        """Get predictions from gap-based methods."""
        try:
            predictions = {}

            # Get predictions from different gap methods
            for method in ["gap", "pattern_gap", "combined"]:
                predictions[method] = self.gap_analyzer.predict_numbers(method, pick_size)

            # Get overdue numbers
            gap_stats = self.gap_analyzer.analyze_number_gaps()
            if gap_stats:
                overdue_numbers = [
                    num
                    for num, stats in sorted(
                        gap_stats.items(), key=lambda x: x[1]["overdue_factor"], reverse=True
                    )[:pick_size]
                ]
                predictions["overdue"] = overdue_numbers

            # Get overdue pairs
            pair_stats = self.gap_analyzer.analyze_pair_gaps()
            if pair_stats:
                pair_numbers = []
                for pair, stats in sorted(
                    pair_stats.items(), key=lambda x: x[1]["overdue_factor"], reverse=True
                )[:5]:
                    pair_numbers.extend(pair)
                predictions["overdue_pairs"] = pair_numbers

            return predictions

        except Exception as e:
            logging.error(f"Error getting gap predictions: {str(e)}")
            return {}

    def get_confidence_levels(self, predictions: Dict[str, List[int]]) -> Dict[str, float]:
        """Calculate confidence levels for each prediction method."""
        try:
            confidence = {}

            # Calculate confidence based on historical performance
            for method, numbers in predictions.items():
                if not numbers:
                    continue

                # Get historical accuracy for this method
                accuracy = self._calculate_historical_accuracy(method, numbers)

                # Adjust confidence based on method type
                if method.startswith("gap"):
                    # Gap methods tend to be more reliable for overdue numbers
                    confidence[method] = accuracy * 1.2
                elif method.startswith("combination"):
                    # Combination methods are more reliable for frequent patterns
                    confidence[method] = accuracy * 1.1
                else:
                    confidence[method] = accuracy

            return confidence

        except Exception as e:
            logging.error(f"Error calculating confidence levels: {str(e)}")
            return {}

    def _calculate_historical_accuracy(self, method: str, numbers: List[int]) -> float:
        """Calculate historical accuracy for a prediction method."""
        try:
            # This is a simplified version - in practice, you would want to:
            # 1. Use a larger historical dataset
            # 2. Consider different time periods
            # 3. Account for partial matches
            # 4. Consider the recency of successful predictions

            if not numbers:
                return 0.0

            # Get recent draws
            recent_draws = self.combination_analyzer.data.tail(100)
            if recent_draws.empty:
                return 0.0

            # Count matches
            matches = 0
            for _, row in recent_draws.iterrows():
                draw_numbers = set(row["winning_numbers"])
                matched_numbers = len(set(numbers) & draw_numbers)
                if matched_numbers >= len(numbers) * 0.7:  # 70% match threshold
                    matches += 1

            return matches / len(recent_draws)

        except Exception as e:
            logging.error(f"Error calculating historical accuracy: {str(e)}")
            return 0.0

    def generate_predictions(self, pick_size: int = 10) -> Dict[str, Any]:
        """Generate predictions using the super-ensemble approach."""
        try:
            # Get predictions from both methods
            traditional_predictions = self.get_traditional_predictions(pick_size)
            gap_predictions = self.get_gap_predictions(pick_size)

            # Combine all predictions
            all_predictions = {**traditional_predictions, **gap_predictions}

            # Calculate confidence levels
            confidence = self.get_confidence_levels(all_predictions)

            # Combine predictions with weighted voting
            votes = defaultdict(float)
            for method, numbers in all_predictions.items():
                weight = confidence.get(method, 1.0)
                for num in numbers:
                    votes[num] += weight

            # Get final predictions
            final_predictions = [
                num
                for num, _ in sorted(votes.items(), key=lambda x: x[1], reverse=True)[:pick_size]
            ]

            # Generate report
            report = {
                "predictions": final_predictions,
                "confidence": confidence,
                "method_predictions": all_predictions,
                "vote_counts": dict(votes),
            }

            return report

        except Exception as e:
            logging.error(f"Error generating predictions: {str(e)}")
            return {}

    def plot_prediction_distribution(self, report: Dict[str, Any]):
        """Plot the distribution of predictions and confidence levels."""
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle("Super-Ensemble Prediction Analysis")

            # Plot 1: Vote Distribution
            votes = report["vote_counts"]
            axes[0, 0].bar(votes.keys(), votes.values())
            axes[0, 0].set_title("Vote Distribution")
            axes[0, 0].set_xlabel("Number")
            axes[0, 0].set_ylabel("Weighted Votes")

            # Plot 2: Confidence Levels
            confidence = report["confidence"]
            axes[0, 1].bar(confidence.keys(), confidence.values())
            axes[0, 1].set_title("Method Confidence Levels")
            axes[0, 1].set_xlabel("Method")
            axes[0, 1].set_ylabel("Confidence")

            # Plot 3: Method Agreement
            method_agreement = defaultdict(int)
            for method, numbers in report["method_predictions"].items():
                for num in numbers:
                    method_agreement[num] += 1
            axes[1, 0].bar(method_agreement.keys(), method_agreement.values())
            axes[1, 0].set_title("Method Agreement")
            axes[1, 0].set_xlabel("Number")
            axes[1, 0].set_ylabel("Number of Methods Agreeing")

            # Plot 4: Final Predictions
            final_predictions = report["predictions"]
            axes[1, 1].bar(range(len(final_predictions)), [votes[num] for num in final_predictions])
            axes[1, 1].set_title("Final Predictions")
            axes[1, 1].set_xlabel("Prediction Rank")
            axes[1, 1].set_ylabel("Weighted Votes")

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "super_ensemble_analysis.png"))
            plt.close()

        except Exception as e:
            logging.error(f"Error plotting prediction distribution: {str(e)}")

    def print_analysis(self):
        """Print the super-ensemble analysis results."""
        print(f"\n{'='*20} Super-Ensemble Prediction Analysis {'='*20}")

        # Generate predictions
        report = self.generate_predictions()

        if not report:
            print("No predictions available. Please check the data and try again.")
            return

        # Generate visualizations
        self.plot_prediction_distribution(report)
        print(f"\nVisualizations saved in: {self.output_dir}")

        print("\nFinal Predictions:")
        for i, num in enumerate(report["predictions"], 1):
            print(f"{i}. Number: {num} (Votes: {report['vote_counts'][num]:.2f})")

        print("\nMethod Confidence Levels:")
        for method, confidence in report["confidence"].items():
            print(f"{method}: {confidence:.2%}")

        print("\nMethod Predictions:")
        for method, numbers in report["method_predictions"].items():
            print(f"\n{method}:")
            print(f"   Numbers: {', '.join(map(str, numbers))}")

        print(f"\n{'='*60}\n")


def main():
    predictor = SuperEnsemblePredictor()
    predictor.print_analysis()


if __name__ == "__main__":
    main()
