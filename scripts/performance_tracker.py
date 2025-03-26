"""
Performance tracking and analysis for Keno predictions.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from keno.analysis.analyzer import KenoAnalyzer
from keno.data.fetcher import KenoDataFetcher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("performance_tracking.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class PerformanceTracker:
    def __init__(self, data_dir: str):
        """Initialize the performance tracker.

        Args:
            data_dir: Directory containing Keno data files
        """
        self.data_dir = data_dir
        self.performance_file = os.path.join(data_dir, "performance_history.json")
        self.analyzer = KenoAnalyzer("historical")
        self.fetcher = KenoDataFetcher(data_dir)

        # Load existing performance data
        self.performance_history = self._load_performance_history()

    def _load_performance_history(self) -> List[Dict]:
        """Load existing performance history."""
        if os.path.exists(self.performance_file):
            try:
                with open(self.performance_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading performance history: {e}")
                return []
        return []

    def _save_performance_history(self):
        """Save performance history to file."""
        try:
            with open(self.performance_file, "w") as f:
                json.dump(self.performance_history, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving performance history: {e}")

    def analyze_performance(self, days: int = 100) -> Dict:
        """Analyze performance over the specified number of days.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary containing performance metrics
        """
        try:
            # Get recent results
            recent_results = self.fetcher.get_recent_results(days)
            if recent_results.empty:
                logger.warning("No recent results found for analysis")
                return {}

            # Initialize analyzer with recent data
            self.analyzer.load_csv_files(self.data_dir)

            # Find optimal strategy
            best_pick_size, best_method, best_roi = self._find_optimal_strategy()

            # Analyze consecutive payouts
            consecutive_results = self._analyze_consecutive_payouts(best_pick_size)

            # Calculate performance metrics
            performance = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "days_analyzed": days,
                "total_games": len(recent_results),
                "best_strategy": {
                    "pick_size": best_pick_size,
                    "method": best_method,
                    "roi": best_roi,
                },
                "consecutive_payouts": consecutive_results,
                "recent_performance": self._calculate_recent_performance(
                    recent_results, best_pick_size
                ),
            }

            # Add to history
            self.performance_history.append(performance)
            self._save_performance_history()

            return performance

        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            return {}

    def _find_optimal_strategy(self) -> Tuple[int, str, float]:
        """Find the optimal betting strategy."""
        best_roi = -float("inf")
        best_pick_size = None
        best_method = None

        for pick_size in [5, 10, 15, 20]:
            for method in ["frequency", "patterns", "markov", "due"]:
                sim_result = self.analyzer.simulate_strategy(method, pick_size)
                if sim_result["roi_percent"] > best_roi:
                    best_roi = sim_result["roi_percent"]
                    best_pick_size = pick_size
                    best_method = method

        return best_pick_size, best_method, best_roi

    def _analyze_consecutive_payouts(self, pick_size: int) -> Dict:
        """Analyze consecutive payouts for different methods."""
        results = {}
        methods = ["frequency", "patterns", "markov", "due"]

        for method in methods:
            consecutive_payouts = []
            current_streak = 0
            total_payout = 0
            total_bets = 0

            for _ in range(1000):  # Simulate 1000 games
                prediction = self.analyzer.predict_next_draw(method, pick_size)
                actual_draw = self.analyzer.data[-1]
                matches = len(set(prediction) & set(actual_draw))

                payout = self.analyzer.calculate_payout(pick_size, matches)
                total_payout += payout
                total_bets += 1

                if payout > 0:
                    current_streak += 1
                    consecutive_payouts.append(current_streak)
                else:
                    current_streak = 0
                    consecutive_payouts.append(0)

            results[method] = {
                "max_streak": max(consecutive_payouts),
                "avg_streak": sum(consecutive_payouts) / len(consecutive_payouts),
                "roi": (total_payout - total_bets) / total_bets * 100,
            }

        return results

    def _calculate_recent_performance(self, recent_results: pd.DataFrame, pick_size: int) -> Dict:
        """Calculate performance metrics for recent results."""
        total_games = len(recent_results)
        if total_games == 0:
            return {}

        wins = 0
        total_payout = 0
        total_cost = total_games

        for _, row in recent_results.iterrows():
            drawn_numbers = [int(row[f"NUMBER DRAWN {i+1}"]) for i in range(20)]
            prediction = self.analyzer.predict_next_draw("frequency", pick_size)
            matches = len(set(prediction) & set(drawn_numbers))

            payout = self.analyzer.calculate_payout(pick_size, matches)
            if payout > 0:
                wins += 1
                total_payout += payout

        return {
            "total_games": total_games,
            "wins": wins,
            "win_percentage": (wins / total_games) * 100,
            "total_payout": total_payout,
            "total_cost": total_cost,
            "net_profit": total_payout - total_cost,
            "roi": ((total_payout - total_cost) / total_cost) * 100,
        }

    def plot_performance(self, save_dir: str):
        """Generate performance visualization plots."""
        if not self.performance_history:
            logger.warning("No performance history to plot")
            return

        os.makedirs(save_dir, exist_ok=True)

        # Plot ROI over time
        plt.figure(figsize=(12, 6))
        dates = [p["date"] for p in self.performance_history]
        rois = [p["best_strategy"]["roi"] for p in self.performance_history]
        plt.plot(dates, rois, marker="o")
        plt.title("ROI Over Time")
        plt.xlabel("Date")
        plt.ylabel("ROI (%)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "roi_over_time.png"))
        plt.close()

        # Plot consecutive payouts
        plt.figure(figsize=(12, 6))
        methods = list(self.performance_history[-1]["consecutive_payouts"].keys())
        max_streaks = [
            self.performance_history[-1]["consecutive_payouts"][m]["max_streak"] for m in methods
        ]
        plt.bar(methods, max_streaks)
        plt.title("Maximum Consecutive Payouts by Method")
        plt.xlabel("Method")
        plt.ylabel("Maximum Streak")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "consecutive_payouts.png"))
        plt.close()


def main():
    # Initialize tracker
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "KenoPastYears")
    tracker = PerformanceTracker(data_dir)

    # Run analysis
    logger.info("Starting performance analysis...")
    performance = tracker.analyze_performance(days=100)

    if performance:
        # Print results
        print("\n=== Performance Analysis Results ===")
        print(f"\nAnalysis Date: {performance['date']}")
        print(f"Days Analyzed: {performance['days_analyzed']}")
        print(f"Total Games: {performance['total_games']}")

        print("\nBest Strategy:")
        print(f"Pick Size: {performance['best_strategy']['pick_size']}")
        print(f"Method: {performance['best_strategy']['method']}")
        print(f"ROI: {performance['best_strategy']['roi']:.2f}%")

        print("\nRecent Performance:")
        recent = performance["recent_performance"]
        print(f"Win Rate: {recent['win_percentage']:.2f}%")
        print(f"Net Profit: ${recent['net_profit']:.2f}")
        print(f"ROI: {recent['roi']:.2f}%")

        print("\nConsecutive Payouts by Method:")
        for method, stats in performance["consecutive_payouts"].items():
            print(f"\n{method.capitalize()} Method:")
            print(f"Maximum Streak: {stats['max_streak']}")
            print(f"Average Streak: {stats['avg_streak']:.2f}")
            print(f"ROI: {stats['roi']:.2f}%")

        # Generate plots
        output_dir = "performance_analysis"
        tracker.plot_performance(output_dir)
        print(f"\nPerformance plots saved to {output_dir}/")

        # Check if we're meeting the 100-game goal
        if recent["net_profit"] > 0:
            print("\n✓ Goal Achieved: Profitable over 100 games!")
        else:
            print("\n✗ Goal Not Met: Not profitable over 100 games")
    else:
        print("Failed to complete performance analysis")


if __name__ == "__main__":
    main()
