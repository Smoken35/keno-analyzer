"""
Keno Validation and Tracking System.
Provides tools for validating predictions, tracking performance, and analyzing results.
"""

import json
import logging
import os
import sqlite3
import uuid
from datetime import date, datetime
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)


class ValidationTracker:
    """
    Tracks and validates Keno predictions, providing performance analysis and reporting.
    """

    def __init__(self, storage_dir: str):
        """
        Initialize the tracker.

        Args:
            storage_dir: Directory to store validation data
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.predictions_file = os.path.join(storage_dir, "predictions.json")
        self.validations_file = os.path.join(storage_dir, "validations.json")
        self._load_data()

    def _load_data(self) -> None:
        """Load validation data from storage."""
        if os.path.exists(self.predictions_file):
            with open(self.predictions_file, "r") as f:
                self.predictions = json.load(f)
        else:
            self.predictions = {}

        if os.path.exists(self.validations_file):
            with open(self.validations_file, "r") as f:
                self.validations = json.load(f)
        else:
            self.validations = {}

    def _convert_to_native_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native_types(item) for item in obj]
        return obj

    def _save_data(self) -> None:
        """Save validation data to storage."""
        with open(self.predictions_file, "w") as f:
            json.dump(self._convert_to_native_types(self.predictions), f)
        with open(self.validations_file, "w") as f:
            json.dump(self._convert_to_native_types(self.validations), f)

    def record_prediction(
        self,
        method: str,
        predicted_numbers: List[int],
        draw_date: Optional[date] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Record a new prediction.

        Args:
            method: Prediction method used
            predicted_numbers: List of predicted numbers
            draw_date: Date of the draw being predicted
            metadata: Additional metadata about the prediction

        Returns:
            Prediction ID
        """
        prediction_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        prediction = {
            "id": prediction_id,
            "method": method,
            "numbers": predicted_numbers,
            "timestamp": timestamp,
            "draw_date": draw_date.isoformat() if draw_date else None,
            "metadata": metadata or {},
        }

        self.predictions[prediction_id] = prediction
        self._save_data()
        return prediction_id

    def validate_prediction(
        self, prediction_id: str, actual_numbers: List[int]
    ) -> Dict[str, Union[int, float]]:
        """
        Validate a prediction against actual results.

        Args:
            prediction_id: ID of the prediction to validate
            actual_numbers: Actual winning numbers

        Returns:
            Dictionary containing validation results
        """
        if prediction_id not in self.predictions:
            raise ValueError(f"Unknown prediction ID: {prediction_id}")

        prediction = self.predictions[prediction_id]
        matches = len(set(prediction["numbers"]) & set(actual_numbers))
        accuracy = matches / len(prediction["numbers"])

        validation = {
            "prediction_id": prediction_id,
            "actual_numbers": actual_numbers,
            "matches": matches,
            "accuracy": accuracy,
            "timestamp": datetime.now().isoformat(),
        }

        self.validations[prediction_id] = validation
        self._save_data()
        return validation

    def analyze_historical_performance(
        self, method: str, pick_size: int, analyzer: "KenoAnalyzer", num_draws: int = 100
    ) -> Dict:
        """
        Analyze historical performance of a prediction method.

        Args:
            method: Prediction method to analyze
            pick_size: Number of numbers to pick
            analyzer: KenoAnalyzer instance
            num_draws: Number of historical draws to analyze

        Returns:
            Dict containing performance metrics
        """
        matches_list = []

        # Use only the last num_draws draws
        historical_draws = analyzer.data[-num_draws:]

        for i in range(len(historical_draws) - 1):
            # Make prediction using data up to current draw
            analyzer.data = historical_draws[: i + 1]
            prediction = analyzer.predict_next_draw(method)[:pick_size]

            # Compare with next draw
            actual = historical_draws[i + 1]
            matches = len(set(prediction) & set(actual))
            matches_list.append(matches)

        # Calculate statistics
        avg_matches = np.mean(matches_list)
        p_value = self._calculate_significance(avg_matches, len(matches_list))

        return {"avg_matches": avg_matches, "p_value": p_value, "num_draws": len(matches_list)}

    def compare_methods(
        self, analyzer: "KenoAnalyzer", pick_size: int = 4, num_draws: int = 100
    ) -> pd.DataFrame:
        """
        Compare performance of different prediction methods.

        Args:
            analyzer: KenoAnalyzer instance
            pick_size: Number of numbers to pick
            num_draws: Number of historical draws to analyze

        Returns:
            DataFrame with comparison results
        """
        methods = ["frequency", "patterns", "markov", "due"]
        results = []

        for method in methods:
            validation = self.analyze_historical_performance(method, pick_size, analyzer, num_draws)
            results.append(
                {
                    "method": method,
                    "avg_matches": validation["avg_matches"],
                    "p_value": validation["p_value"],
                }
            )

        return pd.DataFrame(results)

    def _calculate_significance(self, avg_matches: float, n: int) -> float:
        """
        Calculate statistical significance of prediction results.

        Args:
            avg_matches: Average number of matches
            n: Number of predictions

        Returns:
            P-value from binomial test
        """
        # Use binomtest instead of deprecated binom_test
        result = stats.binomtest(
            k=int(avg_matches * n),  # Total successes
            n=n * 20,  # Total trials (20 numbers per prediction)
            p=20 / 80,  # Expected probability
            alternative="greater",
        )
        return result.pvalue

    def get_method_performance(self, method: str) -> Dict[str, Union[float, int]]:
        """Get performance statistics for a specific method."""
        method_predictions = [
            pred for pred in self.predictions.values() if pred["method"] == method
        ]
        if not method_predictions:
            return {}

        method_validations = [
            self.validations[pred["id"]]
            for pred in method_predictions
            if pred["id"] in self.validations
        ]

        if not method_validations:
            return {}

        matches = [val["matches"] for val in method_validations]
        accuracies = [val["accuracy"] for val in method_validations]

        avg_matches = np.mean(matches)
        avg_accuracy = np.mean(accuracies)
        p_value = self._calculate_significance(avg_matches, len(matches))

        return {
            "accuracy": avg_accuracy,
            "matches": avg_matches,
            "p_value": p_value,
            "predictions": len(method_predictions),
        }

    def get_prediction_history(
        self, method: Optional[str] = None, limit: int = 100
    ) -> pd.DataFrame:
        """
        Get prediction history with validation results.

        Args:
            method: Optional method name to filter by
            limit: Maximum number of records to return

        Returns:
            DataFrame with prediction history
        """
        history_df = pd.DataFrame(self.validations.values())
        history_df = history_df[history_df["method"] == method]
        history_df = history_df.sort_values(by="timestamp", ascending=False).head(limit)
        history_df["numbers"] = history_df["numbers"].apply(json.dumps)
        history_df["actual_numbers"] = history_df["actual_numbers"].apply(json.dumps)
        return history_df

    def plot_method_comparison(self, comparison_df: pd.DataFrame, filename: str) -> None:
        """
        Create a visualization comparing method performance.

        Args:
            comparison_df: DataFrame from compare_methods()
            filename: Output file path
        """
        plt.figure(figsize=(12, 6))

        # Create bar plot
        ax = sns.barplot(data=comparison_df, x="method", y="avg_matches", palette="viridis")

        # Add significance markers
        for i, significant in enumerate(comparison_df["significant"]):
            color = "green" if significant else "red"
            marker = "★" if significant else "✗"
            plt.text(
                i,
                comparison_df["avg_matches"].iloc[i],
                marker,
                color=color,
                ha="center",
                va="bottom",
            )

        plt.title("Prediction Method Comparison")
        plt.xlabel("Method")
        plt.ylabel("Average Matches")
        plt.xticks(rotation=45)

        # Add baseline
        baseline = (20 / 80) * 100  # Expected accuracy by chance
        plt.axhline(y=baseline, color="r", linestyle="--", alpha=0.5, label="Random Chance")
        plt.legend()

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_prediction_history(self, method: str, filename: str, limit: int = 50) -> None:
        """
        Create a visualization of prediction history.

        Args:
            method: Prediction method to plot
            filename: Output file path
            limit: Number of predictions to include
        """
        history_df = self.get_prediction_history(method=method, limit=limit)

        plt.figure(figsize=(12, 6))

        # Plot accuracy over time
        plt.plot(
            range(len(history_df)), history_df["accuracy"], marker="o", linestyle="-", alpha=0.6
        )

        # Add trend line
        z = np.polyfit(range(len(history_df)), history_df["accuracy"], 1)
        p = np.poly1d(z)
        plt.plot(range(len(history_df)), p(range(len(history_df))), "r--", alpha=0.8, label="Trend")

        plt.title(f"Prediction Accuracy History: {method}")
        plt.xlabel("Prediction Number")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def generate_validation_report(self, output_dir: str = "validation_reports") -> str:
        """
        Generate a comprehensive validation report.

        Args:
            output_dir: Directory for report files

        Returns:
            str: Path to report directory
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(output_dir, f"report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)

        # Get performance data
        performance_df = pd.DataFrame(
            self.get_method_performance(method) for method in self.predictions.values()
        )
        performance_df.to_csv(os.path.join(report_dir, "method_performance.csv"), index=False)

        # Create performance plot
        self.plot_method_comparison(
            performance_df, os.path.join(report_dir, "method_comparison.png")
        )

        # Generate history plots for each method
        for method in performance_df["method"]:
            self.plot_prediction_history(
                method=method, filename=os.path.join(report_dir, f"{method}_history.png")
            )

        # Create summary report
        summary = {
            "generated_at": datetime.now().isoformat(),
            "total_predictions": int(performance_df["total_predictions"].sum()),
            "best_method": performance_df.loc[performance_df["avg_accuracy"].idxmax(), "method"],
            "method_stats": performance_df.to_dict("records"),
        }

        with open(os.path.join(report_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        return report_dir

    def bulk_validate(
        self, actuals: Dict[date, List[int]]
    ) -> Dict[str, List[Dict[str, Union[float, int]]]]:
        """Validate multiple predictions against actual results."""
        results = {}
        for draw_date, actual_numbers in actuals.items():
            # Find predictions for this date
            date_predictions = [
                pred
                for pred in self.predictions.values()
                if pred.get("draw_date") == draw_date.isoformat()
            ]

            if not date_predictions:
                continue

            date_results = []
            for pred in date_predictions:
                validation = self.validate_prediction(pred["id"], actual_numbers)
                date_results.append(validation)

            results[draw_date.isoformat()] = date_results

        return results

    def generate_report(self, method: Optional[str] = None) -> Dict[str, Dict]:
        """Generate a performance report for one or all methods."""
        if method:
            stats = self.get_method_performance(method)

            # Add trend analysis
            method_validations = [
                val
                for val in self.validations.values()
                if val.get("prediction_id")
                in [pred["id"] for pred in self.predictions.values() if pred["method"] == method]
            ]

            if method_validations:
                accuracies = [val["accuracy"] for val in method_validations]
                x = np.arange(len(accuracies))
                z = np.polyfit(x, accuracies, 1)
                trend = float(z[0])  # Convert to native Python float
            else:
                trend = 0.0

            stats["trend"] = trend
            return stats

        methods = set(pred["method"] for pred in self.predictions.values())
        return {method: self.get_method_performance(method) for method in methods}
