#!/usr/bin/env python3
"""
Example script demonstrating the Keno Validation and Tracking System.
This script shows how to:
1. Record predictions from different methods
2. Validate predictions against actual results
3. Generate performance reports and visualizations
4. Compare different prediction methods
"""

import json
import os
from datetime import date, timedelta

from keno import KenoAnalyzer, KenoPredictor, ValidationTracker


def main():
    # Initialize components
    analyzer = KenoAnalyzer()
    predictor = KenoPredictor(analyzer)
    tracker = ValidationTracker()

    # Create output directories
    os.makedirs("outputs/validation", exist_ok=True)
    os.makedirs("outputs/visualizations", exist_ok=True)

    # Example 1: Record and validate individual predictions
    print("\nExample 1: Individual Prediction Validation")
    methods = ["frequency", "pattern", "markov", "ensemble"]

    for method in methods:
        # Make prediction
        prediction = predictor.predict(method=method, num_numbers=5)

        # Record prediction
        pred_id = tracker.record_prediction(
            method=method,
            numbers=prediction,
            draw_date=date.today(),
            metadata={"confidence": predictor.get_confidence(method)},
        )
        print(f"Recorded {method} prediction: {prediction}")

        # Simulate actual results (in practice, these would come from real draws)
        actual = [1, 5, 10, 15, 20]  # Example actual numbers

        # Validate prediction
        tracker.validate_prediction(pred_id, actual)
        print(f"Validated {method} prediction against {actual}")

    # Example 2: Bulk validation of historical predictions
    print("\nExample 2: Bulk Validation")

    # Generate some historical predictions
    start_date = date.today() - timedelta(days=30)
    historical_predictions = {}

    for i in range(30):
        current_date = start_date + timedelta(days=i)
        for method in methods:
            pred = predictor.predict(method=method, num_numbers=5)
            pred_id = tracker.record_prediction(method=method, numbers=pred, draw_date=current_date)
            historical_predictions[current_date.isoformat()] = pred

    # Simulate historical results (in practice, these would be real historical data)
    historical_results = {
        d.isoformat(): analyzer.generate_random_numbers(5) for d in historical_predictions.keys()
    }

    # Bulk validate
    tracker.bulk_validate(historical_results)
    print("Completed bulk validation of 30 days of historical predictions")

    # Example 3: Generate performance report
    print("\nExample 3: Performance Analysis")

    # Generate comprehensive report
    report_path = "outputs/validation/performance_report.json"
    tracker.generate_report(output_file=report_path)
    print(f"Generated performance report: {report_path}")

    # Plot method comparisons
    tracker.plot_method_comparison(
        methods=methods, metric="accuracy", filename="outputs/visualizations/method_comparison.png"
    )
    print("Generated method comparison visualization")

    # Example 4: Analyze prediction patterns
    print("\nExample 4: Pattern Analysis")

    # Generate trend visualizations for each method
    for method in methods:
        tracker.plot_prediction_history(
            method=method, filename=f"outputs/visualizations/{method}_trend.png"
        )
        print(f"Generated trend visualization for {method}")

    # Example 5: Statistical analysis
    print("\nExample 5: Statistical Analysis")

    # Compare methods statistically
    stats = tracker.compare_methods_statistically(methods)
    stats_path = "outputs/validation/statistical_analysis.json"

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Generated statistical analysis: {stats_path}")

    # Print summary
    print("\nValidation System Demo Complete")
    print("Generated files:")
    print(f"- Performance Report: {report_path}")
    print(f"- Statistical Analysis: {stats_path}")
    print("- Visualizations in: outputs/visualizations/")


if __name__ == "__main__":
    main()
