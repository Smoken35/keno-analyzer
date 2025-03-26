"""
Command-line interface for the Keno Prediction Tool.
"""

import argparse
import logging
import sys
from datetime import datetime
from typing import List, Optional

from .analyzer import KenoAnalyzer
from .validation.tracker import ValidationTracker
from .visualization.visualizer import KenoVisualizer


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def predict(args) -> None:
    """Handle prediction commands."""
    analyzer = KenoAnalyzer()
    analyzer.scrape_data(source=args.source, days=args.days)

    if args.method == "all":
        methods = ["frequency", "cycles", "markov", "due", "ensemble"]
        for method in methods:
            predictions = getattr(analyzer, f"predict_using_{method}")(args.picks)
            print(f"{method.capitalize()} predictions: {predictions}")
    else:
        predictions = getattr(analyzer, f"predict_using_{args.method}")(args.picks)
        print(f"Predictions using {args.method}: {predictions}")


def analyze(args) -> None:
    """Handle analysis commands."""
    analyzer = KenoAnalyzer()
    analyzer.scrape_data(source=args.source, days=args.days)

    if args.type == "frequency":
        freq = analyzer.analyze_frequency()
        print("Frequency Analysis:")
        for num, count in sorted(freq.items()):
            print(f"Number {num}: {count} appearances")

    elif args.type == "patterns":
        patterns = analyzer.analyze_patterns()
        print("\nPattern Analysis:")
        print(f"Hot numbers: {patterns['hot_numbers']}")
        print(f"Cold numbers: {patterns['cold_numbers']}")
        print("\nTop 5 pairs:")
        pairs = sorted(patterns["pairs"].items(), key=lambda x: x[1], reverse=True)[:5]
        for pair, count in pairs:
            print(f"{pair}: {count} appearances")


def visualize(args) -> None:
    """Handle visualization commands."""
    analyzer = KenoAnalyzer()
    analyzer.scrape_data(source=args.source, days=args.days)

    visualizer = KenoVisualizer(analyzer)

    if args.type == "frequency":
        visualizer.plot_frequency_heatmap("frequency_heatmap.png")
        print("Created frequency heatmap: frequency_heatmap.png")

    elif args.type == "patterns":
        visualizer.plot_pattern_analysis("pattern_analysis.png")
        print("Created pattern analysis: pattern_analysis.png")

    elif args.type == "predictions":
        visualizer.plot_prediction_comparison(args.picks, 100, "prediction_comparison.png")
        print("Created prediction comparison: prediction_comparison.png")

    elif args.type == "dashboard":
        tracker = ValidationTracker()
        visualizer.create_dashboard(tracker)
        print("Created visualization dashboard")


def validate(args) -> None:
    """Handle validation commands."""
    analyzer = KenoAnalyzer()
    analyzer.scrape_data(source=args.source, days=args.days)

    tracker = ValidationTracker()

    if args.method == "all":
        methods = ["frequency", "cycles", "markov", "due", "ensemble"]
        results = []
        for method in methods:
            result = tracker.analyze_historical_performance(
                method, args.picks, analyzer, args.draws
            )
            results.append(result)

        print("\nValidation Results:")
        for result in results:
            print(f"\nMethod: {result['method']}")
            print(f"Accuracy: {result['accuracy_pct']:.2f}%")
            print(f"Average matches: {result['avg_matches']:.2f}")
            print(f"Consistency: {result['consistency']:.2f}")
    else:
        result = tracker.analyze_historical_performance(
            args.method, args.picks, analyzer, args.draws
        )
        print("\nValidation Results:")
        print(f"Method: {result['method']}")
        print(f"Accuracy: {result['accuracy_pct']:.2f}%")
        print(f"Average matches: {result['avg_matches']:.2f}")
        print(f"Consistency: {result['consistency']:.2f}")


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Keno Prediction Tool - Analyze and predict Keno numbers"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument(
        "--method",
        choices=["frequency", "cycles", "markov", "due", "ensemble", "all"],
        default="ensemble",
        help="Prediction method to use",
    )
    predict_parser.add_argument("--picks", type=int, default=4, help="Number of picks to predict")
    predict_parser.add_argument("--source", default="sample", help="Data source (sample or URL)")
    predict_parser.add_argument(
        "--days", type=int, default=30, help="Number of days of data to use"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze patterns")
    analyze_parser.add_argument(
        "--type",
        choices=["frequency", "patterns"],
        default="patterns",
        help="Type of analysis to perform",
    )
    analyze_parser.add_argument("--source", default="sample", help="Data source (sample or URL)")
    analyze_parser.add_argument(
        "--days", type=int, default=30, help="Number of days of data to analyze"
    )

    # Visualize command
    visualize_parser = subparsers.add_parser("visualize", help="Create visualizations")
    visualize_parser.add_argument(
        "--type",
        choices=["frequency", "patterns", "predictions", "dashboard"],
        default="dashboard",
        help="Type of visualization to create",
    )
    visualize_parser.add_argument(
        "--picks", type=int, default=4, help="Number of picks for prediction visualization"
    )
    visualize_parser.add_argument("--source", default="sample", help="Data source (sample or URL)")
    visualize_parser.add_argument(
        "--days", type=int, default=30, help="Number of days of data to visualize"
    )

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate predictions")
    validate_parser.add_argument(
        "--method",
        choices=["frequency", "cycles", "markov", "due", "ensemble", "all"],
        default="ensemble",
        help="Method to validate",
    )
    validate_parser.add_argument("--picks", type=int, default=4, help="Number of picks to validate")
    validate_parser.add_argument(
        "--draws", type=int, default=100, help="Number of draws to validate against"
    )
    validate_parser.add_argument("--source", default="sample", help="Data source (sample or URL)")
    validate_parser.add_argument(
        "--days", type=int, default=30, help="Number of days of data to use"
    )

    args = parser.parse_args(args)
    setup_logging(args.verbose)

    try:
        if args.command == "predict":
            predict(args)
        elif args.command == "analyze":
            analyze(args)
        elif args.command == "visualize":
            visualize(args)
        elif args.command == "validate":
            validate(args)
        else:
            parser.print_help()
            return 1

        return 0

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
