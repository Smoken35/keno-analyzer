#!/usr/bin/env python3
"""
Script to optimize monitoring thresholds based on historical data.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from prometheus_client import start_http_server
from prometheus_client.core import REGISTRY, GaugeMetricFamily

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """Class to handle threshold optimization for monitoring metrics."""

    def __init__(self, data_path: str):
        """Initialize the threshold optimizer.

        Args:
            data_path: Path to the historical data file
        """
        self.data_path = Path(data_path)
        self.metrics_data: Dict[str, List[float]] = {}
        self.optimized_thresholds: Dict[str, Dict[str, float]] = {}

    def load_data(self) -> None:
        """Load historical data from the specified path."""
        try:
            with open(self.data_path, "r") as f:
                self.metrics_data = json.load(f)
            logger.info(f"Loaded data from {self.data_path}")
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in data file: {self.data_path}")
            raise

    def optimize_thresholds(self, method: str = "percentile") -> None:
        """Optimize thresholds for each metric.

        Args:
            method: Method to use for threshold optimization ('percentile' or 'std')
        """
        for metric_name, values in self.metrics_data.items():
            if not values:
                logger.warning(f"No data for metric: {metric_name}")
                continue

            values_array = np.array(values)

            if method == "percentile":
                # Use 95th percentile for warning and 99th for critical
                warning = np.percentile(values_array, 95)
                critical = np.percentile(values_array, 99)
            else:  # std method
                mean = np.mean(values_array)
                std = np.std(values_array)
                warning = mean + 2 * std
                critical = mean + 3 * std

            self.optimized_thresholds[metric_name] = {
                "warning": float(warning),
                "critical": float(critical),
            }

    def save_thresholds(self, output_path: Optional[str] = None) -> None:
        """Save optimized thresholds to a file.

        Args:
            output_path: Path to save the thresholds (defaults to data_path with _thresholds suffix)
        """
        if not output_path:
            output_path = str(self.data_path).replace(".json", "_thresholds.json")

        try:
            with open(output_path, "w") as f:
                json.dump(self.optimized_thresholds, f, indent=2)
            logger.info(f"Saved thresholds to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save thresholds: {e}")
            raise


def main():
    """Main function to run the threshold optimization."""
    parser = argparse.ArgumentParser(description="Optimize monitoring thresholds")
    parser.add_argument("--data-path", required=True, help="Path to historical data file")
    parser.add_argument(
        "--method",
        choices=["percentile", "std"],
        default="percentile",
        help="Method to use for threshold optimization",
    )
    parser.add_argument("--output-path", help="Path to save optimized thresholds")
    parser.add_argument("--port", type=int, default=9090, help="Port to expose Prometheus metrics")

    args = parser.parse_args()

    try:
        optimizer = ThresholdOptimizer(args.data_path)
        optimizer.load_data()
        optimizer.optimize_thresholds(method=args.method)
        optimizer.save_thresholds(args.output_path)

        # Start Prometheus metrics server
        start_http_server(args.port)
        logger.info(f"Started metrics server on port {args.port}")

        # Keep the script running
        while True:
            pass

    except Exception as e:
        logger.error(f"Error during threshold optimization: {e}")
        raise


if __name__ == "__main__":
    main()
