"""
Integration tests for the Keno Prediction Tool.
"""

import json
import os
import tempfile
import unittest
from datetime import date, datetime

import numpy as np

from keno.analysis.analyzer import KenoAnalyzer
from keno.validation.tracker import ValidationTracker
from keno.visualization.visualizer import KenoVisualizer


class TestIntegration(unittest.TestCase):
    """Integration test cases."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that can be reused."""
        # Create test directories
        cls.test_dirs = [
            "~/.keno/test/data",
            "~/.keno/test/cache",
            "~/.keno/test/validation",
            "~/.keno/test/visualizations",
        ]

        for directory in cls.test_dirs:
            dir_path = os.path.expanduser(directory)
            os.makedirs(dir_path, exist_ok=True)

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.data = []
        for _ in range(30):
            draw = sorted(np.random.choice(range(1, 81), size=20, replace=False))
            self.data.append(draw)

        # Initialize analyzer with sample data
        self.analyzer = KenoAnalyzer("sample")
        self.analyzer.data = self.data

        self.standard_payouts = {
            4: {4: 100, 3: 5, 2: 1, 1: 0, 0: 0},
            5: {5: 500, 4: 15, 3: 2, 2: 0, 1: 0, 0: 0},
        }
        self.analyzer.payout_table = self.standard_payouts

        # Create temporary directory for ValidationTracker
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = ValidationTracker(storage_dir=self.temp_dir)

        self.visualizer = KenoVisualizer(self.analyzer)

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create visualization directory if it doesn't exist
        vis_dir = os.path.join(os.path.dirname(__file__), "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # 1. Record and validate prediction
        prediction_id = self.tracker.record_prediction(
            method="frequency", predicted_numbers=self.analyzer.predict_next_draw("frequency")
        )

        actual_numbers = sorted(np.random.choice(range(1, 81), size=20, replace=False))
        result = self.tracker.validate_prediction(prediction_id, actual_numbers)

        # Check validation result
        self.assertIsInstance(result, dict)
        self.assertIn("matches", result)
        self.assertIn("accuracy", result)

        # 2. Analyze historical performance
        validation = self.tracker.analyze_historical_performance(
            method="frequency", pick_size=4, analyzer=self.analyzer, num_draws=10
        )

        # Check validation structure
        self.assertIsInstance(validation, dict)
        self.assertIn("avg_matches", validation)
        self.assertIn("p_value", validation)

        # 3. Generate visualizations
        heatmap_path = os.path.join(vis_dir, "test_heatmap.png")
        pattern_path = os.path.join(vis_dir, "test_patterns.png")
        prediction_path = os.path.join(vis_dir, "test_predictions.png")

        self.visualizer.plot_frequency_heatmap(heatmap_path)
        self.visualizer.plot_pattern_analysis(pattern_path)
        self.visualizer.plot_prediction_comparison(4, 10, prediction_path)

        # Check file creation
        self.assertTrue(os.path.exists(heatmap_path))
        self.assertTrue(os.path.exists(pattern_path))
        self.assertTrue(os.path.exists(prediction_path))

    def test_data_flow_integrity(self):
        """Test data integrity through the analysis pipeline."""
        # Create visualization directory if it doesn't exist
        vis_dir = os.path.join(os.path.dirname(__file__), "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # 1. Record prediction
        prediction_id = self.tracker.record_prediction(
            method="frequency", predicted_numbers=self.analyzer.predict_next_draw("frequency")
        )

        # 2. Validate prediction
        actual_numbers = sorted(np.random.choice(range(1, 81), size=20, replace=False))
        result = self.tracker.validate_prediction(prediction_id, actual_numbers)

        # Check validation result
        self.assertIsInstance(result, dict)
        self.assertIn("matches", result)
        self.assertIn("accuracy", result)

        # 3. Generate visualizations
        flow_heatmap_path = os.path.join(vis_dir, "test_flow_heatmap.png")
        flow_pattern_path = os.path.join(vis_dir, "test_flow_patterns.png")
        flow_prediction_path = os.path.join(vis_dir, "test_flow.png")

        self.visualizer.plot_frequency_heatmap(flow_heatmap_path)
        self.visualizer.plot_pattern_analysis(flow_pattern_path)
        self.visualizer.plot_prediction_comparison(4, 10, flow_prediction_path)

        # Check file creation
        self.assertTrue(os.path.exists(flow_heatmap_path))
        self.assertTrue(os.path.exists(flow_pattern_path))
        self.assertTrue(os.path.exists(flow_prediction_path))

    def test_error_handling(self):
        """Test error handling in integrated workflow."""
        # 1. Test invalid prediction method
        with self.assertRaises(ValueError):
            self.analyzer.predict_next_draw("invalid_method")

        # 2. Test invalid pick size
        with self.assertRaises(ValueError):
            self.analyzer.calculate_expected_value(pick_size=10, method="frequency")

        # 3. Test invalid validation
        with self.assertRaises(ValueError):
            self.tracker.validate_prediction(999999, [1, 2, 3, 4, 5])

        # 4. Test data validation
        validation = self.tracker.analyze_historical_performance(
            method="frequency", pick_size=4, analyzer=self.analyzer, num_draws=5
        )

        self.assertIsInstance(validation, dict)
        self.assertIn("avg_matches", validation)
        self.assertIn("p_value", validation)

    def test_performance_simulation(self):
        """Test long-term performance simulation."""
        # Set up payout table for pick size 4
        self.analyzer.set_payout_table(
            {
                4: {
                    4: 75.0,  # All 4 correct
                    3: 5.0,  # 3 correct
                    2: 1.0,  # 2 correct
                    1: 0.0,  # 1 correct
                    0: 0.0,  # None correct
                }
            }
        )

        # Run simulation
        result = self.analyzer.simulate_strategy(
            method="frequency", pick_size=4, bet_size=1.0, num_simulations=100
        )

        # Check structure of results
        self.assertIn("total_return", result)
        self.assertIn("roi_percent", result)
        self.assertIn("match_distribution", result)
        self.assertIn("num_simulations", result)

        # Check match distribution
        self.assertEqual(sum(result["match_distribution"].values()), result["num_simulations"])

        # Check ROI calculation
        expected_roi = (result["total_return"] / (result["num_simulations"])) * 100
        self.assertAlmostEqual(result["roi_percent"], expected_roi, places=2)


if __name__ == "__main__":
    unittest.main()
