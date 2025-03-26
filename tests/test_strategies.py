"""
Test prediction strategies.
"""

import unittest

import numpy as np

from keno.strategies.strategy_analyzer import StrategyAnalyzer


class TestStrategyAnalyzer(unittest.TestCase):
    """Test cases for the StrategyAnalyzer class."""

    def setUp(self):
        """Set up test environment."""
        # Create sample historical data
        self.historical_data = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])
        self.analyzer = StrategyAnalyzer(historical_data=self.historical_data)

    def test_analyze_strategy(self):
        """Test basic strategy analysis."""
        # Create sample data
        sample_data = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])

        # Test strategy analysis
        results = self.analyzer.analyze_strategy(sample_data)
        self.assertIsNotNone(results)
        self.assertTrue(isinstance(results, dict))

    def test_evaluate_performance(self):
        """Test strategy performance evaluation."""
        # Create sample predictions and actual results
        predictions = np.array([1, 2, 3, 4, 5])
        actual = np.array([2, 3, 4, 5, 6])

        # Test performance evaluation
        score = self.analyzer.evaluate_performance(predictions, actual)
        self.assertIsInstance(score, float)
        self.assertTrue(0 <= score <= 1)


if __name__ == "__main__":
    unittest.main()
