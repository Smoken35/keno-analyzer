"""
Unit tests for Keno prediction strategies.
"""

import unittest

import numpy as np

from keno_analyzer.prediction.strategies.pattern_strategy import PatternBasedStrategy
from keno_analyzer.prediction.strategies.rule_strategy import RuleBasedStrategy

# Sample data for testing
sample_data = [
    [3, 12, 24, 33, 45, 56, 60, 70, 72, 80, 1, 5, 7, 19, 29, 37, 44, 52, 63, 78],
    [2, 11, 22, 34, 46, 55, 59, 68, 73, 79, 6, 8, 18, 27, 31, 42, 50, 61, 67, 76],
    [1, 10, 20, 35, 48, 54, 57, 66, 74, 77, 9, 15, 21, 25, 32, 41, 49, 62, 65, 75],
    [4, 13, 23, 36, 47, 53, 58, 69, 71, 81, 17, 26, 28, 30, 38, 43, 51, 64, 66, 77],
    [14, 16, 19, 33, 45, 55, 60, 70, 72, 80, 1, 6, 7, 18, 27, 37, 44, 52, 63, 78],
]


class TestKenoStrategies(unittest.TestCase):
    """Test cases for Keno prediction strategies."""

    def setUp(self):
        """Set up test fixtures."""
        self.pattern_strategy = PatternBasedStrategy()
        self.rule_strategy = RuleBasedStrategy()
        self.pattern_strategy.fit(sample_data)
        self.rule_strategy.fit(sample_data)

    def test_strategy_initialization(self):
        """Test strategy initialization and configuration."""
        for strategy in [self.pattern_strategy, self.rule_strategy]:
            # Test strategy info
            info = strategy.get_strategy_info()
            self.assertIsInstance(info, dict)
            self.assertIn("name", info)
            self.assertIn("description", info)
            self.assertIn("parameters", info)

            # Test configuration
            self.assertIsInstance(strategy.config, dict)

    def test_prediction_validity(self):
        """Test prediction validity for both strategies."""
        for strategy in [self.pattern_strategy, self.rule_strategy]:
            # Test different prediction sizes
            for num_picks in [5, 10, 15, 20]:
                prediction = strategy.predict(draw_index=0, num_picks=num_picks)

                # Check length
                self.assertEqual(len(prediction), num_picks)

                # Check uniqueness
                self.assertEqual(len(set(prediction)), num_picks)

                # Check number range
                self.assertTrue(all(1 <= n <= 80 for n in prediction))

                # Check sorted order
                self.assertEqual(prediction, sorted(prediction))

    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        for strategy in [self.pattern_strategy, self.rule_strategy]:
            prediction = strategy.predict(draw_index=0)
            confidence = strategy.get_confidence(prediction)

            # Check confidence range
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)

            # Test confidence with invalid prediction
            invalid_prediction = [1, 2, 3, 4, 5]  # Too few numbers
            with self.assertRaises(ValueError):
                strategy.get_confidence(invalid_prediction)

    def test_pattern_strategy_specific(self):
        """Test pattern-specific functionality."""
        # Test pattern extraction
        self.assertGreater(len(self.pattern_strategy.patterns), 0)

        # Test pattern weights
        for pattern in self.pattern_strategy.patterns:
            self.assertGreaterEqual(pattern["support"], 0.0)
            self.assertLessEqual(pattern["support"], 1.0)

    def test_rule_strategy_specific(self):
        """Test rule-specific functionality."""
        # Test rule extraction
        self.assertGreater(len(self.rule_strategy.rules), 0)

        # Test rule quality
        for rule in self.rule_strategy.rules:
            self.assertGreaterEqual(rule["lift"], 1.0)
            self.assertGreaterEqual(rule["confidence"], 0.0)
            self.assertLessEqual(rule["confidence"], 1.0)

    def test_strategy_fitting(self):
        """Test strategy fitting with different data sizes."""
        for strategy in [self.pattern_strategy, self.rule_strategy]:
            # Test with minimal data
            minimal_data = sample_data[:2]
            strategy.fit(minimal_data)
            prediction = strategy.predict(draw_index=0)
            self.assertEqual(len(prediction), 20)

            # Test with larger dataset
            larger_data = sample_data * 3
            strategy.fit(larger_data)
            prediction = strategy.predict(draw_index=0)
            self.assertEqual(len(prediction), 20)

    def test_invalid_configurations(self):
        """Test invalid configuration handling."""
        # Test pattern strategy invalid config
        with self.assertRaises(ValueError):
            PatternBasedStrategy(config={"min_support": 2.0})  # Invalid support

        # Test rule strategy invalid config
        with self.assertRaises(ValueError):
            RuleBasedStrategy(config={"min_lift": 0.5})  # Invalid lift

    def test_prediction_consistency(self):
        """Test prediction consistency across multiple calls."""
        for strategy in [self.pattern_strategy, self.rule_strategy]:
            # Generate multiple predictions
            predictions = [strategy.predict(draw_index=0) for _ in range(5)]

            # Check that predictions are valid
            for pred in predictions:
                self.assertEqual(len(pred), 20)
                self.assertEqual(len(set(pred)), 20)
                self.assertTrue(all(1 <= n <= 80 for n in pred))

            # Check that predictions are different (due to randomization)
            unique_predictions = set(tuple(pred) for pred in predictions)
            self.assertGreater(len(unique_predictions), 1)


if __name__ == "__main__":
    unittest.main()
