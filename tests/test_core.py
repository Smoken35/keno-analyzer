"""
Core functionality tests for the Keno Prediction Tool.
"""

import unittest
import os
import numpy as np
from keno.analysis.analyzer import KenoAnalyzer
from keno.validation.tracker import ValidationTracker
from keno.visualization.visualizer import KenoVisualizer

class TestKenoCore(unittest.TestCase):
    """Test cases for core Keno functionality."""
    
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
        
        self.tracker = ValidationTracker()
        self.visualizer = KenoVisualizer(self.analyzer)
        
    def test_data_generation(self):
        """Test data generation and validation."""
        # Check data structure
        self.assertIsNotNone(self.analyzer.data)
        self.assertIsInstance(self.analyzer.data, list)
        
        # Check data length
        self.assertEqual(len(self.analyzer.data), 30)
        
        # Check data types
        self.assertTrue(all(isinstance(nums, list) for nums in self.analyzer.data))
        
    def test_number_range(self):
        """Test number range validation."""
        for numbers in self.analyzer.data:
            # Check number range
            self.assertTrue(all(1 <= n <= 80 for n in numbers))
            
            # Check for duplicates
            self.assertEqual(len(numbers), len(set(numbers)))
            
            # Check length
            self.assertEqual(len(numbers), 20)
            
    def test_frequency_analysis(self):
        """Test frequency analysis method."""
        freq = self.analyzer.analyze_frequency()
        
        # Check structure
        self.assertIsInstance(freq, dict)
        self.assertTrue(all(1 <= num <= 80 for num in freq.keys()))
        
        # Check values
        for num, count in freq.items():
            self.assertIsInstance(num, int)
            self.assertIsInstance(count, int)
            self.assertGreaterEqual(count, 0)
            
    def test_pattern_analysis(self):
        """Test pattern analysis methods."""
        patterns = self.analyzer.analyze_patterns()
        
        # Check pattern structure
        self.assertIn('hot_numbers', patterns)
        self.assertIn('cold_numbers', patterns)
        
        # Check pattern content
        self.assertEqual(len(patterns['hot_numbers']), 20)
        self.assertEqual(len(patterns['cold_numbers']), 20)
        
        # Check number ranges
        self.assertTrue(all(1 <= n <= 80 for n in patterns['hot_numbers']))
        self.assertTrue(all(1 <= n <= 80 for n in patterns['cold_numbers']))
        
        # Check for duplicates
        self.assertEqual(len(set(patterns['hot_numbers'])), len(patterns['hot_numbers']))
        self.assertEqual(len(set(patterns['cold_numbers'])), len(patterns['cold_numbers']))
        
    def test_prediction_methods(self):
        """Test various prediction methods."""
        methods = ['frequency', 'patterns', 'markov', 'due']
        
        for method in methods:
            predictions = self.analyzer.predict_next_draw(method)
            
            # Check structure
            self.assertIsInstance(predictions, list)
            self.assertEqual(len(predictions), 20)
            
            # Check values
            self.assertTrue(all(1 <= num <= 80 for num in predictions))
            self.assertEqual(len(set(predictions)), 20)  # No duplicates
            
    def test_expected_value(self):
        """Test expected value calculations."""
        # Test with standard payout table
        ev = self.analyzer.calculate_expected_value(
            pick_size=4,
            method='frequency'
        )
        
        # Check value is reasonable
        self.assertIsInstance(ev, float)
        self.assertTrue(-1 <= ev <= 1)  # EV should be between -1 and 1 for typical payout tables
        
    def test_strategy_simulation(self):
        """Test strategy simulation."""
        # Set up payout table for pick size 4
        self.analyzer.set_payout_table({
            4: {
                4: 75.0,  # All 4 correct
                3: 5.0,   # 3 correct
                2: 1.0,   # 2 correct
                1: 0.0,   # 1 correct
                0: 0.0    # None correct
            }
        })
        
        # Run simulation
        result = self.analyzer.simulate_strategy(
            method='frequency',
            pick_size=4,
            bet_size=1.0,
            num_simulations=100
        )
        
        # Check structure of results
        self.assertIn('total_return', result)
        self.assertIn('roi_percent', result)
        self.assertIn('match_distribution', result)
        self.assertIn('num_simulations', result)
        
        # Check match distribution
        self.assertEqual(
            sum(result['match_distribution'].values()),
            result['num_simulations']
        )
        
        # Check ROI calculation
        expected_roi = (result['total_return'] / (result['num_simulations'])) * 100
        self.assertAlmostEqual(result['roi_percent'], expected_roi, places=2)
        
    def test_skip_and_hit_patterns(self):
        """Test skip and hit pattern analysis."""
        patterns = self.analyzer.analyze_skip_and_hit_patterns()
        
        # Check structure
        self.assertIsInstance(patterns, dict)
        self.assertTrue(all(1 <= num <= 80 for num in patterns.keys()))
        
        # Check values
        for num, stats in patterns.items():
            self.assertIn('avg_skip', stats)
            self.assertIn('max_skip', stats)
            self.assertIn('hit_rate', stats)
            self.assertTrue(0 <= stats['hit_rate'] <= 1)
            
    def test_cyclic_patterns(self):
        """Test cyclic pattern analysis."""
        cycles = self.analyzer.analyze_cyclic_patterns()
        
        # Check structure
        self.assertIsInstance(cycles, dict)
        
        # Check values
        total_prob = sum(cycles.values())
        self.assertAlmostEqual(total_prob, 1.0, places=5)
        
    def test_markov_chain_analysis(self):
        """Test Markov chain analysis."""
        matrix = self.analyzer.build_transition_matrix()
        
        # Check dimensions
        self.assertEqual(matrix.shape, (80, 80))
        
        # Check probabilities
        for row in matrix:
            if np.sum(row) > 0:  # Skip rows with all zeros
                self.assertAlmostEqual(np.sum(row), 1.0, places=5)
                
    def test_due_theory_analysis(self):
        """Test due theory analysis."""
        due_numbers = self.analyzer.analyze_due_numbers()
        
        # Check structure
        self.assertIsInstance(due_numbers, list)
        self.assertTrue(all(isinstance(item, tuple) for item in due_numbers))
        
        # Check values
        for num, score in due_numbers:
            self.assertTrue(1 <= num <= 80)
            self.assertTrue(0 <= score <= 1)
            
    def test_validation_workflow(self):
        """Test complete validation workflow."""
        # Record prediction
        prediction_id = self.tracker.record_prediction(
            method='frequency',
            predicted_numbers=self.analyzer.predict_next_draw('frequency')
        )
        
        # Validate prediction
        actual_numbers = sorted(np.random.choice(range(1, 81), size=20, replace=False))
        result = self.tracker.validate_prediction(prediction_id, actual_numbers)
        
        # Check validation result
        self.assertIsInstance(result, dict)
        self.assertIn('matches', result)
        self.assertIn('accuracy', result)
        
    def test_visualization_workflow(self):
        """Test complete visualization workflow."""
        # Create visualization directory if it doesn't exist
        vis_dir = os.path.join(os.path.dirname(__file__), 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generate visualizations
        heatmap_path = os.path.join(vis_dir, 'test_heatmap.png')
        pattern_path = os.path.join(vis_dir, 'test_patterns.png')
        prediction_path = os.path.join(vis_dir, 'test_predictions.png')
        
        self.visualizer.plot_frequency_heatmap(heatmap_path)
        self.visualizer.plot_pattern_analysis(pattern_path)
        self.visualizer.plot_prediction_comparison(4, 50, prediction_path)
        
        # Check file creation
        self.assertTrue(os.path.exists(heatmap_path))
        self.assertTrue(os.path.exists(pattern_path))
        self.assertTrue(os.path.exists(prediction_path))

if __name__ == '__main__':
    unittest.main() 