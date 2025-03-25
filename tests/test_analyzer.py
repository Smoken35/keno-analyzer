"""
Tests for the KenoAnalyzer class.
"""

import pytest
import numpy as np
from keno.analysis.analyzer import KenoAnalyzer

@pytest.fixture
def sample_data():
    """Create sample Keno data for testing."""
    return [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    ]

@pytest.fixture
def test_analyzer(sample_data):
    """Create a KenoAnalyzer instance with sample data."""
    analyzer = KenoAnalyzer("dummy_path")
    analyzer.data = sample_data
    return analyzer

class TestKenoAnalyzer:
    """Test cases for KenoAnalyzer class."""
    
    def test_analyze_frequency(self, test_analyzer):
        """Test frequency analysis."""
        freq = test_analyzer.analyze_frequency()
        
        # Check that all numbers from 1 to 22 are present
        assert all(i in freq for i in range(1, 23))
        
        # Check frequencies
        assert freq[1] == 1  # Only in first draw
        assert freq[2] == 2  # In first and second draws
        assert freq[3] == 3  # In all three draws
        assert freq[21] == 2  # In second and third draws
        assert freq[22] == 1  # Only in third draw

    def test_analyze_patterns(self, test_analyzer):
        """Test pattern analysis."""
        patterns = test_analyzer.analyze_patterns(window=2)
        
        # Check that we get the expected number of hot and cold numbers
        assert len(patterns['hot_numbers']) == 20
        assert len(patterns['cold_numbers']) == 20
        
        # Check that numbers are sorted
        assert patterns['hot_numbers'] == sorted(patterns['hot_numbers'])
        assert patterns['cold_numbers'] == sorted(patterns['cold_numbers'])
        
        # Check that numbers that appear in both recent draws are hot
        assert all(num in patterns['hot_numbers'] for num in range(3, 21))

    def test_analyze_due_numbers(self, test_analyzer):
        """Test due number analysis."""
        due_numbers = test_analyzer.analyze_due_numbers()
        
        # Check that numbers are sorted by due score
        scores = [score for _, score in due_numbers]
        assert scores == sorted(scores, reverse=True)
        
        # Check that all numbers 1-80 are included
        numbers = [num for num, _ in due_numbers]
        assert len(numbers) == 80
        assert all(1 <= num <= 80 for num in numbers)
        
        # Check that numbers that appear in all draws have lower scores
        assert any(num == 3 and score < 0.5 for num, score in due_numbers)

    def test_predict_next_draw(self, test_analyzer):
        """Test next draw prediction."""
        # Test frequency-based prediction
        freq_pred = test_analyzer.predict_next_draw('frequency', picks=5)
        assert len(freq_pred) == 5
        assert all(isinstance(x, int) for x in freq_pred)
        
        # Test pattern-based prediction
        pattern_pred = test_analyzer.predict_next_draw('patterns', picks=5)
        assert len(pattern_pred) == 5
        assert all(isinstance(x, int) for x in pattern_pred)
        
        # Test due number prediction
        due_pred = test_analyzer.predict_next_draw('due', picks=5)
        assert len(due_pred) == 5
        assert all(isinstance(x, int) for x in due_pred)
        
        # Test invalid method
        with pytest.raises(ValueError):
            test_analyzer.predict_next_draw('invalid_method', picks=5)

    def test_calculate_expected_value(self, test_analyzer):
        """Test expected value calculation."""
        # Set up a simple payout table
        payout_table = {
            5: {0: 0, 1: 1, 2: 5, 3: 20, 4: 100, 5: 500}
        }
        test_analyzer.set_payout_table(payout_table)
        
        # Calculate expected value
        ev = test_analyzer.calculate_expected_value(pick_size=5)
        assert isinstance(ev, float)
        assert ev >= 0  # Expected value should be non-negative
        
        # Test with invalid pick size
        ev = test_analyzer.calculate_expected_value(pick_size=6)
        assert ev == 0.0  # Should return 0 for invalid pick size 