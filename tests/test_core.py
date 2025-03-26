"""Tests for core Keno analysis functionality."""

import numpy as np
import pytest

from keno.analysis.analyzer import KenoAnalyzer


@pytest.fixture
def sample_data():
    """Create sample Keno data for testing."""
    return [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    ]


@pytest.fixture
def analyzer(sample_data):
    """Create a KenoAnalyzer instance with sample data."""
    return KenoAnalyzer(sample_data)


def test_analyze_frequency(analyzer):
    """Test frequency analysis of numbers."""
    freq = analyzer.analyze_frequency()

    # Check that all numbers are present
    assert len(freq) == 80
    assert all(1 <= num <= 80 for num in freq.keys())

    # Check that frequencies are non-negative
    assert all(count >= 0 for count in freq.values())

    # Check specific frequencies from sample data
    assert freq[1] == 1
    assert freq[2] == 2
    assert freq[3] == 3
    assert freq[21] == 2
    assert freq[22] == 1


def test_analyze_patterns(analyzer):
    """Test pattern analysis of numbers."""
    patterns = analyzer.analyze_patterns()

    # Check structure
    assert "hot_numbers" in patterns
    assert "cold_numbers" in patterns

    # Check lengths
    assert len(patterns["hot_numbers"]) == 20
    assert len(patterns["cold_numbers"]) == 20

    # Check sorting
    assert patterns["hot_numbers"] == sorted(patterns["hot_numbers"])
    assert patterns["cold_numbers"] == sorted(patterns["cold_numbers"])

    # Check ranges
    assert all(1 <= num <= 80 for num in patterns["hot_numbers"])
    assert all(1 <= num <= 80 for num in patterns["cold_numbers"])


def test_analyze_due_numbers(analyzer):
    """Test due number analysis."""
    due_nums = analyzer.analyze_due_numbers()

    # Check structure
    assert len(due_nums) == 80
    assert all(isinstance(item, tuple) and len(item) == 2 for item in due_nums)

    # Check sorting
    assert due_nums == sorted(due_nums, key=lambda x: x[1], reverse=True)

    # Check ranges
    assert all(1 <= num <= 80 for num, _ in due_nums)
    assert all(0 <= score <= 1 for _, score in due_nums)


def test_predict_next_draw(analyzer):
    """Test prediction methods."""
    # Test frequency method
    pred_freq = analyzer.predict_next_draw(method="frequency", picks=10)
    assert len(pred_freq) == 10
    assert all(1 <= num <= 80 for num in pred_freq)
    assert pred_freq == sorted(pred_freq)

    # Test patterns method
    pred_patterns = analyzer.predict_next_draw(method="patterns", picks=10)
    assert len(pred_patterns) == 10
    assert all(1 <= num <= 80 for num in pred_patterns)
    assert pred_patterns == sorted(pred_patterns)

    # Test due method
    pred_due = analyzer.predict_next_draw(method="due", picks=10)
    assert len(pred_due) == 10
    assert all(1 <= num <= 80 for num in pred_due)
    assert pred_due == sorted(pred_due)

    # Test markov method
    pred_markov = analyzer.predict_next_draw(method="markov", picks=10)
    assert len(pred_markov) == 10
    assert all(1 <= num <= 80 for num in pred_markov)
    assert pred_markov == sorted(pred_markov)


def test_predict_next_draw_validation(analyzer):
    """Test prediction method validation."""
    # Test invalid picks
    with pytest.raises(ValueError, match="picks must be an integer between 1 and 20"):
        analyzer.predict_next_draw(picks=0)
    with pytest.raises(ValueError, match="picks must be an integer between 1 and 20"):
        analyzer.predict_next_draw(picks=21)

    # Test invalid method
    with pytest.raises(ValueError, match="Invalid prediction method"):
        analyzer.predict_next_draw(method="invalid")


def test_empty_data():
    """Test behavior with empty data."""
    analyzer = KenoAnalyzer([])

    # Test frequency analysis
    freq = analyzer.analyze_frequency()
    assert len(freq) == 80
    assert all(count == 0 for count in freq.values())

    # Test pattern analysis
    patterns = analyzer.analyze_patterns()
    assert len(patterns["hot_numbers"]) == 0
    assert len(patterns["cold_numbers"]) == 0

    # Test due numbers
    due_nums = analyzer.analyze_due_numbers()
    assert len(due_nums) == 80
    assert all(score == 0.0 for _, score in due_nums)

    # Test predictions
    for method in ["frequency", "patterns", "due", "markov"]:
        pred = analyzer.predict_next_draw(method=method, picks=10)
        assert len(pred) == 10
        assert all(1 <= num <= 80 for num in pred)
        assert pred == sorted(pred)
