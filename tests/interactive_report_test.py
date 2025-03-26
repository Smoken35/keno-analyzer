"""Tests for interactive report generation."""

import os
import tempfile
from typing import List

import pytest

from src.keno.scripts.interactive_report import analyze_patterns, calculate_trend, generate_report


def test_calculate_trend():
    """Test trend calculation."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    trend = calculate_trend(data, window=3)
    assert len(trend) == len(data)
    assert trend[-1] == 4.0  # Last 3 numbers average


def test_analyze_patterns():
    """Test pattern analysis."""
    data = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    patterns = analyze_patterns(data, window=3)
    assert len(patterns) > 0
    assert "(1, 2, 3)" in patterns


def test_generate_report():
    """Test report generation."""
    data = list(range(1, 81))
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as temp:
        generate_report(data, temp.name)
        assert os.path.exists(temp.name)
        assert os.path.getsize(temp.name) > 0
        os.unlink(temp.name)


def test_empty_data():
    """Test handling of empty data."""
    with pytest.raises(ValueError):
        generate_report([])


def test_invalid_data():
    """Test handling of invalid data."""
    with pytest.raises(ValueError):
        generate_report([0, 81])  # Numbers outside valid range
