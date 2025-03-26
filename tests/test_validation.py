import json
import os
import tempfile
from datetime import date, datetime

import pytest

from keno.validation.tracker import ValidationTracker


@pytest.fixture(scope="function")
def tracker():
    """Create a ValidationTracker instance for testing."""
    temp_dir = tempfile.mkdtemp()
    tracker = ValidationTracker(storage_dir=temp_dir)
    yield tracker
    # Cleanup
    import shutil

    shutil.rmtree(temp_dir)


def test_record_prediction(tracker):
    """Test recording a new prediction."""
    method = "test_method"
    predicted_numbers = [1, 5, 10, 15, 20]
    draw_date = date(2024, 3, 15)
    metadata = {"confidence": 0.8}

    # Record prediction
    pred_id = tracker.record_prediction(
        method=method, predicted_numbers=predicted_numbers, draw_date=draw_date, metadata=metadata
    )

    assert isinstance(pred_id, str)
    assert len(pred_id) > 0


def test_validate_prediction(tracker):
    """Test validating a prediction."""
    method = "test_method"
    pred_numbers = [1, 5, 10, 15, 20]
    actual_numbers = [1, 5, 7, 15, 25]

    # Record and validate
    pred_id = tracker.record_prediction(method=method, predicted_numbers=pred_numbers)
    validation_result = tracker.validate_prediction(pred_id, actual_numbers)

    assert isinstance(validation_result, dict)
    assert "accuracy" in validation_result
    assert 0 <= validation_result["accuracy"] <= 1


def test_method_performance(tracker):
    """Test retrieving method performance statistics."""
    method = "test_method"

    # Record multiple predictions
    numbers = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    actuals = [[1, 2, 4], [4, 5, 7], [7, 8, 9]]

    for pred, actual in zip(numbers, actuals):
        pred_id = tracker.record_prediction(method=method, predicted_numbers=pred)
        tracker.validate_prediction(pred_id, actual)

    # Get performance stats
    stats = tracker.get_method_performance(method)
    assert isinstance(stats, dict)
    assert "accuracy" in stats
    assert "predictions" in stats


def test_bulk_validation(tracker):
    """Test bulk validation of predictions."""
    method = "test_method"
    predictions = [([1, 2, 3], date(2024, 3, 15)), ([4, 5, 6], date(2024, 3, 16))]

    # Record predictions
    pred_ids = []
    for numbers, draw_date in predictions:
        pred_id = tracker.record_prediction(
            method=method, predicted_numbers=numbers, draw_date=draw_date
        )
        pred_ids.append(pred_id)

    # Bulk validate
    actuals = {date(2024, 3, 15): [1, 2, 4], date(2024, 3, 16): [4, 5, 7]}
    results = tracker.bulk_validate(actuals)

    assert isinstance(results, dict)
    assert len(results) == len(predictions)


def test_generate_report(tracker):
    """Test report generation."""
    method = "test_method"

    # Record and validate some predictions
    for i in range(5):
        pred_id = tracker.record_prediction(
            method=method, predicted_numbers=[1, 2, 3], draw_date=date(2024, 3, 15 + i)
        )
        tracker.validate_prediction(pred_id, [1, 2, 4])

    # Generate report
    report = tracker.generate_report(method)
    assert isinstance(report, dict)
    assert "accuracy" in report
    assert "trend" in report
