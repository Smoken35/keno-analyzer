import os
import json
import pytest
import tempfile
from datetime import datetime, date
from keno.validation.tracker import ValidationTracker

@pytest.fixture
def tracker():
    """Create a temporary ValidationTracker instance for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    tracker = ValidationTracker(db_path=db_path)
    yield tracker
    
    # Cleanup
    os.unlink(db_path)

def test_record_prediction(tracker):
    """Test recording a new prediction."""
    method = "test_method"
    numbers = [1, 5, 10, 15, 20]
    draw_date = date(2024, 3, 15)
    metadata = {"confidence": 0.8}
    
    # Record prediction
    pred_id = tracker.record_prediction(
        method=method,
        numbers=numbers,
        draw_date=draw_date,
        metadata=metadata
    )
    
    # Verify recorded prediction
    pred = tracker.get_prediction(pred_id)
    assert pred['method'] == method
    assert json.loads(pred['predicted_numbers']) == numbers
    assert pred['draw_date'] == draw_date.isoformat()
    assert json.loads(pred['metadata']) == metadata

def test_validate_prediction(tracker):
    """Test validating a prediction."""
    method = "test_method"
    pred_numbers = [1, 5, 10, 15, 20]
    actual_numbers = [1, 5, 7, 15, 25]
    
    # Record and validate
    pred_id = tracker.record_prediction(method=method, numbers=pred_numbers)
    tracker.validate_prediction(pred_id, actual_numbers)
    
    # Check validation results
    pred = tracker.get_prediction(pred_id)
    assert json.loads(pred['draw_numbers']) == actual_numbers
    assert pred['matches'] == 3  # 1, 5, and 15 match
    assert pred['accuracy'] == pytest.approx(0.6)  # 3/5 = 0.6

def test_method_performance(tracker):
    """Test retrieving method performance statistics."""
    method = "test_method"
    
    # Record multiple predictions
    numbers = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    actuals = [[1, 2, 4], [4, 5, 7], [7, 8, 9]]
    
    for pred, actual in zip(numbers, actuals):
        pred_id = tracker.record_prediction(method=method, numbers=pred)
        tracker.validate_prediction(pred_id, actual)
    
    # Check performance stats
    stats = tracker.get_method_performance(method)
    assert stats['total_predictions'] == 3
    assert stats['average_accuracy'] == pytest.approx(0.778, abs=0.01)  # (2/3 + 2/3 + 3/3) / 3

def test_bulk_validation(tracker):
    """Test bulk validation of predictions."""
    method = "test_method"
    predictions = [
        ([1, 2, 3], date(2024, 3, 15)),
        ([4, 5, 6], date(2024, 3, 16))
    ]
    
    # Record predictions
    pred_ids = []
    for numbers, draw_date in predictions:
        pred_id = tracker.record_prediction(
            method=method,
            numbers=numbers,
            draw_date=draw_date
        )
        pred_ids.append(pred_id)
    
    # Bulk validate
    validation_data = {
        date(2024, 3, 15).isoformat(): [1, 2, 4],
        date(2024, 3, 16).isoformat(): [4, 5, 7]
    }
    tracker.bulk_validate(validation_data)
    
    # Check results
    for pred_id in pred_ids:
        pred = tracker.get_prediction(pred_id)
        assert pred['draw_numbers'] is not None
        assert pred['accuracy'] is not None

def test_generate_report(tracker):
    """Test report generation."""
    method = "test_method"
    
    # Record and validate some predictions
    for i in range(5):
        pred_id = tracker.record_prediction(
            method=method,
            numbers=[1, 2, 3],
            draw_date=date(2024, 3, 15 + i)
        )
        tracker.validate_prediction(pred_id, [1, 2, 4])
    
    # Generate report
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        report_path = f.name
    
    tracker.generate_report(output_file=report_path)
    
    # Verify report contents
    with open(report_path) as f:
        report = json.load(f)
    
    assert method in report['methods']
    assert report['methods'][method]['total_predictions'] == 5
    assert 'accuracy_trend' in report['methods'][method]
    
    # Cleanup
    os.unlink(report_path) 