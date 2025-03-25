import requests
import time
import logging
import json
from datetime import datetime, timezone
import sys
import os

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(log_dir, 'test_monitoring.log'))
    ]
)
logger = logging.getLogger(__name__)

def test_metrics_endpoint():
    """Test if metrics endpoint is accessible and returning data"""
    try:
        response = requests.get('http://localhost:8000/metrics', timeout=5)
        response.raise_for_status()
        
        # Verify we have our custom metrics
        metrics = response.text
        required_metrics = [
            'http_requests_total',
            'http_request_duration_seconds',
            'memory_usage_bytes'
        ]
        
        for metric in required_metrics:
            if metric not in metrics:
                logger.error(f"Required metric '{metric}' not found in response")
                return False
                
        logger.info("Metrics endpoint test passed")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Metrics endpoint test failed: {str(e)}")
        return False

def test_alert_receiver():
    """Test if alert receiver is working by sending a test alert"""
    try:
        # First check health endpoint
        health_response = requests.get('http://localhost:3456/health', timeout=5)
        health_response.raise_for_status()
        
        # Send test alert
        test_alert = {
            "alerts": [{
                "status": "firing",
                "labels": {
                    "severity": "critical",
                    "alertname": "TestAlert"
                },
                "annotations": {
                    "description": "This is a test alert"
                },
                "startsAt": datetime.now(timezone.utc).isoformat()
            }]
        }
        
        response = requests.post(
            'http://localhost:3456/alert',
            json=test_alert,
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        response.raise_for_status()
        
        # Verify response
        if response.json().get('status') != 'success':
            logger.error("Alert receiver returned non-success status")
            return False
            
        logger.info("Alert receiver test passed")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Alert receiver test failed: {str(e)}")
        return False

def main():
    """Run all monitoring tests"""
    logger.info("Starting monitoring tests")
    
    # Wait for services to be fully up
    time.sleep(2)
    
    tests = [
        ("Metrics Endpoint", test_metrics_endpoint),
        ("Alert Receiver", test_alert_receiver)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        logger.info(f"Running {test_name} test...")
        if not test_func():
            all_passed = False
            logger.error(f"{test_name} test failed")
    
    if all_passed:
        logger.info("All monitoring tests passed!")
        sys.exit(0)
    else:
        logger.error("Some monitoring tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 