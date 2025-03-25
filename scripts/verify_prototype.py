#!/usr/bin/env python3
"""
Verification script for the Keno Prediction Tool prototype.
Runs all tests and verifies core functionality.
"""

import os
import sys
import json
import unittest
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from keno import KenoAnalyzer, ValidationTracker, KenoVisualizer
import numpy as np
from scipy import stats

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_test_environment():
    """Set up the test environment and directories."""
    test_dirs = [
        '~/.keno/test/data',
        '~/.keno/test/cache',
        '~/.keno/test/validation',
        '~/.keno/test/visualizations',
        '~/.keno/test/reports'
    ]
    
    for directory in test_dirs:
        dir_path = os.path.expanduser(directory)
        os.makedirs(dir_path, exist_ok=True)
        
    logger.info("Test environment set up successfully")
    return os.path.expanduser('~/.keno/test')

def run_unit_tests():
    """Run all unit tests and return results."""
    logger.info("Running unit tests...")
    
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

def verify_core_functionality():
    """Verify core functionality of the Keno Prediction Tool."""
    logger.info("Verifying core functionality...")
    
    try:
        # Initialize components
        analyzer = KenoAnalyzer(data_source="sample")
        
        # Generate sample data for testing
        numbers = [
            sorted(np.random.choice(range(1, 81), size=20, replace=False))
            for _ in range(100)
        ]
        analyzer.data = numbers
        
        # Set up payout table
        payout_table = {
            4: {  # Pick size of 4
                4: 75.0,  # All 4 correct
                3: 5.0,   # 3 correct
                2: 1.0,   # 2 correct
                1: 0.0,   # 1 correct
                0: 0.0    # None correct
            }
        }
        analyzer.payout_table = payout_table
        
        # Test prediction methods
        methods = ['frequency', 'patterns', 'markov', 'due']
        predictions = {}
        
        for method in methods:
            predictions[method] = analyzer.predict_next_draw(method=method, pick_size=4)
            if not predictions[method] or len(predictions[method]) != 4:
                raise ValueError(f"Invalid prediction length from {method} method")
            if not all(1 <= n <= 80 for n in predictions[method]):
                raise ValueError(f"Invalid number range in prediction from {method} method")
                
        logger.info("Core functionality verified successfully")
        return True
        
    except Exception as e:
        logger.error(f"Core functionality verification failed: {str(e)}")
        return False

def check_data_integrity():
    """Check data integrity and consistency."""
    logger.info("Checking data integrity...")
    
    try:
        analyzer = KenoAnalyzer(data_source="sample")
        
        # Generate sample data for testing
        numbers = [
            sorted(np.random.choice(range(1, 81), size=20, replace=False))
            for _ in range(30)
        ]
        analyzer.data = numbers
        
        # Check data structure
        if not isinstance(analyzer.data, list):
            raise ValueError("Data is not a list")
            
        # Check data types
        if not all(isinstance(nums, list) for nums in analyzer.data):
            raise ValueError("Invalid data type in numbers")
            
        # Check number ranges
        if not all(all(1 <= n <= 80 for n in nums) for nums in analyzer.data):
            raise ValueError("Numbers out of valid range")
            
        logger.info("Data integrity check passed")
        return True
        
    except Exception as e:
        logger.error(f"Data integrity check failed: {str(e)}")
        return False

def verify_performance_metrics():
    """Verify performance metrics and generate report."""
    logger.info("Verifying performance metrics...")
    
    try:
        # Initialize components
        analyzer = KenoAnalyzer(data_source="sample")
        
        # Generate sample data for testing
        numbers = [
            sorted(np.random.choice(range(1, 81), size=20, replace=False))
            for _ in range(100)
        ]
        analyzer.data = numbers
        
        # Set up payout table
        payout_table = {
            4: {  # Pick size of 4
                4: 75.0,  # All 4 correct
                3: 5.0,   # 3 correct
                2: 1.0,   # 2 correct
                1: 0.0,   # 1 correct
                0: 0.0    # None correct
            }
        }
        analyzer.payout_table = payout_table
        
        # Run performance simulation
        methods = ['frequency', 'patterns', 'markov', 'due']
        results = {}
        
        for method in methods:
            # Run strategy simulation
            simulation = analyzer.calculate_expected_value(
                pick_size=4,
                method=method
            )
            
            # Analyze historical performance
            historical_results = []
            for i in range(len(analyzer.data) - 1):
                prediction = analyzer.predict_next_draw(method=method, pick_size=4)
                actual = analyzer.data[i + 1]
                matches = len(set(prediction) & set(actual))
                historical_results.append(matches)
            
            avg_matches = np.mean(historical_results)
            p_value = stats.ttest_1samp(historical_results, 2.0)[1]  # Test against random expectation
            
            results[method] = {
                'simulation': {
                    'roi_percent': (simulation - 1.0) * 100,
                    'num_simulations': 100
                },
                'validation': {
                    'avg_matches': float(avg_matches),
                    'p_value': float(p_value)
                }
            }
        
        # Generate performance report
        report = {
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'metrics': {
                method: {
                    'roi_percent': result['simulation']['roi_percent'],
                    'avg_matches': result['validation']['avg_matches'],
                    'p_value': result['validation']['p_value'],
                    'num_simulations': 100
                }
                for method, result in results.items()
            }
        }
        
        # Ensure the reports directory exists
        os.makedirs(os.path.expanduser('~/.keno/test/reports'), exist_ok=True)
        
        # Save report
        report_path = os.path.expanduser('~/.keno/test/reports/performance_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
            
        logger.info("Performance metrics verified successfully")
        return True
        
    except Exception as e:
        logger.error(f"Performance metrics verification failed: {str(e)}")
        return False

def generate_verification_report(results):
    """Generate final verification report."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'status': all(results.values()),
        'results': results,
        'recommendations': []
    }
    
    # Add recommendations based on results
    if not results['unit_tests']:
        report['recommendations'].append(
            "Fix failing unit tests before proceeding"
        )
    if not results['core_functionality']:
        report['recommendations'].append(
            "Review and fix core functionality issues"
        )
    if not results['data_integrity']:
        report['recommendations'].append(
            "Address data integrity concerns"
        )
    if not results['performance_metrics']:
        report['recommendations'].append(
            "Investigate performance metrics issues"
        )
        
    # Save report
    report_path = os.path.expanduser('~/.keno/test/verification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    return report

def main():
    """Main verification process."""
    try:
        # Set up test environment
        test_dir = setup_test_environment()
        
        # Run verifications
        results = {
            'unit_tests': run_unit_tests(),
            'core_functionality': verify_core_functionality(),
            'data_integrity': check_data_integrity(),
            'performance_metrics': verify_performance_metrics()
        }
        
        # Generate report
        report = generate_verification_report(results)
        
        # Log results
        logger.info("\nVerification Results:")
        logger.info("-" * 50)
        for check, passed in results.items():
            status = "PASSED" if passed else "FAILED"
            logger.info(f"{check:.<30}{status}")
        logger.info("-" * 50)
        logger.info(f"Overall Status: {'PASSED' if report['status'] else 'FAILED'}")
        
        if report['recommendations']:
            logger.info("\nRecommendations:")
            for rec in report['recommendations']:
                logger.info(f"- {rec}")
                
        return 0 if report['status'] else 1
        
    except Exception as e:
        logger.error(f"Verification process failed: {str(e)}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 