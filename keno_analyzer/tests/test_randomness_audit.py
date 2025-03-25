#!/usr/bin/env python3
"""
Unit tests for Keno randomness audit functionality.
"""

import unittest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

from scripts.randomness_audit import KenoRandomnessAuditor
from utils.stat_tests import KenoRandomnessTester

class TestKenoRandomnessAudit(unittest.TestCase):
    """Test cases for Keno randomness audit functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / 'test_keno.db'
        self.report_path = Path(self.temp_dir) / 'test_report.json'
        
        # Initialize test data
        self._create_test_database()
        self._create_test_report()
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary files
        if self.db_path.exists():
            self.db_path.unlink()
        if self.report_path.exists():
            self.report_path.unlink()
        if Path(self.temp_dir).exists():
            Path(self.temp_dir).rmdir()
    
    def _create_test_database(self):
        """Create a test database with sample data."""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE games (
                id INTEGER PRIMARY KEY,
                numbers TEXT NOT NULL,
                draw_date DATE NOT NULL
            )
        ''')
        
        # Insert test data
        test_data = [
            (json.dumps(list(range(1, 21))), '2024-01-01'),
            (json.dumps(list(range(21, 41))), '2024-01-02'),
            (json.dumps(list(range(41, 61))), '2024-01-03'),
            (json.dumps(list(range(61, 81))), '2024-01-04')
        ]
        
        cursor.executemany(
            'INSERT INTO games (numbers, draw_date) VALUES (?, ?)',
            test_data
        )
        
        conn.commit()
        conn.close()
    
    def _create_test_report(self):
        """Create a test report with sample data."""
        report = {
            'entropy': {
                'score': 0.85,
                'window_entropies': [6.2, 6.3, 6.1, 6.4],
                'overrepresented': [1, 2],
                'underrepresented': [79, 80]
            },
            'chi_square': {
                'score': 0.9,
                'statistic': 85.5,
                'p_value': 0.15,
                'dates': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
                'p_values': [0.2, 0.18, 0.15, 0.12]
            },
            'temporal_drift': {
                'score': 0.95,
                'drift_scores': {
                    '2024-01-01': [0.1] * 80,
                    '2024-01-02': [0.2] * 80,
                    '2024-01-03': [0.3] * 80,
                    '2024-01-04': [0.4] * 80
                }
            },
            'autocorrelation': {
                'score': 0.88,
                'acf_values': np.random.rand(80, 10).tolist()
            },
            'overall_score': 0.895
        }
        
        with open(self.report_path, 'w') as f:
            json.dump(report, f)
    
    def test_auditor_initialization(self):
        """Test auditor initialization."""
        auditor = KenoRandomnessAuditor(str(self.db_path))
        self.assertIsNotNone(auditor.conn)
        self.assertIsNotNone(auditor.cursor)
        auditor.close()
    
    def test_load_draw_data(self):
        """Test loading draw data from database."""
        auditor = KenoRandomnessAuditor(str(self.db_path))
        
        # Test without date range
        numbers, dates = auditor.load_draw_data()
        self.assertEqual(len(numbers), 4)
        self.assertEqual(len(dates), 4)
        
        # Test with date range
        numbers, dates = auditor.load_draw_data(
            start_date='2024-01-01',
            end_date='2024-01-02'
        )
        self.assertEqual(len(numbers), 2)
        self.assertEqual(len(dates), 2)
        
        auditor.close()
    
    def test_run_audit(self):
        """Test running a complete audit."""
        auditor = KenoRandomnessAuditor(str(self.db_path))
        
        results = auditor.run_audit()
        
        # Check results structure
        self.assertIn('entropy', results)
        self.assertIn('chi_square', results)
        self.assertIn('temporal_drift', results)
        self.assertIn('autocorrelation', results)
        self.assertIn('overall_score', results)
        self.assertIn('metadata', results)
        
        # Check metadata
        self.assertEqual(results['metadata']['total_draws'], 4)
        self.assertEqual(results['metadata']['start_date'], '2024-01-01')
        self.assertEqual(results['metadata']['end_date'], '2024-01-04')
        
        auditor.close()
    
    def test_generate_report(self):
        """Test report generation."""
        auditor = KenoRandomnessAuditor(str(self.db_path))
        
        # Run audit and generate report
        results = auditor.run_audit()
        auditor.generate_report(results, str(self.report_path))
        
        # Verify report file exists
        self.assertTrue(self.report_path.exists())
        
        # Verify report contents
        with open(self.report_path, 'r') as f:
            saved_results = json.load(f)
        
        self.assertEqual(saved_results['overall_score'], results['overall_score'])
        
        auditor.close()
    
    def test_stat_tests(self):
        """Test statistical test functions."""
        tester = KenoRandomnessTester()
        
        # Test entropy calculation
        numbers = [list(range(1, 21))] * 10
        entropy_results = tester.calculate_entropy(numbers)
        self.assertIn('score', entropy_results)
        self.assertIn('window_entropies', entropy_results)
        
        # Test chi-square test
        chi_square_results = tester.chi_square_test(numbers)
        self.assertIn('statistic', chi_square_results)
        self.assertIn('p_value', chi_square_results)
        
        # Test temporal drift detection
        dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
                for i in range(10)]
        drift_results = tester.detect_temporal_drift(numbers, dates)
        self.assertIn('drift_scores', drift_results)
        
        # Test autocorrelation detection
        acf_results = tester.detect_autocorrelation(numbers)
        self.assertIn('acf_values', acf_results)
        
        # Test overall score calculation
        overall_score = tester.calculate_randomness_score(numbers, dates)
        self.assertIn('overall_score', overall_score)
        self.assertGreaterEqual(overall_score['overall_score'], 0)
        self.assertLessEqual(overall_score['overall_score'], 1)

if __name__ == '__main__':
    unittest.main() 