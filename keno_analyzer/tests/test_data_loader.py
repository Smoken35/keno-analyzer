"""
Unit tests for the Keno data loader.
"""

import unittest
import tempfile
import os
import json
import csv
from datetime import datetime
from pathlib import Path

from scripts.load_keno_data import KenoDataLoader

class TestKenoDataLoader(unittest.TestCase):
    """Test cases for KenoDataLoader."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_keno.db')
        
        # Create sample data files
        self.create_sample_csv()
        self.create_sample_json()
        
        # Initialize loader
        self.loader = KenoDataLoader(self.db_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.loader.close()
        # Clean up temporary files
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
    
    def create_sample_csv(self):
        """Create a sample CSV file with valid and invalid data."""
        csv_path = os.path.join(self.temp_dir, 'sample.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['draw_date', 'numbers'])
            # Valid row
            writer.writerow([
                '2024-03-24 12:00:00',
                '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20'
            ])
            # Invalid row (wrong number of numbers)
            writer.writerow([
                '2024-03-24 12:01:00',
                '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19'
            ])
            # Invalid row (number out of range)
            writer.writerow([
                '2024-03-24 12:02:00',
                '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,81'
            ])
    
    def create_sample_json(self):
        """Create a sample JSON file with valid and invalid data."""
        json_path = os.path.join(self.temp_dir, 'sample.json')
        data = [
            {
                'draw_date': '2024-03-24 12:03:00',
                'numbers': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            },
            {
                'draw_date': '2024-03-24 12:04:00',
                'numbers': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]  # Invalid
            },
            {
                'draw_date': '2024-03-24 12:05:00',
                'numbers': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,81]  # Invalid
            }
        ]
        with open(json_path, 'w') as f:
            json.dump(data, f)
    
    def test_database_setup(self):
        """Test database table creation."""
        # Check if tables exist
        self.cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name IN ('games', 'game_numbers')
        """)
        tables = {row[0] for row in self.cursor.fetchall()}
        self.assertEqual(tables, {'games', 'game_numbers'})
        
        # Check if indexes exist
        self.cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='index' AND name IN ('idx_draw_date', 'idx_number')
        """)
        indexes = {row[0] for row in self.cursor.fetchall()}
        self.assertEqual(indexes, {'idx_draw_date', 'idx_number'})
    
    def test_number_validation(self):
        """Test number validation logic."""
        # Valid numbers
        valid_numbers = list(range(1, 21))
        self.assertTrue(self.loader.validate_numbers(valid_numbers))
        
        # Invalid: wrong count
        invalid_count = list(range(1, 20))
        self.assertFalse(self.loader.validate_numbers(invalid_count))
        
        # Invalid: out of range
        invalid_range = list(range(1, 20)) + [81]
        self.assertFalse(self.loader.validate_numbers(invalid_range))
        
        # Invalid: duplicates
        invalid_duplicates = list(range(1, 20)) + [1]
        self.assertFalse(self.loader.validate_numbers(invalid_duplicates))
    
    def test_csv_loading(self):
        """Test loading data from CSV."""
        csv_path = os.path.join(self.temp_dir, 'sample.csv')
        self.loader.load_csv(csv_path)
        
        # Check if only valid row was loaded
        self.cursor.execute('SELECT COUNT(*) FROM games')
        count = self.cursor.fetchone()[0]
        self.assertEqual(count, 1)
        
        # Check if numbers were loaded correctly
        self.cursor.execute('SELECT numbers FROM games')
        numbers = json.loads(self.cursor.fetchone()[0])
        self.assertEqual(numbers, list(range(1, 21)))
    
    def test_json_loading(self):
        """Test loading data from JSON."""
        json_path = os.path.join(self.temp_dir, 'sample.json')
        self.loader.load_json(json_path)
        
        # Check if only valid row was loaded
        self.cursor.execute('SELECT COUNT(*) FROM games')
        count = self.cursor.fetchone()[0]
        self.assertEqual(count, 1)
        
        # Check if numbers were loaded correctly
        self.cursor.execute('SELECT numbers FROM games')
        numbers = json.loads(self.cursor.fetchone()[0])
        self.assertEqual(numbers, list(range(1, 21)))
    
    def test_game_numbers_indexing(self):
        """Test if individual numbers are properly indexed."""
        # Load valid data
        json_path = os.path.join(self.temp_dir, 'sample.json')
        self.loader.load_json(json_path)
        
        # Check if numbers are in the game_numbers table
        self.cursor.execute('SELECT COUNT(*) FROM game_numbers')
        count = self.cursor.fetchone()[0]
        self.assertEqual(count, 20)
        
        # Check if we can query by number
        self.cursor.execute('SELECT COUNT(*) FROM game_numbers WHERE number = 1')
        count = self.cursor.fetchone()[0]
        self.assertEqual(count, 1)

if __name__ == '__main__':
    unittest.main() 