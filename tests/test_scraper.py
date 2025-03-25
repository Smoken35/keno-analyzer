"""
Unit tests for the PlayNow Keno scraper.
"""

import unittest
import os
import tempfile
import shutil
from datetime import datetime
from src.keno.data.playnow_scraper import PlayNowKenoScraper

class TestPlayNowKenoScraper(unittest.TestCase):
    """Test cases for the PlayNowKenoScraper class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.scraper = PlayNowKenoScraper(data_dir=self.test_dir)
        
        # Create sample HTML content for testing
        self.sample_html = """
        <html>
            <body>
                <div class="keno-result-container">
                    <div class="date">Mar 23, 2025</div>
                    <div class="time">14:30</div>
                    <div class="draw-number">12345</div>
                    <div class="keno-number">1</div>
                    <div class="keno-number">2</div>
                    <div class="keno-number">3</div>
                    <div class="keno-number">4</div>
                    <div class="keno-number">5</div>
                    <div class="keno-number">6</div>
                    <div class="keno-number">7</div>
                    <div class="keno-number">8</div>
                    <div class="keno-number">9</div>
                    <div class="keno-number">10</div>
                    <div class="keno-number">11</div>
                    <div class="keno-number">12</div>
                    <div class="keno-number">13</div>
                    <div class="keno-number">14</div>
                    <div class="keno-number">15</div>
                    <div class="keno-number">16</div>
                    <div class="keno-number">17</div>
                    <div class="keno-number">18</div>
                    <div class="keno-number">19</div>
                    <div class="keno-number">20</div>
                </div>
            </body>
        </html>
        """
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_parse_winning_numbers(self):
        """Test parsing of winning numbers from HTML."""
        results = self.scraper.parse_winning_numbers(self.sample_html)
        
        self.assertEqual(len(results), 1)
        result = results[0]
        
        self.assertEqual(result['date'], '2025-03-23')
        self.assertEqual(result['time'], '14:30')
        self.assertEqual(result['draw_number'], '12345')
        self.assertEqual(len(result['winning_numbers']), 20)
        self.assertEqual(result['winning_numbers'][0], 1)
        self.assertEqual(result['winning_numbers'][-1], 20)
    
    def test_save_results(self):
        """Test saving results to files."""
        # Create sample results
        results = [{
            'date': '2025-03-23',
            'time': '14:30',
            'draw_number': '12345',
            'winning_numbers': list(range(1, 21))
        }]
        
        # Save results
        csv_path, json_path, daily_path = self.scraper.save_results(results)
        
        # Check if files were created
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(daily_path))
        
        # Check CSV content
        import pandas as pd
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 1)
        self.assertEqual(df['draw_number'].iloc[0], '12345')
        
        # Check JSON content
        import json
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        self.assertEqual(len(json_data), 1)
        self.assertEqual(json_data[0]['draw_number'], '12345')
    
    def test_convert_to_standard_format(self):
        """Test conversion to standard format."""
        # Create a temporary CSV file
        csv_path = os.path.join(self.test_dir, 'test.csv')
        import pandas as pd
        df = pd.DataFrame([{
            'date': '2025-03-23',
            'time': '14:30',
            'draw_number': '12345',
            'winning_numbers': list(range(1, 21))
        }])
        df.to_csv(csv_path, index=False)
        
        # Convert to standard format
        standardized_path = self.scraper.convert_to_standard_format(csv_path)
        
        # Check if file was created
        self.assertTrue(os.path.exists(standardized_path))
        
        # Check content
        df_standardized = pd.read_csv(standardized_path)
        self.assertEqual(len(df_standardized), 1)
        self.assertEqual(df_standardized['DRAW_NUMBER'].iloc[0], '12345')
        self.assertEqual(df_standardized['NUMBER_DRAWN_1'].iloc[0], 1)
        self.assertEqual(df_standardized['NUMBER_DRAWN_20'].iloc[0], 20)
    
    def test_directory_creation(self):
        """Test creation of data directories."""
        # Check if directories were created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'daily')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'csv')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'json')))
    
    def test_duplicate_handling(self):
        """Test handling of duplicate results."""
        # Create sample results
        results = [{
            'date': '2025-03-23',
            'time': '14:30',
            'draw_number': '12345',
            'winning_numbers': list(range(1, 21))
        }]
        
        # Save results twice
        self.scraper.save_results(results)
        self.scraper.save_results(results)
        
        # Check consolidated file
        consolidated_path = os.path.join(self.test_dir, 'keno_results_all.csv')
        import pandas as pd
        df = pd.read_csv(consolidated_path)
        self.assertEqual(len(df), 1)  # Should only have one entry
    
    def test_date_standardization(self):
        """Test date format standardization."""
        # Test various date formats
        test_cases = [
            ('Mar 23, 2025', '2025-03-23'),
            ('March 23, 2025', '2025-03-23'),
            ('2025-03-23', '2025-03-23')  # Already standardized
        ]
        
        for input_date, expected_date in test_cases:
            html = f"""
            <html>
                <body>
                    <div class="keno-result-container">
                        <div class="date">{input_date}</div>
                        <div class="time">14:30</div>
                        <div class="draw-number">12345</div>
                        <div class="keno-number">1</div>
                    </div>
                </body>
            </html>
            """
            
            results = self.scraper.parse_winning_numbers(html)
            self.assertEqual(results[0]['date'], expected_date)

if __name__ == '__main__':
    unittest.main() 