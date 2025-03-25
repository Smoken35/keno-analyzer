"""
Enhanced PlayNow Keno Scraper with real browser support
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import time
import os
import json
import logging
import random
import re
from typing import List, Dict, Any, Tuple, Optional
import argparse
import schedule
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException, StaleElementReferenceException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from urllib3.exceptions import NewConnectionError
import backoff

logger = logging.getLogger(__name__)

class PlayNowScraper:
    def __init__(self, data_dir: str = 'keno_data'):
        self.setup_logging()
        self.data_dir = data_dir
        self.setup_directories()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
    def setup_logging(self):
        """Set up logging configuration."""
        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'keno_data')
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(log_dir, 'scraper.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def setup_directories(self):
        """Create necessary directories for data storage."""
        dirs = [
            os.path.join(self.data_dir, 'daily'),
            os.path.join(self.data_dir, 'csv'),
            os.path.join(self.data_dir, 'json')
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            
    def get_latest_results(self) -> Optional[List[Dict[str, Any]]]:
        """Fetch the latest Keno results from PlayNow.com."""
        try:
            url = 'https://www.playnow.com/lottery/keno/winning-numbers'
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Find the results table
            table = soup.find('table', {'class': 'winning-numbers-table'})
            if not table:
                logging.error("Could not find results table")
                return None
                
            # Parse each row
            for row in table.find_all('tr')[1:]:  # Skip header row
                cols = row.find_all('td')
                if len(cols) >= 3:
                    draw_number = cols[0].text.strip()
                    draw_time = cols[1].text.strip()
                    numbers = [int(n.strip()) for n in cols[2].text.strip().split(',')]
                    
                    results.append({
                        'draw_number': draw_number,
                        'draw_time': draw_time,
                        'winning_numbers': numbers
                    })
            
            return results
            
        except Exception as e:
            logging.error(f"Error fetching results: {str(e)}")
            return None
            
    def save_results(self, results: List[Dict[str, Any]], format: str = 'both'):
        """Save results in specified format(s)."""
        if not results:
            logging.error("No results to save")
            return False
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # Save as CSV
            if format in ['csv', 'both']:
                csv_file = os.path.join(self.data_dir, 'daily', f'keno_results_{timestamp}.csv')
                df = pd.DataFrame(results)
                df.to_csv(csv_file, index=False)
                logging.info(f"Saved results to {csv_file}")
                
            # Save as JSON
            if format in ['json', 'both']:
                json_file = os.path.join(self.data_dir, 'daily', f'keno_results_{timestamp}.json')
                with open(json_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logging.info(f"Saved results to {json_file}")
                
            return True
            
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")
            return False
            
    def update_all_time_results(self, new_results: List[Dict[str, Any]]):
        """Update the all-time results file with new data."""
        try:
            all_time_file = os.path.join(self.data_dir, 'keno_results_all.csv')
            
            # Load existing results if file exists
            existing_results = []
            if os.path.exists(all_time_file):
                df = pd.read_csv(all_time_file)
                existing_results = df.to_dict('records')
                
            # Combine with new results
            all_results = existing_results + new_results
            
            # Remove duplicates based on draw_number
            seen = set()
            unique_results = []
            for result in all_results:
                if result['draw_number'] not in seen:
                    seen.add(result['draw_number'])
                    unique_results.append(result)
                    
            # Save updated results
            df = pd.DataFrame(unique_results)
            df.to_csv(all_time_file, index=False)
            logging.info(f"Updated all-time results file with {len(new_results)} new entries")
            
        except Exception as e:
            logging.error(f"Error updating all-time results: {str(e)}")
            
    def run_daily_update(self):
        """Run a daily update of Keno results."""
        logging.info("Starting daily update")
        
        # Add random delay to avoid detection
        time.sleep(random.uniform(1, 3))
        
        results = self.get_latest_results()
        if not results:
            logging.error("Failed to fetch latest results")
            return False
            
        if self.save_results(results):
            self.update_all_time_results(results)
            logging.info("Daily update completed successfully")
            return True
            
        return False

def main():
    parser = argparse.ArgumentParser(description='PlayNow Keno Scraper')
    parser.add_argument('--data-dir', default='keno_data', help='Directory to store data files')
    parser.add_argument('--daily-update', action='store_true', help='Run a daily update of results')
    args = parser.parse_args()
    
    scraper = PlayNowScraper(data_dir=args.data_dir)
    
    if args.daily_update:
        if scraper.run_daily_update():
            print("Daily update completed successfully")
        else:
            print("Daily update failed. Check the log file for details.")
    else:
        results = scraper.get_latest_results()
        if results:
            if scraper.save_results(results):
                print("Results saved successfully")
            else:
                print("Failed to save results")
        else:
            print("Failed to fetch results")

if __name__ == '__main__':
    main()