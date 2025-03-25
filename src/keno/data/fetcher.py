"""
Keno data fetcher for automatic result updates.
"""

import requests
import pandas as pd
import os
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
import time
import json

class KenoDataFetcher:
    def __init__(self, data_dir: str):
        """Initialize the data fetcher.
        
        Args:
            data_dir: Directory to store the data files
        """
        self.data_dir = data_dir
        self.base_url = "https://www.bclc.com/api/keno/draws"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Origin': 'https://www.bclc.com',
            'Referer': 'https://www.bclc.com/winning-numbers/keno/past-results.html'
        }
        self.logger = logging.getLogger(__name__)
        
    def fetch_latest_results(self) -> Optional[List[List[int]]]:
        """Fetch the latest Keno results from BCLC API.
        
        Returns:
            List of lists containing the drawn numbers, or None if fetch fails
        """
        try:
            # Add delay to avoid rate limiting
            time.sleep(1)
            
            # Get today's date for the API request
            today = datetime.now().strftime('%Y-%m-%d')
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            # First try to get today's results
            params = {
                'date': today,
                'product': 'KENO'
            }
            
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params=params
            )
            
            if response.status_code == 404:
                # If no results for today, try yesterday
                params['date'] = yesterday
                response = requests.get(
                    self.base_url,
                    headers=self.headers,
                    params=params
                )
            
            response.raise_for_status()
            
            data = response.json()
            if not data or 'draws' not in data:
                self.logger.error("No draws data found in response")
                return None
            
            results = []
            for draw in data['draws']:
                if 'numbers' in draw and len(draw['numbers']) == 20:
                    # BCLC returns numbers as strings, convert to int
                    results.append([int(n) for n in draw['numbers']])
            
            if not results:
                self.logger.error("No valid results found in the response")
                return None
            
            return results
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error fetching results: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON response: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching results: {str(e)}")
            return None
    
    def update_historical_data(self) -> bool:
        """Update the historical data with latest results.
        
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            # Fetch latest results
            latest_results = self.fetch_latest_results()
            if not latest_results:
                return False
                
            # Get today's date for filename
            today = datetime.now().strftime('%Y%m%d')
            filename = f"KenoYear{today[:4]}.csv"
            filepath = os.path.join(self.data_dir, filename)
            
            # Create DataFrame with latest results
            df = pd.DataFrame(latest_results)
            df.columns = [f'NUMBER DRAWN {i+1}' for i in range(20)]
            df['DRAW DATE'] = datetime.now().strftime('%Y-%m-%d')
            df['DRAW NUMBER'] = range(1, len(df) + 1)
            df['PRODUCT'] = 'KENO'
            df['BONUS MULTIPLIER'] = 1
            
            # Reorder columns to match existing format
            cols = ['PRODUCT', 'DRAW NUMBER', 'DRAW DATE', 'BONUS MULTIPLIER'] + \
                   [f'NUMBER DRAWN {i+1}' for i in range(20)]
            df = df[cols]
            
            # Append to existing file or create new one
            if os.path.exists(filepath):
                df.to_csv(filepath, mode='a', header=False, index=False)
            else:
                df.to_csv(filepath, index=False)
                
            self.logger.info(f"Successfully updated data in {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating historical data: {str(e)}")
            return False
    
    def get_recent_results(self, days: int = 100) -> pd.DataFrame:
        """Get the most recent results.
        
        Args:
            days: Number of days of results to fetch
            
        Returns:
            DataFrame containing the recent results
        """
        try:
            results = []
            current_date = datetime.now()
            start_date = current_date - timedelta(days=days)
            
            # Fetch results for each day in the range
            date = current_date
            while date >= start_date:
                params = {
                    'date': date.strftime('%Y-%m-%d'),
                    'product': 'KENO'
                }
                
                try:
                    response = requests.get(
                        self.base_url,
                        headers=self.headers,
                        params=params
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data and 'draws' in data:
                            for draw in data['draws']:
                                if 'numbers' in draw and len(draw['numbers']) == 20:
                                    row = {
                                        'PRODUCT': 'KENO',
                                        'DRAW NUMBER': draw.get('drawNumber', 0),
                                        'DRAW DATE': date.strftime('%Y-%m-%d'),
                                        'BONUS MULTIPLIER': draw.get('multiplier', 1)
                                    }
                                    # Add the numbers
                                    for i, num in enumerate(draw['numbers'], 1):
                                        row[f'NUMBER DRAWN {i}'] = int(num)
                                    results.append(row)
                    
                    # Add delay to avoid rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    self.logger.warning(f"Error fetching results for {date.strftime('%Y-%m-%d')}: {str(e)}")
                    time.sleep(1)  # Longer delay on error
                
                date -= timedelta(days=1)
            
            if results:
                df = pd.DataFrame(results)
                return df.sort_values('DRAW DATE')
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting recent results: {str(e)}")
            return pd.DataFrame() 