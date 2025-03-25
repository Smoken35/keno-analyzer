#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class KenoPredictor:
    def __init__(self):
        self.setup_logging()
        self.data = self.load_data()
        
    def setup_logging(self):
        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'keno_data')
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(log_dir, 'predictor.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare the historical data."""
        try:
            base_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'keno_data')
            all_time_file = os.path.join(base_dir, 'keno_results_all.csv')
            
            if not os.path.exists(all_time_file) or os.path.getsize(all_time_file) == 0:
                print("No historical data found. Please process some data first.")
                return pd.DataFrame()
                
            df = pd.read_csv(all_time_file)
            if isinstance(df['winning_numbers'].iloc[0], str):
                df['winning_numbers'] = df['winning_numbers'].apply(lambda x: eval(x))
            return df
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            print(f"Error loading data: {str(e)}")
            return pd.DataFrame()
            
    def analyze_frequency(self) -> Dict[int, float]:
        """Analyze the frequency of each number."""
        frequency = {}
        total_draws = len(self.data)
        
        for numbers in self.data['winning_numbers']:
            for num in numbers:
                frequency[num] = frequency.get(num, 0) + 1
                
        # Convert to percentages
        return {num: (count / total_draws) * 100 for num, count in frequency.items()}
        
    def find_hot_numbers(self, days: int = 7) -> List[int]:
        """Find numbers that have appeared frequently in recent draws."""
        recent_data = self.data.tail(100)  # Last 100 draws
        recent_freq = {}
        
        for numbers in recent_data['winning_numbers']:
            for num in numbers:
                recent_freq[num] = recent_freq.get(num, 0) + 1
                
        # Sort by frequency and return top numbers
        return sorted(recent_freq.items(), key=lambda x: x[1], reverse=True)
        
    def find_cold_numbers(self, days: int = 7) -> List[int]:
        """Find numbers that haven't appeared recently."""
        recent_data = self.data.tail(100)  # Last 100 draws
        recent_numbers = set()
        
        for numbers in recent_data['winning_numbers']:
            recent_numbers.update(numbers)
            
        # Return numbers that haven't appeared recently
        return sorted(set(range(1, 81)) - recent_numbers)
        
    def find_patterns(self, pattern_size: int = 3) -> List[Dict[str, Any]]:
        """Find common patterns in the data."""
        patterns = {}
        
        for numbers in self.data['winning_numbers']:
            # Generate all possible patterns of size pattern_size
            for i in range(len(numbers) - pattern_size + 1):
                pattern = tuple(sorted(numbers[i:i + pattern_size]))
                if pattern in patterns:
                    patterns[pattern]['count'] += 1
                else:
                    patterns[pattern] = {'count': 1}
                    
        # Convert to list and sort by frequency
        pattern_list = [
            {
                'pattern': list(pattern),
                'frequency': info['count'] / len(self.data) * 100
            }
            for pattern, info in patterns.items()
        ]
        
        return sorted(pattern_list, key=lambda x: x['frequency'], reverse=True)
        
    def generate_prediction(self, pick_size: int = 10) -> Dict[str, Any]:
        """Generate a prediction for the next draw."""
        try:
            # Get hot and cold numbers
            hot_numbers = [num for num, _ in self.find_hot_numbers()]
            cold_numbers = self.find_cold_numbers()
            
            # Get frequency analysis
            frequency = self.analyze_frequency()
            
            # Get common patterns
            patterns = self.find_patterns()
            
            # Select numbers based on multiple factors
            selected_numbers = set()
            
            # Add some hot numbers
            selected_numbers.update(hot_numbers[:pick_size//2])
            
            # Add some cold numbers
            selected_numbers.update(cold_numbers[:pick_size//2])
            
            # Add numbers from common patterns
            for pattern in patterns[:5]:
                selected_numbers.update(pattern['pattern'])
                
            # If we need more numbers, add based on frequency
            while len(selected_numbers) < pick_size:
                remaining = set(range(1, 81)) - selected_numbers
                if not remaining:
                    break
                # Add number with highest frequency
                next_num = max(remaining, key=lambda x: frequency.get(x, 0))
                selected_numbers.add(next_num)
                
            # Convert to sorted list
            prediction = sorted(list(selected_numbers))
            
            # Calculate confidence
            confidence = self._calculate_confidence(prediction)
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'hot_numbers': hot_numbers[:5],
                'cold_numbers': cold_numbers[:5],
                'common_patterns': patterns[:5]
            }
            
        except Exception as e:
            logging.error(f"Error generating prediction: {str(e)}")
            return {
                'prediction': [],
                'confidence': 0.0,
                'error': str(e)
            }
            
    def _calculate_confidence(self, numbers: List[int]) -> float:
        """Calculate confidence level for a prediction."""
        try:
            # Get frequency of each number
            frequency = self.analyze_frequency()
            
            # Calculate average frequency
            avg_freq = sum(frequency.get(num, 0) for num in numbers) / len(numbers)
            
            # Check if numbers appear in recent draws
            recent_data = self.data.tail(10)
            recent_numbers = set()
            for nums in recent_data['winning_numbers']:
                recent_numbers.update(nums)
            
            # Calculate percentage of numbers that appeared recently
            recent_match = len(set(numbers) & recent_numbers) / len(numbers)
            
            # Combine factors for confidence
            confidence = (avg_freq * 0.6 + recent_match * 0.4)
            
            return min(confidence, 100.0)  # Cap at 100%
            
        except Exception as e:
            logging.error(f"Error calculating confidence: {str(e)}")
            return 0.0
            
    def print_prediction(self, prediction: Dict[str, Any]):
        """Print the prediction in a formatted way."""
        print("\n=== Keno Prediction ===")
        print(f"Predicted Numbers: {prediction['prediction']}")
        print(f"Confidence Level: {prediction['confidence']:.1f}%")
        
        print("\nHot Numbers (Recently Frequent):")
        for num in prediction['hot_numbers']:
            print(f"  {num}")
            
        print("\nCold Numbers (Due to Appear):")
        for num in prediction['cold_numbers']:
            print(f"  {num}")
            
        print("\nCommon Patterns Found:")
        for pattern in prediction['common_patterns']:
            print(f"  {pattern['pattern']} (Frequency: {pattern['frequency']:.1f}%)")
            
        print("\n=====================\n")

def main():
    predictor = KenoPredictor()
    
    if predictor.data.empty:
        print("No data available. Please process some data first.")
        return
        
    # Generate prediction
    prediction = predictor.generate_prediction()
    
    # Print results
    predictor.print_prediction(prediction)

if __name__ == '__main__':
    main() 