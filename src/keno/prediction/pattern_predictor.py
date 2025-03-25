#!/usr/bin/env python3
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging
import os

class PatternPredictor:
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'keno_data')
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(log_dir, 'pattern_predictor.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def analyze_pattern(self, numbers: List[int], data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze a pattern of numbers against historical data."""
        try:
            # Convert numbers to set for faster lookups
            number_set = set(numbers)
            
            # Find historical matches
            matches = []
            for _, row in data.iterrows():
                draw_numbers = set(row['winning_numbers'])
                common_numbers = number_set.intersection(draw_numbers)
                
                if len(common_numbers) >= len(numbers) * 0.5:  # At least 50% match
                    matches.append({
                        'draw_number': row['draw_number'],
                        'draw_time': row['draw_time'],
                        'matches': list(common_numbers)
                    })
            
            # Calculate statistics
            total_draws = len(data)
            match_count = len(matches)
            frequency = (match_count / total_draws) * 100 if total_draws > 0 else 0
            
            # Calculate confidence based on recent matches
            recent_matches = [m for m in matches if self._is_recent(m['draw_time'])]
            recent_confidence = (len(recent_matches) / min(100, total_draws)) * 100
            
            # Calculate pattern strength
            pattern_strength = self._calculate_pattern_strength(numbers, matches)
            
            # Combine confidence metrics
            confidence = (recent_confidence * 0.6 + pattern_strength * 0.4)
            
            return {
                'frequency': frequency,
                'confidence': confidence,
                'historical_matches': matches,
                'pattern_strength': pattern_strength,
                'recent_matches': len(recent_matches)
            }
            
        except Exception as e:
            logging.error(f"Error analyzing pattern: {str(e)}")
            raise
            
    def _is_recent(self, draw_time: str) -> bool:
        """Check if a draw time is recent (within last 30 days)."""
        try:
            draw_date = pd.to_datetime(draw_time)
            thirty_days_ago = datetime.now() - timedelta(days=30)
            return draw_date >= thirty_days_ago
        except:
            return False
            
    def _calculate_pattern_strength(self, numbers: List[int], matches: List[Dict]) -> float:
        """Calculate the strength of a pattern based on historical matches."""
        if not matches:
            return 0.0
            
        # Calculate average match percentage
        match_percentages = []
        for match in matches:
            match_percentage = len(match['matches']) / len(numbers)
            match_percentages.append(match_percentage)
            
        # Weight recent matches more heavily
        recent_matches = [m for m in matches if self._is_recent(m['draw_time'])]
        if recent_matches:
            recent_percentages = [len(m['matches']) / len(numbers) for m in recent_matches]
            recent_weight = 0.7
            historical_weight = 0.3
            strength = (np.mean(recent_percentages) * recent_weight + 
                       np.mean(match_percentages) * historical_weight)
        else:
            strength = np.mean(match_percentages)
            
        return strength * 100  # Convert to percentage
        
    def find_common_patterns(self, data: pd.DataFrame, min_pattern_size: int = 3) -> List[Dict[str, Any]]:
        """Find common patterns in the historical data."""
        try:
            patterns = {}
            
            # Analyze each draw
            for _, row in data.iterrows():
                numbers = row['winning_numbers']
                
                # Generate all possible patterns of size min_pattern_size
                for i in range(len(numbers) - min_pattern_size + 1):
                    pattern = tuple(sorted(numbers[i:i + min_pattern_size]))
                    if pattern in patterns:
                        patterns[pattern]['count'] += 1
                        patterns[pattern]['last_seen'] = row['draw_time']
                    else:
                        patterns[pattern] = {
                            'count': 1,
                            'last_seen': row['draw_time']
                        }
            
            # Convert to list and sort by frequency
            pattern_list = [
                {
                    'pattern': list(pattern),
                    'frequency': info['count'] / len(data) * 100,
                    'last_seen': info['last_seen']
                }
                for pattern, info in patterns.items()
            ]
            
            return sorted(pattern_list, key=lambda x: x['frequency'], reverse=True)
            
        except Exception as e:
            logging.error(f"Error finding common patterns: {str(e)}")
            return []
            
    def predict_next_pattern(self, data: pd.DataFrame, pattern_size: int = 10) -> Dict[str, Any]:
        """Predict the next pattern based on historical data."""
        try:
            # Get recent draws
            recent_data = data.tail(100)  # Last 100 draws
            
            # Find common patterns
            patterns = self.find_common_patterns(recent_data, min_pattern_size=3)
            
            # Analyze frequency of individual numbers
            number_freq = {}
            for _, row in recent_data.iterrows():
                for num in row['winning_numbers']:
                    number_freq[num] = number_freq.get(num, 0) + 1
            
            # Sort numbers by frequency
            sorted_numbers = sorted(number_freq.items(), key=lambda x: x[1], reverse=True)
            
            # Select top numbers based on frequency
            predicted_numbers = [num for num, _ in sorted_numbers[:pattern_size]]
            
            # Analyze the predicted pattern
            analysis = self.analyze_pattern(predicted_numbers, data)
            
            return {
                'predicted_numbers': predicted_numbers,
                'confidence': analysis['confidence'],
                'frequency': analysis['frequency'],
                'pattern_strength': analysis['pattern_strength']
            }
            
        except Exception as e:
            logging.error(f"Error predicting next pattern: {str(e)}")
            return {
                'predicted_numbers': [],
                'confidence': 0.0,
                'frequency': 0.0,
                'pattern_strength': 0.0,
                'error': str(e)
            } 