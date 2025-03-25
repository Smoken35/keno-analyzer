"""
Core Keno analysis module for prediction and pattern detection.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import json
import os
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict

class KenoAnalyzer:
    """Main class for analyzing Keno patterns and making predictions."""
    
    def __init__(self):
        """Initialize the KenoAnalyzer with default settings."""
        self.data = pd.DataFrame()
        self.payout_tables = {}
        self.cache_dir = os.path.expanduser('~/.keno/cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def scrape_data(self, source: str = "sample", days: int = 30) -> None:
        """
        Scrape or load Keno data from specified source.
        
        Args:
            source: Data source ("sample" or URL)
            days: Number of days of historical data to fetch
        """
        if source == "sample":
            self._generate_sample_data(days)
        else:
            # Implement actual web scraping here
            pass
            
    def _generate_sample_data(self, days: int) -> None:
        """Generate sample data for testing."""
        dates = [datetime.now() - timedelta(days=i) for i in range(days)]
        numbers = [sorted(np.random.choice(range(1, 81), size=20, replace=False)) 
                  for _ in range(days)]
        
        self.data = pd.DataFrame({
            'date': dates,
            'numbers': numbers
        })
        
    def set_payout_table(self, name: str, payouts: Dict[int, Dict[int, float]]) -> None:
        """
        Set payout table for different pick sizes.
        
        Args:
            name: Name of the payout table
            payouts: Dict mapping pick size to dict of matches and payouts
        """
        self.payout_tables[name] = payouts
        
    def analyze_frequency(self, window: int = None) -> Dict[int, int]:
        """
        Analyze number frequency in the dataset.
        
        Args:
            window: Optional rolling window size
            
        Returns:
            Dict mapping numbers to their frequencies
        """
        freq = defaultdict(int)
        data = self.data if window is None else self.data.tail(window)
        
        for numbers in data['numbers']:
            for num in numbers:
                freq[num] += 1
                
        return dict(freq)
        
    def analyze_patterns(self) -> Dict[str, any]:
        """Analyze various patterns in the data."""
        patterns = {
            'hot_numbers': [],
            'cold_numbers': [],
            'pairs': defaultdict(int),
            'sequences': []
        }
        
        # Analyze hot and cold numbers
        freq = self.analyze_frequency()
        sorted_nums = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        patterns['hot_numbers'] = [n for n, _ in sorted_nums[:10]]
        patterns['cold_numbers'] = [n for n, _ in sorted_nums[-10:]]
        
        # Analyze pairs
        for draw in self.data['numbers']:
            for i, n1 in enumerate(draw):
                for n2 in draw[i+1:]:
                    patterns['pairs'][(n1, n2)] += 1
                    
        # Find common sequences
        for draw in self.data['numbers']:
            for i in range(len(draw)-2):
                if draw[i+1] == draw[i] + 1 and draw[i+2] == draw[i] + 2:
                    patterns['sequences'].append((draw[i], draw[i+1], draw[i+2]))
                    
        return patterns
        
    def predict_using_frequency(self, num_picks: int) -> List[int]:
        """Predict numbers based on frequency analysis."""
        freq = self.analyze_frequency()
        return sorted(dict(sorted(freq.items(), 
                                key=lambda x: x[1], 
                                reverse=True)[:num_picks]).keys())
                                
    def predict_using_cycles(self, num_picks: int) -> List[int]:
        """Predict numbers based on cyclic patterns."""
        patterns = self.analyze_patterns()
        candidates = set(patterns['hot_numbers'])
        
        # Add numbers from common pairs
        common_pairs = sorted(patterns['pairs'].items(), 
                            key=lambda x: x[1], 
                            reverse=True)[:5]
        for pair, _ in common_pairs:
            candidates.update(pair)
            
        return sorted(list(candidates)[:num_picks])
        
    def predict_using_markov(self, num_picks: int) -> List[int]:
        """Predict numbers using Markov chain analysis."""
        transitions = defaultdict(lambda: defaultdict(int))
        
        # Build transition matrix
        for draw in self.data['numbers']:
            for i, n1 in enumerate(draw[:-1]):
                transitions[n1][draw[i+1]] += 1
                
        # Select numbers with highest transition probabilities
        candidates = []
        current = self.data['numbers'].iloc[-1][0]  # Start with first number of last draw
        
        while len(candidates) < num_picks:
            if current not in transitions:
                break
            next_num = max(transitions[current].items(), 
                          key=lambda x: x[1])[0]
            if next_num not in candidates:
                candidates.append(next_num)
            current = next_num
            
        # Fill remaining picks with frequency-based selections
        if len(candidates) < num_picks:
            freq_picks = self.predict_using_frequency(num_picks - len(candidates))
            candidates.extend([n for n in freq_picks if n not in candidates])
            
        return sorted(candidates[:num_picks])
        
    def predict_using_due(self, num_picks: int) -> List[int]:
        """Predict numbers based on 'due' theory."""
        all_numbers = set(range(1, 81))
        recent_draws = set()
        
        # Get numbers from recent draws
        for draw in self.data['numbers'].tail(5):
            recent_draws.update(draw)
            
        # Get 'due' numbers (not drawn recently)
        due_numbers = list(all_numbers - recent_draws)
        np.random.shuffle(due_numbers)
        
        return sorted(due_numbers[:num_picks])
        
    def predict_using_ensemble(self, num_picks: int) -> List[int]:
        """Combine multiple prediction methods."""
        predictions = {
            'frequency': set(self.predict_using_frequency(num_picks)),
            'cycles': set(self.predict_using_cycles(num_picks)),
            'markov': set(self.predict_using_markov(num_picks)),
            'due': set(self.predict_using_due(num_picks))
        }
        
        # Count how many methods predicted each number
        number_counts = defaultdict(int)
        for method_predictions in predictions.values():
            for num in method_predictions:
                number_counts[num] += 1
                
        # Select numbers predicted by multiple methods
        candidates = sorted(number_counts.items(), 
                          key=lambda x: (x[1], x[0]), 
                          reverse=True)
        
        result = [num for num, _ in candidates[:num_picks]]
        
        # If we don't have enough numbers, add from frequency analysis
        if len(result) < num_picks:
            freq_picks = self.predict_using_frequency(num_picks - len(result))
            result.extend([n for n in freq_picks if n not in result])
            
        return sorted(result[:num_picks])
        
    def calculate_expected_value(self, 
                               pick_size: int, 
                               payout_table: str,
                               prediction_method: str = 'ensemble') -> float:
        """
        Calculate expected value for a prediction method.
        
        Args:
            pick_size: Number of picks
            payout_table: Name of payout table to use
            prediction_method: Prediction method to analyze
            
        Returns:
            Expected value per play
        """
        if payout_table not in self.payout_tables:
            raise ValueError(f"Unknown payout table: {payout_table}")
            
        payouts = self.payout_tables[payout_table][pick_size]
        
        # Get historical match counts
        match_counts = defaultdict(int)
        total_plays = 0
        
        for actual in self.data['numbers']:
            prediction = getattr(self, f'predict_using_{prediction_method}')(pick_size)
            matches = len(set(prediction) & set(actual))
            match_counts[matches] += 1
            total_plays += 1
            
        # Calculate expected value
        ev = 0
        for matches, count in match_counts.items():
            probability = count / total_plays
            payout = payouts.get(matches, 0)
            ev += probability * payout
            
        return ev - 1  # Subtract cost of play 