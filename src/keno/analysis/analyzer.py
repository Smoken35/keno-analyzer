"""
Core analyzer module for Keno data analysis and predictions.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import glob
import os

class KenoAnalyzer:
    """Analyzes Keno data and makes predictions."""
    
    def __init__(self, data_source: str):
        """
        Initialize the analyzer.
        
        Args:
            data_source: Path to the data source (CSV file or directory)
        """
        self.data_source = data_source
        self.data: List[List[int]] = []
        self.payout_table: Dict[int, Dict[int, float]] = {}
        
    def load_data(self) -> None:
        """Load Keno data from the data source."""
        if os.path.isdir(self.data_source):
            self._load_csv_files(self.data_source)
        elif os.path.isfile(self.data_source):
            self._load_single_file(self.data_source)
        else:
            raise ValueError(f"Invalid data source: {self.data_source}")
            
    def _load_csv_files(self, directory: str) -> None:
        """Load Keno data from CSV files in the specified directory."""
        all_draws = []
        csv_files = glob.glob(os.path.join(directory, "*.csv"))
        
        for file_path in sorted(csv_files):
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                numbers = [int(row[f"NUMBER DRAWN {i}"]) for i in range(1, 21)]
                all_draws.append(sorted(numbers))
        
        self.data = all_draws
        
    def _load_single_file(self, file_path: str) -> None:
        """Load Keno data from a single CSV file."""
        df = pd.read_csv(file_path)
        self.data = [
            sorted([int(row[f"NUMBER DRAWN {i}"]) for i in range(1, 21)])
            for _, row in df.iterrows()
        ]
        
    def analyze_frequency(self) -> Dict[int, int]:
        """
        Analyze the frequency of each number.
        
        Returns:
            Dict mapping numbers to their frequencies
        """
        frequencies = defaultdict(int)
        for draw in self.data:
            for num in draw:
                frequencies[num] += 1
        return dict(frequencies)
    
    def analyze_patterns(self, window: int = 30) -> Dict[str, List[int]]:
        """
        Analyze hot and cold numbers based on recent draws.
        
        Args:
            window: Number of recent draws to analyze
            
        Returns:
            Dict containing hot and cold numbers
        """
        recent_draws = self.data[-window:]
        number_counts = defaultdict(int)
        
        for draw in recent_draws:
            for num in draw:
                number_counts[num] += 1
                
        sorted_numbers = sorted(
            number_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        hot_numbers = [num for num, _ in sorted_numbers[:20]]
        cold_numbers = [num for num, _ in sorted_numbers[-20:]]
        
        return {
            'hot_numbers': sorted(hot_numbers),
            'cold_numbers': sorted(cold_numbers)
        }
    
    def analyze_due_numbers(self) -> List[Tuple[int, float]]:
        """
        Analyze numbers that are "due" to appear based on their absence.
        
        Returns:
            List of tuples (number, due score)
        """
        last_seen = {num: 0 for num in range(1, 81)}
        current_draw = 0
        
        for draw in self.data:
            current_draw += 1
            for num in range(1, 81):
                if num in draw:
                    last_seen[num] = current_draw
        
        due_scores = [
            (num, (current_draw - last_draw) / current_draw)
            for num, last_draw in last_seen.items()
        ]
        return sorted(due_scores, key=lambda x: x[1], reverse=True)
    
    def predict_next_draw(self, method: str, picks: int) -> List[int]:
        """
        Predict the next draw using the specified method.
        
        Args:
            method: Prediction method ('frequency', 'patterns', 'due')
            picks: Number of numbers to predict
            
        Returns:
            List of predicted numbers
        """
        if method == 'frequency':
            freq = self.analyze_frequency()
            return sorted(dict(sorted(freq.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)[:picks]).keys())
        elif method == 'patterns':
            patterns = self.analyze_patterns()
            return sorted(patterns['hot_numbers'][:picks])
        elif method == 'due':
            due_numbers = self.analyze_due_numbers()
            return sorted([num for num, _ in due_numbers[:picks]])
        else:
            raise ValueError(f"Unknown prediction method: {method}")
            
    def calculate_expected_value(self, pick_size: int, method: str = 'frequency') -> float:
        """
        Calculate the expected value for a bet using the specified method.
        
        Args:
            pick_size: Number of numbers to pick
            method: Method to use for probability calculation
            
        Returns:
            Expected value of the bet
        """
        if pick_size not in self.payout_table:
            return 0.0
            
        payouts = self.payout_table[pick_size]
        total_ev = 0.0
        
        # Get probabilities based on the method
        if method == 'frequency':
            freq = self.analyze_frequency()
            total_draws = len(self.data)
            probs = {num: count/total_draws for num, count in freq.items()}
        else:
            # Default to uniform probability
            probs = {num: 1/80 for num in range(1, 81)}
            
        # Calculate probability of each number of matches
        for matches, payout in payouts.items():
            prob = self._calculate_hit_probability(pick_size, matches, probs)
            total_ev += prob * payout
            
        return total_ev
        
    def _calculate_hit_probability(self, picks: int, matches: int, probs: Dict[int, float]) -> float:
        """
        Calculate the probability of getting exactly k matches.
        
        Args:
            picks: Number of numbers picked
            matches: Number of matches to calculate probability for
            probs: Dictionary mapping numbers to their probabilities
            
        Returns:
            Probability of getting exactly k matches
        """
        # Use hypergeometric distribution for uniform probability
        if all(p == 1/80 for p in probs.values()):
            return stats.hypergeom.pmf(matches, 80, 20, picks)
            
        # For non-uniform probabilities, use simulation
        n_simulations = 10000
        match_count = 0
        
        for _ in range(n_simulations):
            # Simulate a draw
            draw = np.random.choice(
                list(probs.keys()), 
                size=20, 
                replace=False, 
                p=[p/sum(probs.values()) for p in probs.values()]
            )
            # Count matches with predicted numbers
            prediction = self.predict_next_draw('frequency', picks)
            matches_in_sim = len(set(prediction) & set(draw))
            if matches_in_sim == matches:
                match_count += 1
                
        return match_count / n_simulations
    
    def set_payout_table(self, payout_table: Dict[int, Dict[int, float]]) -> None:
        """
        Set the payout table for different pick sizes.
        
        Args:
            payout_table: Dictionary mapping pick sizes to their payout tables.
                         Each payout table maps number of matches to payout multiplier.
        """
        self.payout_table = payout_table 