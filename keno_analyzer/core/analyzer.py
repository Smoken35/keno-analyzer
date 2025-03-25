"""
Core Keno analysis functionality.
"""

import numpy as np
from typing import List, Dict, Union, Optional, Tuple
import pandas as pd
from scipy import stats

class KenoAnalyzer:
    """Analyzes Keno game data and generates predictions."""
    
    def __init__(self, historical_data: List[List[int]]):
        """
        Initialize the analyzer with historical Keno draw data.
        
        Args:
            historical_data: List of lists containing historical draw numbers
        """
        self.historical_data = historical_data
        self.numbers = list(range(1, 81))  # Keno numbers 1-80
        self.data: List[List[int]] = []
        self.payout_table: Dict[int, Dict[int, float]] = {}
        
    def analyze_frequency(self) -> Dict[int, int]:
        """
        Analyze the frequency of each number in historical draws.
        
        Returns:
            Dictionary mapping numbers to their frequency
        """
        frequency = {num: 0 for num in self.numbers}
        for draw in self.historical_data:
            for num in draw:
                frequency[num] += 1
        return frequency
    
    def analyze_patterns(self) -> Dict[str, List[int]]:
        """
        Analyze patterns in the historical data.
        
        Returns:
            Dictionary containing hot and cold numbers
        """
        frequency = self.analyze_frequency()
        sorted_nums = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "hot_numbers": [num for num, _ in sorted_nums[:10]],
            "cold_numbers": [num for num, _ in sorted_nums[-10:]]
        }
    
    def predict_next_draw(self, method: str = "frequency") -> List[int]:
        """
        Predict the next draw using specified method.
        
        Args:
            method: Prediction method to use ("frequency" or "pattern")
            
        Returns:
            List of predicted numbers
            
        Raises:
            ValueError: If method is not supported
        """
        if method == "frequency":
            frequency = self.analyze_frequency()
            return sorted(np.random.choice(
                self.numbers,
                size=20,
                p=[frequency[num]/sum(frequency.values()) for num in self.numbers],
                replace=False
            ))
        elif method == "pattern":
            patterns = self.analyze_patterns()
            hot_nums = patterns["hot_numbers"]
            cold_nums = patterns["cold_numbers"]
            # Mix hot and cold numbers
            return sorted(np.random.choice(
                hot_nums + cold_nums,
                size=20,
                replace=False
            ))
        else:
            raise ValueError(f"Unsupported prediction method: {method}")
    
    def calculate_expected_value(self, bet_amount: float, payout_table: Dict[int, float]) -> float:
        """
        Calculate expected value for a bet.
        
        Args:
            bet_amount: Amount to bet
            payout_table: Dictionary mapping number of matches to payout multiplier
            
        Returns:
            Expected value of the bet
        """
        if not payout_table:
            raise ValueError("Payout table is required for expected value calculation")
            
        # Calculate probability of each number of matches
        prob_matches = {}
        for matches in payout_table.keys():
            # This is a simplified calculation
            prob = 1.0 / (80 * 79 * 78 * 77 * 76)  # Very rough approximation
            prob_matches[matches] = prob
            
        # Calculate expected value
        ev = -bet_amount  # Start with cost of bet
        for matches, prob in prob_matches.items():
            if matches in payout_table:
                ev += bet_amount * payout_table[matches] * prob
                
        return ev
    
    def _calculate_hit_probability(
        self, 
        prediction: List[int], 
        hits: int
    ) -> float:
        """Calculate probability of getting exactly n hits."""
        total_matches = sum(1 for draw in self.data[-100:] 
                          if len(set(prediction) & set(draw)) == hits)
        return total_matches / 100 if self.data else 0.0 