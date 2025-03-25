"""
Strategy analysis module for Keno game analysis.
Handles different number selection strategies and their performance analysis.
"""

import random
from typing import Dict, List, Optional, Tuple
import pandas as pd
from collections import Counter

class StrategyAnalyzer:
    def __init__(self, historical_data: pd.DataFrame):
        """
        Initialize strategy analyzer with historical data
        
        Args:
            historical_data: DataFrame containing historical Keno draws
        """
        self.data = historical_data
        self.strategies = {
            'frequency': self._frequency_strategy,
            'pattern': self._pattern_strategy,
            'random': self._random_strategy
        }
        
    def analyze_frequency(self) -> pd.DataFrame:
        """
        Analyze the frequency of each number in historical data
        
        Returns:
            DataFrame with frequency analysis for each number
        """
        if self.data.empty:
            print("No historical data available")
            return pd.DataFrame()
            
        # Flatten all numbers from all draws
        all_numbers = []
        for nums in self.data['numbers']:
            all_numbers.extend(nums)
            
        # Count frequency of each number
        frequency = Counter(all_numbers)
        
        # Convert to DataFrame for easier analysis
        freq_df = pd.DataFrame({
            'number': list(range(1, 81)),
            'frequency': [frequency.get(i, 0) for i in range(1, 81)],
            'percentage': [frequency.get(i, 0) / len(self.data) * 5 for i in range(1, 81)]
        })
        
        return freq_df.sort_values('frequency', ascending=False)
    
    def _frequency_strategy(self, pick_size: int) -> List[int]:
        """
        Select numbers based on historical frequency
        
        Args:
            pick_size: Number of spots to pick
            
        Returns:
            List of selected numbers
        """
        freq_df = self.analyze_frequency()
        return freq_df.head(pick_size)['number'].tolist()
    
    def _pattern_strategy(self, pick_size: int) -> List[int]:
        """
        Select numbers based on identified patterns
        
        Args:
            pick_size: Number of spots to pick
            
        Returns:
            List of selected numbers
        """
        if self.data.empty:
            return self._random_strategy(pick_size)
            
        patterns = self._analyze_patterns()
        
        # Use pattern analysis to guide selection
        target_even = round(pick_size * patterns['avg_even_ratio'])
        numbers = []
        
        # Select even numbers
        even_candidates = [n for n in range(1, 81) if n % 2 == 0]
        numbers.extend(random.sample(even_candidates, target_even))
        
        # Select odd numbers
        odd_candidates = [n for n in range(1, 81) if n % 2 != 0 and n not in numbers]
        numbers.extend(random.sample(odd_candidates, pick_size - target_even))
        
        return sorted(numbers)
    
    def _random_strategy(self, pick_size: int) -> List[int]:
        """
        Select numbers randomly
        
        Args:
            pick_size: Number of spots to pick
            
        Returns:
            List of selected numbers
        """
        return sorted(random.sample(range(1, 81), pick_size))
    
    def _analyze_patterns(self) -> Dict:
        """
        Analyze patterns in historical data
        
        Returns:
            Dictionary containing pattern statistics
        """
        if self.data.empty:
            return {
                'avg_even_ratio': 0.5,
                'avg_high_ratio': 0.5
            }
            
        even_ratios = []
        high_ratios = []
        
        for nums in self.data['numbers']:
            even_count = sum(1 for n in nums if n % 2 == 0)
            high_count = sum(1 for n in nums if n > 40)
            
            even_ratios.append(even_count / len(nums))
            high_ratios.append(high_count / len(nums))
        
        return {
            'avg_even_ratio': sum(even_ratios) / len(even_ratios),
            'avg_high_ratio': sum(high_ratios) / len(high_ratios)
        }
    
    def simulate_strategy(
        self,
        strategy: str,
        pick_size: int,
        payout_table: Dict[int, float],
        num_draws: int = 100,
        bet_amount: float = 1.0
    ) -> Dict:
        """
        Simulate a strategy's performance
        
        Args:
            strategy: Name of strategy to use ('frequency', 'pattern', 'random')
            pick_size: Number of spots to pick
            payout_table: Dictionary mapping number of matches to payout multiplier
            num_draws: Number of draws to simulate
            bet_amount: Amount bet per draw
            
        Returns:
            Dictionary containing simulation results
        """
        if strategy not in self.strategies:
            print(f"Unknown strategy: {strategy}")
            return {}
            
        if len(self.data) < 100:
            print("Insufficient historical data for meaningful simulation")
            return {}
            
        # Select numbers using the specified strategy
        selected_numbers = self.strategies[strategy](pick_size)
        
        # Simulate draws
        total_spent = num_draws * bet_amount
        total_won = 0.0
        matches_dist = {i: 0 for i in range(pick_size + 1)}
        
        for i in range(num_draws):
            draw_idx = i % len(self.data)  # Cycle through historical draws
            draw_numbers = self.data.iloc[draw_idx]['numbers']
            
            # Count matches
            matches = len(set(selected_numbers).intersection(set(draw_numbers)))
            
            # Calculate winnings
            winnings = payout_table.get(matches, 0) * bet_amount
            total_won += winnings
            
            # Update matches distribution
            matches_dist[matches] += 1
        
        return {
            'strategy': strategy,
            'selected_numbers': selected_numbers,
            'total_spent': total_spent,
            'total_won': total_won,
            'net_profit': total_won - total_spent,
            'roi': ((total_won - total_spent) / total_spent) * 100,
            'matches_distribution': matches_dist
        }
    
    def compare_strategies(
        self,
        pick_size: int,
        payout_table: Dict[int, float],
        num_draws: List[int] = [1, 5, 10, 20],
        bet_amount: float = 1.0
    ) -> pd.DataFrame:
        """
        Compare performance of different strategies
        
        Args:
            pick_size: Number of spots to pick
            payout_table: Dictionary mapping number of matches to payout multiplier
            num_draws: List of number of draws to simulate
            bet_amount: Amount bet per draw
            
        Returns:
            DataFrame comparing strategy performances
        """
        results = []
        
        for strategy in self.strategies.keys():
            for draws in num_draws:
                sim_result = self.simulate_strategy(
                    strategy, pick_size, payout_table, draws, bet_amount
                )
                
                if sim_result:
                    results.append({
                        'Strategy': strategy,
                        'Draws': draws,
                        'Selected Numbers': sim_result['selected_numbers'],
                        'Total Spent': sim_result['total_spent'],
                        'Total Won': sim_result['total_won'],
                        'Net Profit': sim_result['net_profit'],
                        'ROI %': sim_result['roi']
                    })
        
        return pd.DataFrame(results).sort_values('ROI %', ascending=False)
    
    def get_best_strategy(
        self,
        pick_size: int,
        payout_table: Dict[int, float],
        num_draws: List[int] = [1, 5, 10, 20],
        bet_amount: float = 1.0
    ) -> Tuple[str, int, List[int], float]:
        """
        Determine the best strategy and number of draws
        
        Args:
            pick_size: Number of spots to pick
            payout_table: Dictionary mapping number of matches to payout multiplier
            num_draws: List of number of draws to simulate
            bet_amount: Amount bet per draw
            
        Returns:
            Tuple of (best strategy, optimal draws, recommended numbers, expected ROI)
        """
        comparison = self.compare_strategies(pick_size, payout_table, num_draws, bet_amount)
        if comparison.empty:
            return ('random', 1, self._random_strategy(pick_size), 0.0)
            
        best_result = comparison.iloc[0]
        return (
            best_result['Strategy'],
            best_result['Draws'],
            best_result['Selected Numbers'],
            best_result['ROI %']
        ) 