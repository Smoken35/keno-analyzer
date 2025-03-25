#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class KenoGapAnalyzer:
    def __init__(self):
        self.setup_logging()
        self.data = self.load_data()
        self.setup_output_dir()
        
    def setup_logging(self):
        """Set up logging configuration."""
        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'keno_data')
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(log_dir, 'gap_analysis.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def setup_output_dir(self):
        """Set up directory for analysis outputs."""
        self.output_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'keno_data', 'analysis')
        os.makedirs(self.output_dir, exist_ok=True)
        
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
            df['draw_time'] = pd.to_datetime(df['draw_time'])
            return df
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            print(f"Error loading data: {str(e)}")
            return pd.DataFrame()
            
    def analyze_number_gaps(self) -> Dict[int, Dict[str, Any]]:
        """Analyze the gaps between appearances of each number."""
        try:
            gap_stats = defaultdict(lambda: {
                'last_seen': None,
                'current_gap': 0,
                'max_gap': 0,
                'avg_gap': 0,
                'gap_history': [],
                'appearances': 0,
                'overdue_factor': 0
            })
            
            # Sort data by draw time
            sorted_data = self.data.sort_values('draw_time')
            
            # Calculate gaps for each number
            for _, row in sorted_data.iterrows():
                draw_time = row['draw_time']
                numbers = row['winning_numbers']
                
                for num in range(1, 81):  # All possible Keno numbers
                    if num in numbers:
                        if gap_stats[num]['last_seen'] is not None:
                            gap = (draw_time - gap_stats[num]['last_seen']).days
                            gap_stats[num]['gap_history'].append(gap)
                            gap_stats[num]['max_gap'] = max(gap_stats[num]['max_gap'], gap)
                        gap_stats[num]['last_seen'] = draw_time
                        gap_stats[num]['appearances'] += 1
                    else:
                        if gap_stats[num]['last_seen'] is not None:
                            gap_stats[num]['current_gap'] = (draw_time - gap_stats[num]['last_seen']).days
            
            # Calculate average gaps and overdue factors
            for num in gap_stats:
                if gap_stats[num]['gap_history']:
                    gap_stats[num]['avg_gap'] = np.mean(gap_stats[num]['gap_history'])
                    # Overdue factor: current gap / average gap
                    gap_stats[num]['overdue_factor'] = (
                        gap_stats[num]['current_gap'] / gap_stats[num]['avg_gap']
                        if gap_stats[num]['avg_gap'] > 0 else 0
                    )
            
            return dict(gap_stats)
            
        except Exception as e:
            logging.error(f"Error analyzing number gaps: {str(e)}")
            return {}
            
    def analyze_pair_gaps(self) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """Analyze gaps between co-occurrences of number pairs."""
        try:
            pair_stats = defaultdict(lambda: {
                'last_seen': None,
                'current_gap': 0,
                'max_gap': 0,
                'avg_gap': 0,
                'gap_history': [],
                'co_occurrences': 0,
                'overdue_factor': 0
            })
            
            # Sort data by draw time
            sorted_data = self.data.sort_values('draw_time')
            
            # Calculate gaps for each pair
            for _, row in sorted_data.iterrows():
                draw_time = row['draw_time']
                numbers = set(row['winning_numbers'])
                
                # Check all possible pairs
                for i in range(1, 81):
                    for j in range(i + 1, 81):
                        pair = (i, j)
                        if i in numbers and j in numbers:
                            if pair_stats[pair]['last_seen'] is not None:
                                gap = (draw_time - pair_stats[pair]['last_seen']).days
                                pair_stats[pair]['gap_history'].append(gap)
                                pair_stats[pair]['max_gap'] = max(pair_stats[pair]['max_gap'], gap)
                            pair_stats[pair]['last_seen'] = draw_time
                            pair_stats[pair]['co_occurrences'] += 1
                        else:
                            if pair_stats[pair]['last_seen'] is not None:
                                pair_stats[pair]['current_gap'] = (draw_time - pair_stats[pair]['last_seen']).days
            
            # Calculate average gaps and overdue factors
            for pair in pair_stats:
                if pair_stats[pair]['gap_history']:
                    pair_stats[pair]['avg_gap'] = np.mean(pair_stats[pair]['gap_history'])
                    pair_stats[pair]['overdue_factor'] = (
                        pair_stats[pair]['current_gap'] / pair_stats[pair]['avg_gap']
                        if pair_stats[pair]['avg_gap'] > 0 else 0
                    )
            
            return dict(pair_stats)
            
        except Exception as e:
            logging.error(f"Error analyzing pair gaps: {str(e)}")
            return {}
            
    def find_pattern_gaps(self) -> Dict[str, Dict[str, Any]]:
        """Find gaps in pattern occurrences."""
        try:
            pattern_stats = defaultdict(lambda: {
                'last_seen': None,
                'current_gap': 0,
                'max_gap': 0,
                'avg_gap': 0,
                'gap_history': [],
                'occurrences': 0,
                'overdue_factor': 0
            })
            
            # Define patterns to track
            patterns = {
                'consecutive': lambda nums: any(nums[i+1] - nums[i] == 1 for i in range(len(nums)-1)),
                'even_heavy': lambda nums: len([n for n in nums if n % 2 == 0]) > len(nums) / 2,
                'odd_heavy': lambda nums: len([n for n in nums if n % 2 == 1]) > len(nums) / 2,
                'high_heavy': lambda nums: len([n for n in nums if n > 40]) > len(nums) / 2,
                'low_heavy': lambda nums: len([n for n in nums if n <= 40]) > len(nums) / 2
            }
            
            # Sort data by draw time
            sorted_data = self.data.sort_values('draw_time')
            
            # Track pattern occurrences
            for _, row in sorted_data.iterrows():
                draw_time = row['draw_time']
                numbers = sorted(row['winning_numbers'])
                
                for pattern_name, pattern_func in patterns.items():
                    if pattern_func(numbers):
                        if pattern_stats[pattern_name]['last_seen'] is not None:
                            gap = (draw_time - pattern_stats[pattern_name]['last_seen']).days
                            pattern_stats[pattern_name]['gap_history'].append(gap)
                            pattern_stats[pattern_name]['max_gap'] = max(
                                pattern_stats[pattern_name]['max_gap'], gap
                            )
                        pattern_stats[pattern_name]['last_seen'] = draw_time
                        pattern_stats[pattern_name]['occurrences'] += 1
                    else:
                        if pattern_stats[pattern_name]['last_seen'] is not None:
                            pattern_stats[pattern_name]['current_gap'] = (
                                draw_time - pattern_stats[pattern_name]['last_seen']
                            ).days
            
            # Calculate average gaps and overdue factors
            for pattern in pattern_stats:
                if pattern_stats[pattern]['gap_history']:
                    pattern_stats[pattern]['avg_gap'] = np.mean(pattern_stats[pattern]['gap_history'])
                    pattern_stats[pattern]['overdue_factor'] = (
                        pattern_stats[pattern]['current_gap'] / pattern_stats[pattern]['avg_gap']
                        if pattern_stats[pattern]['avg_gap'] > 0 else 0
                    )
            
            return dict(pattern_stats)
            
        except Exception as e:
            logging.error(f"Error finding pattern gaps: {str(e)}")
            return {}
            
    def predict_numbers(self, method: str = 'combined', pick_size: int = 10) -> List[int]:
        """Generate predictions based on gap analysis."""
        try:
            predictions = []
            
            if method in ['gap', 'combined']:
                # Get overdue numbers
                gap_stats = self.analyze_number_gaps()
                overdue_numbers = [
                    num for num, stats in sorted(
                        gap_stats.items(),
                        key=lambda x: x[1]['overdue_factor'],
                        reverse=True
                    )[:20]
                ]
                predictions.extend(overdue_numbers)
            
            if method in ['pattern_gap', 'combined']:
                # Get numbers from overdue patterns
                pattern_stats = self.find_pattern_gaps()
                pattern_numbers = []
                for pattern, stats in sorted(
                    pattern_stats.items(),
                    key=lambda x: x[1]['overdue_factor'],
                    reverse=True
                )[:3]:
                    # Get numbers that fit the pattern
                    recent_draws = self.data.tail(100)
                    for _, row in recent_draws.iterrows():
                        numbers = sorted(row['winning_numbers'])
                        if self._matches_pattern(pattern, numbers):
                            pattern_numbers.extend(numbers)
                predictions.extend(pattern_numbers)
            
            if method == 'combined':
                # Combine predictions with weighted voting
                votes = defaultdict(int)
                for num in predictions:
                    votes[num] += 1
                
                # Add numbers from overdue pairs
                pair_stats = self.analyze_pair_gaps()
                for pair, stats in sorted(
                    pair_stats.items(),
                    key=lambda x: x[1]['overdue_factor'],
                    reverse=True
                )[:10]:
                    votes[pair[0]] += 1
                    votes[pair[1]] += 1
            
            # Return top numbers by votes
            return [num for num, _ in sorted(votes.items(), key=lambda x: x[1], reverse=True)[:pick_size]]
            
        except Exception as e:
            logging.error(f"Error predicting numbers: {str(e)}")
            return []
            
    def _matches_pattern(self, pattern: str, numbers: List[int]) -> bool:
        """Check if numbers match a specific pattern."""
        if pattern == 'consecutive':
            return any(numbers[i+1] - numbers[i] == 1 for i in range(len(numbers)-1))
        elif pattern == 'even_heavy':
            return len([n for n in numbers if n % 2 == 0]) > len(numbers) / 2
        elif pattern == 'odd_heavy':
            return len([n for n in numbers if n % 2 == 1]) > len(numbers) / 2
        elif pattern == 'high_heavy':
            return len([n for n in numbers if n > 40]) > len(numbers) / 2
        elif pattern == 'low_heavy':
            return len([n for n in numbers if n <= 40]) > len(numbers) / 2
        return False
        
    def plot_gap_distribution(self):
        """Plot the distribution of gaps for each number."""
        try:
            gap_stats = self.analyze_number_gaps()
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle('Gap Analysis Distribution')
            
            # Plot 1: Current Gaps
            current_gaps = [stats['current_gap'] for stats in gap_stats.values()]
            axes[0, 0].hist(current_gaps, bins=30)
            axes[0, 0].set_title('Current Gaps Distribution')
            axes[0, 0].set_xlabel('Days')
            axes[0, 0].set_ylabel('Count')
            
            # Plot 2: Average Gaps
            avg_gaps = [stats['avg_gap'] for stats in gap_stats.values()]
            axes[0, 1].hist(avg_gaps, bins=30)
            axes[0, 1].set_title('Average Gaps Distribution')
            axes[0, 1].set_xlabel('Days')
            axes[0, 1].set_ylabel('Count')
            
            # Plot 3: Overdue Factors
            overdue_factors = [stats['overdue_factor'] for stats in gap_stats.values()]
            axes[1, 0].hist(overdue_factors, bins=30)
            axes[1, 0].set_title('Overdue Factors Distribution')
            axes[1, 0].set_xlabel('Factor')
            axes[1, 0].set_ylabel('Count')
            
            # Plot 4: Number of Appearances
            appearances = [stats['appearances'] for stats in gap_stats.values()]
            axes[1, 1].hist(appearances, bins=30)
            axes[1, 1].set_title('Number of Appearances Distribution')
            axes[1, 1].set_xlabel('Appearances')
            axes[1, 1].set_ylabel('Count')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'gap_distribution.png'))
            plt.close()
            
        except Exception as e:
            logging.error(f"Error plotting gap distribution: {str(e)}")
            
    def print_analysis(self):
        """Print the gap analysis results."""
        print(f"\n{'='*20} Keno Gap Analysis {'='*20}")
        print(f"Total historical draws analyzed: {len(self.data)}")
        
        # Generate visualizations
        self.plot_gap_distribution()
        print(f"\nVisualizations saved in: {self.output_dir}")
        
        print("\nMost Overdue Numbers:")
        gap_stats = self.analyze_number_gaps()
        overdue_numbers = sorted(
            gap_stats.items(),
            key=lambda x: x[1]['overdue_factor'],
            reverse=True
        )[:10]
        
        for i, (num, stats) in enumerate(overdue_numbers, 1):
            print(f"\n{i}. Number: {num}")
            print(f"   Current Gap: {stats['current_gap']} days")
            print(f"   Average Gap: {stats['avg_gap']:.1f} days")
            print(f"   Max Gap: {stats['max_gap']} days")
            print(f"   Overdue Factor: {stats['overdue_factor']:.2f}")
            print(f"   Total Appearances: {stats['appearances']}")
        
        print("\nMost Overdue Pairs:")
        pair_stats = self.analyze_pair_gaps()
        overdue_pairs = sorted(
            pair_stats.items(),
            key=lambda x: x[1]['overdue_factor'],
            reverse=True
        )[:10]
        
        for i, ((num1, num2), stats) in enumerate(overdue_pairs, 1):
            print(f"\n{i}. Pair: {num1}-{num2}")
            print(f"   Current Gap: {stats['current_gap']} days")
            print(f"   Average Gap: {stats['avg_gap']:.1f} days")
            print(f"   Max Gap: {stats['max_gap']} days")
            print(f"   Overdue Factor: {stats['overdue_factor']:.2f}")
            print(f"   Co-occurrences: {stats['co_occurrences']}")
        
        print("\nPattern Analysis:")
        pattern_stats = self.find_pattern_gaps()
        for pattern, stats in pattern_stats.items():
            print(f"\nPattern: {pattern}")
            print(f"   Current Gap: {stats['current_gap']} days")
            print(f"   Average Gap: {stats['avg_gap']:.1f} days")
            print(f"   Max Gap: {stats['max_gap']} days")
            print(f"   Overdue Factor: {stats['overdue_factor']:.2f}")
            print(f"   Total Occurrences: {stats['occurrences']}")
        
        print("\nPredictions:")
        for method in ['gap', 'pattern_gap', 'combined']:
            predictions = self.predict_numbers(method)
            print(f"\n{method.title()} Method:")
            print(f"   Recommended Numbers: {', '.join(map(str, predictions))}")
        
        print(f"\n{'='*60}\n")

def main():
    analyzer = KenoGapAnalyzer()
    
    if analyzer.data.empty:
        print("No data available. Please process some data first.")
        return
        
    analyzer.print_analysis()

if __name__ == '__main__':
    main() 