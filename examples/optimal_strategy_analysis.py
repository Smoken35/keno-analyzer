"""
Analysis of optimal Keno betting strategies and consecutive payouts.
"""

from keno.analysis.analyzer import KenoAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from datetime import datetime
from collections import defaultdict

def setup_payout_table(analyzer):
    """Set up standard Keno payout table."""
    payout_table = {
        5: {5: 500, 4: 15, 3: 2, 2: 0, 1: 0, 0: 0},
        10: {10: 2000, 9: 200, 8: 50, 7: 10, 6: 2, 5: 1, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0},
        15: {15: 10000, 14: 1500, 13: 500, 12: 100, 11: 25, 10: 10, 9: 5, 8: 2, 7: 1, 6: 0, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0},
        20: {20: 100000, 19: 10000, 18: 2500, 17: 500, 16: 100, 15: 50, 14: 20, 13: 10, 12: 5, 11: 2, 10: 1, 9: 0, 8: 0, 7: 0, 6: 0, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0}
    }
    analyzer.set_payout_table(payout_table)

def analyze_consecutive_payouts(analyzer, pick_size, num_simulations=1000):
    """Analyze consecutive payouts for different betting strategies."""
    methods = ['frequency', 'patterns', 'markov', 'due']
    results = defaultdict(list)
    
    for method in methods:
        consecutive_payouts = []
        current_streak = 0
        
        for _ in range(num_simulations):
            prediction = analyzer.predict_next_draw(method, pick_size)
            actual_draw = analyzer.data[-1]  # Use last draw as actual
            matches = len(set(prediction) & set(actual_draw))
            
            # Calculate payout for this draw
            payout = analyzer.calculate_payout(pick_size, matches)
            
            if payout > 0:
                current_streak += 1
                consecutive_payouts.append(current_streak)
            else:
                current_streak = 0
                consecutive_payouts.append(0)
        
        results[method] = consecutive_payouts
    
    return results

def analyze_optimal_pick_size(analyzer, num_simulations=1000):
    """Analyze which pick size provides the best ROI."""
    pick_sizes = [5, 10, 15, 20]
    results = []
    
    for pick_size in pick_sizes:
        for method in ['frequency', 'patterns', 'markov', 'due']:
            sim_result = analyzer.simulate_strategy(method, pick_size, num_simulations=num_simulations)
            results.append({
                'pick_size': pick_size,
                'method': method,
                'roi': sim_result['roi_percent'],
                'avg_matches': np.mean([m for m, c in sim_result['match_distribution'].items() for _ in range(c)]),
                'max_consecutive_payouts': max(analyze_consecutive_payouts(analyzer, pick_size, num_simulations=100)[method])
            })
    
    return pd.DataFrame(results)

def plot_consecutive_payouts(results, save_path):
    """Plot consecutive payouts for different methods."""
    plt.figure(figsize=(12, 6))
    for method, payouts in results.items():
        plt.plot(payouts, label=method)
    
    plt.title('Consecutive Payouts by Prediction Method')
    plt.xlabel('Draw Number')
    plt.ylabel('Consecutive Payouts')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    # Create output directory for plots
    output_dir = 'strategy_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer and load data
    analyzer = KenoAnalyzer("historical")
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "KenoPastYears")
    analyzer.load_csv_files(data_dir)
    
    # Set up payout table
    setup_payout_table(analyzer)
    
    # 1. Analyze optimal pick size
    print("\n=== Optimal Pick Size Analysis ===")
    optimal_results = analyze_optimal_pick_size(analyzer)
    
    # Plot ROI by pick size and method
    plt.figure(figsize=(12, 6))
    for method in optimal_results['method'].unique():
        method_data = optimal_results[optimal_results['method'] == method]
        plt.plot(method_data['pick_size'], method_data['roi'], marker='o', label=method)
    
    plt.title('ROI by Pick Size and Prediction Method')
    plt.xlabel('Pick Size')
    plt.ylabel('ROI (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'roi_by_pick_size.png'))
    plt.close()
    
    # Print optimal results
    print("\nOptimal Strategy Analysis:")
    print(optimal_results.to_string(index=False))
    
    # Find best performing strategy
    best_strategy = optimal_results.loc[optimal_results['roi'].idxmax()]
    print(f"\nBest Performing Strategy:")
    print(f"Pick Size: {best_strategy['pick_size']}")
    print(f"Method: {best_strategy['method']}")
    print(f"ROI: {best_strategy['roi']:.2f}%")
    print(f"Average Matches: {best_strategy['avg_matches']:.2f}")
    print(f"Max Consecutive Payouts: {best_strategy['max_consecutive_payouts']}")
    
    # 2. Analyze consecutive payouts for best strategy
    print("\n=== Consecutive Payouts Analysis ===")
    consecutive_results = analyze_consecutive_payouts(
        analyzer, 
        best_strategy['pick_size'],
        num_simulations=1000
    )
    
    # Plot consecutive payouts
    plot_consecutive_payouts(
        consecutive_results,
        os.path.join(output_dir, 'consecutive_payouts.png')
    )
    
    # Print consecutive payout statistics
    print("\nConsecutive Payouts Statistics:")
    for method, payouts in consecutive_results.items():
        max_streak = max(payouts)
        avg_streak = np.mean(payouts)
        print(f"\n{method.capitalize()} Method:")
        print(f"Maximum Consecutive Payouts: {max_streak}")
        print(f"Average Consecutive Payouts: {avg_streak:.2f}")
    
    # 3. Generate optimal betting recommendations
    print("\n=== Optimal Betting Recommendations ===")
    print(f"\nBased on historical data analysis:")
    print(f"1. Optimal Pick Size: {best_strategy['pick_size']} numbers")
    print(f"2. Recommended Prediction Method: {best_strategy['method']}")
    print(f"3. Expected ROI: {best_strategy['roi']:.2f}%")
    print(f"4. Average Matches per Draw: {best_strategy['avg_matches']:.2f}")
    print(f"5. Maximum Consecutive Payouts: {best_strategy['max_consecutive_payouts']}")
    
    # Generate next draw prediction using best strategy
    prediction = analyzer.predict_next_draw(
        best_strategy['method'],
        best_strategy['pick_size']
    )
    print(f"\nRecommended Numbers for Next Draw:")
    print(f"Numbers: {sorted(prediction)}")

if __name__ == "__main__":
    main() 