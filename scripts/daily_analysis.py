"""
Daily Keno analysis script for optimal consecutive payouts.
"""

import os
import logging
from datetime import datetime
from keno.data.fetcher import KenoDataFetcher
from keno.analysis.analyzer import KenoAnalyzer
from performance_tracker import PerformanceTracker
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('keno_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_payout_table(analyzer):
    """Set up standard Keno payout table."""
    payout_table = {
        5: {5: 500, 4: 15, 3: 2, 2: 0, 1: 0, 0: 0},
        10: {10: 2000, 9: 200, 8: 50, 7: 10, 6: 2, 5: 1, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0},
        15: {15: 10000, 14: 1500, 13: 500, 12: 100, 11: 25, 10: 10, 9: 5, 8: 2, 7: 1, 6: 0, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0},
        20: {20: 100000, 19: 10000, 18: 2500, 17: 500, 16: 100, 15: 50, 14: 20, 13: 10, 12: 5, 11: 2, 10: 1, 9: 0, 8: 0, 7: 0, 6: 0, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0}
    }
    analyzer.set_payout_table(payout_table)

def main():
    # Initialize data fetcher and analyzer
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "KenoPastYears")
    fetcher = KenoDataFetcher(data_dir)
    analyzer = KenoAnalyzer("historical")
    
    # Update historical data
    logger.info("Updating historical data...")
    if not fetcher.update_historical_data():
        logger.error("Failed to update historical data")
        return
    
    # Load data into analyzer
    analyzer.load_csv_files(data_dir)
    setup_payout_table(analyzer)
    
    # Initialize performance tracker
    tracker = PerformanceTracker(data_dir)
    
    # Run performance analysis
    logger.info("Running performance analysis...")
    performance = tracker.analyze_performance(days=100)
    
    if performance:
        # Print results
        print("\n=== Daily Analysis Results ===")
        print(f"\nAnalysis Date: {performance['date']}")
        print(f"Days Analyzed: {performance['days_analyzed']}")
        print(f"Total Games: {performance['total_games']}")
        
        print("\nBest Strategy:")
        print(f"Pick Size: {performance['best_strategy']['pick_size']}")
        print(f"Method: {performance['best_strategy']['method']}")
        print(f"ROI: {performance['best_strategy']['roi']:.2f}%")
        
        print("\nRecent Performance:")
        recent = performance['recent_performance']
        print(f"Win Rate: {recent['win_percentage']:.2f}%")
        print(f"Net Profit: ${recent['net_profit']:.2f}")
        print(f"ROI: {recent['roi']:.2f}%")
        
        print("\nConsecutive Payouts by Method:")
        for method, stats in performance['consecutive_payouts'].items():
            print(f"\n{method.capitalize()} Method:")
            print(f"Maximum Streak: {stats['max_streak']}")
            print(f"Average Streak: {stats['avg_streak']:.2f}")
            print(f"ROI: {stats['roi']:.2f}%")
        
        # Generate predictions for next draw
        best_method = performance['best_strategy']['method']
        best_pick_size = performance['best_strategy']['pick_size']
        prediction = analyzer.predict_next_draw(best_method, best_pick_size)
        
        print(f"\nRecommended Numbers for Next Draw:")
        print(f"Numbers: {sorted(prediction)}")
        
        # Generate plots
        output_dir = 'strategy_analysis'
        tracker.plot_performance(output_dir)
        print(f"\nPerformance plots saved to {output_dir}/")
        
        # Save results to file
        with open(os.path.join(output_dir, 'daily_results.txt'), 'w') as f:
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\nBest Strategy:\n")
            f.write(f"Pick Size: {best_pick_size}\n")
            f.write(f"Method: {best_method}\n")
            f.write(f"Expected ROI: {performance['best_strategy']['roi']:.2f}%\n")
            f.write(f"\nRecommended Numbers: {sorted(prediction)}\n")
            f.write("\nPerformance by Method:\n")
            for method, stats in performance['consecutive_payouts'].items():
                f.write(f"\n{method.capitalize()} Method:\n")
                f.write(f"ROI: {stats['roi']:.2f}%\n")
                f.write(f"Maximum Streak: {stats['max_streak']}\n")
                f.write(f"Average Streak: {stats['avg_streak']:.2f}\n")
            f.write(f"\nRecent Performance:\n")
            f.write(f"Win Rate: {recent['win_percentage']:.2f}%\n")
            f.write(f"Net Profit: ${recent['net_profit']:.2f}\n")
            f.write(f"ROI: {recent['roi']:.2f}%\n")
    else:
        print("Failed to complete performance analysis")

if __name__ == "__main__":
    main() 