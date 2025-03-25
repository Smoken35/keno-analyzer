"""
Example script demonstrating the enhanced Keno analyzer functionality.
"""

from analyzer import KenoAnalyzer

def main():
    # Initialize the analyzer
    analyzer = KenoAnalyzer()
    
    # Get sample data (replace with actual URL for real data)
    analyzer.scrape_data(source="sample")
    
    # Set payout table for a standard Keno game
    # This is just an example - actual payouts vary by casino/lottery
    standard_payouts = {
        2: {2: 10, 1: 0, 0: 0},
        3: {3: 25, 2: 2, 1: 0, 0: 0},
        4: {4: 100, 3: 5, 2: 1, 1: 0, 0: 0},
        5: {5: 500, 4: 15, 3: 2, 2: 0, 1: 0, 0: 0},
        6: {6: 1500, 5: 50, 4: 5, 3: 1, 2: 0, 1: 0, 0: 0},
        7: {7: 5000, 6: 100, 5: 20, 4: 2, 3: 1, 2: 0, 1: 0, 0: 0},
        8: {8: 15000, 7: 1000, 6: 100, 5: 10, 4: 2, 3: 0, 2: 0, 1: 0, 0: 0},
        9: {9: 25000, 8: 2500, 7: 150, 6: 25, 5: 5, 4: 1, 3: 0, 2: 0, 1: 0, 0: 0},
        10: {10: 100000, 9: 5000, 8: 1000, 7: 100, 6: 20, 5: 5, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0}
    }
    analyzer.set_payout_table("Standard", standard_payouts)
    
    # Perform comprehensive analysis
    print("\nAnalyzing Standard Keno game...")
    analysis = analyzer.analyze_game("Standard", bet_amount=1.0)
    
    if analysis:
        # Print recommendations
        rec = analysis['recommended_play']
        print("\nRecommended Play Strategy:")
        print(f"Pick Size: {rec['pick_size']} numbers")
        print(f"Expected Return: {rec['expected_return']:.2f}%")
        print(f"Best Strategy: {rec['strategy']}")
        print(f"Optimal Number of Draws: {rec['optimal_draws']}")
        print(f"Recommended Numbers: {rec['recommended_numbers']}")
        print(f"Expected ROI: {rec['expected_roi']:.2f}%")
        
        # Print probability analysis
        print("\nProbability Analysis:")
        for matches, prob in analysis['probability_analysis'].items():
            print(f"{matches} matches: {prob:.6f}")
        
        # Print strategy comparison summary
        print("\nStrategy Comparison Summary:")
        for strat in analysis['strategy_comparison'][:3]:  # Show top 3
            print(f"Strategy: {strat['Strategy']}")
            print(f"Number of Draws: {strat['Draws']}")
            print(f"ROI: {strat['ROI %']:.2f}%")
            print(f"Net Profit: ${strat['Net Profit']:.2f}")
            print()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    analyzer.visualize_analysis("Standard")
    
    # Export analysis
    print("\nExporting analysis...")
    analyzer.export_analysis("Standard", "keno_analysis.csv")

if __name__ == "__main__":
    main() 