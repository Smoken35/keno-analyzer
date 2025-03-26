"""
Example script demonstrating the Keno visualization system.
"""

import os

from keno import KenoAnalyzer, KenoVisualizer


def main():
    # Initialize the analyzer and get sample data
    analyzer = KenoAnalyzer()
    analyzer.scrape_data(source="sample")

    # Set up standard payout table
    standard_payouts = {
        2: {2: 10, 1: 0, 0: 0},
        3: {3: 25, 2: 2, 1: 0, 0: 0},
        4: {4: 100, 3: 5, 2: 1, 1: 0, 0: 0},
        5: {5: 500, 4: 15, 3: 2, 2: 0, 1: 0, 0: 0},
        6: {6: 1500, 5: 50, 4: 5, 3: 1, 2: 0, 1: 0, 0: 0},
        7: {7: 5000, 6: 100, 5: 20, 4: 2, 3: 1, 2: 0, 1: 0, 0: 0},
        8: {8: 15000, 7: 1000, 6: 100, 5: 10, 4: 2, 3: 0, 2: 0, 1: 0, 0: 0},
        9: {9: 25000, 8: 2500, 7: 150, 6: 25, 5: 5, 4: 1, 3: 0, 2: 0, 1: 0, 0: 0},
        10: {10: 100000, 9: 5000, 8: 1000, 7: 100, 6: 20, 5: 5, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0},
    }
    analyzer.set_payout_table("Standard", standard_payouts)

    # Initialize the visualizer
    visualizer = KenoVisualizer(analyzer)

    # Create output directory
    os.makedirs(visualizer.save_path, exist_ok=True)

    print("Creating individual visualizations...")

    # Create frequency heatmap
    print("- Generating frequency heatmap...")
    visualizer.plot_frequency_heatmap("frequency_heatmap.png")

    # Create pattern analysis
    print("- Generating pattern analysis...")
    visualizer.plot_patterns("pattern_analysis.png")

    # Create overdue numbers chart
    print("- Generating overdue numbers chart...")
    visualizer.plot_overdue_numbers("overdue_numbers.png")

    # Create cyclic pattern chart
    print("- Generating cyclic pattern chart...")
    visualizer.plot_cyclic_patterns("cyclic_patterns.png")

    # Create pair analysis
    print("- Generating pair analysis chart...")
    visualizer.plot_pair_analysis("pair_analysis.png")

    # Create hot numbers history
    print("- Generating hot numbers history...")
    hot_numbers = analyzer.find_hot_numbers(5)["number"].tolist()
    visualizer.plot_number_history(hot_numbers, "hot_numbers_history.png")

    # Create prediction comparison
    print("- Generating prediction comparison...")
    methods = {
        "Frequency": analyzer.predict_next_draw(method="frequency")[:10],
        "Pattern": analyzer.predict_next_draw(method="pattern")[:10],
        "Cycle": analyzer.predict_using_cycles(10),
        "Markov": analyzer.predict_using_markov_chains(10),
        "Due": analyzer.predict_using_due_theory(10),
        "Ensemble": analyzer.predict_using_ensemble(10),
    }
    visualizer.plot_prediction_comparison(methods, "prediction_comparison.png")

    # Create strategy performance comparison
    print("- Generating strategy performance comparison...")
    strategies = []
    for method_name, numbers in methods.items():
        sim = analyzer.simulate_strategy(
            "Standard", len(numbers), "custom", num_draws=100, bet_amount=1, custom_numbers=numbers
        )
        if sim:
            sim["selection_method"] = method_name
            strategies.append(sim)

    if strategies:
        visualizer.plot_strategy_performance(strategies, "strategy_performance.png")

        # Create match distribution for best strategy
        best_strategy = max(strategies, key=lambda x: x["roi"])
        visualizer.plot_match_distribution(best_strategy, "match_distribution.png")

    # Create comprehensive dashboard
    print("\nCreating comprehensive dashboard...")
    dashboard_path = visualizer.create_dashboard()
    print(f"Dashboard created at: {dashboard_path}")

    print("\nVisualization examples completed!")
    print(f"Individual visualizations saved in: {visualizer.save_path}")


if __name__ == "__main__":
    main()
