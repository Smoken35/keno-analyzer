"""
Script to analyze historical Keno data.
"""

import os

from keno.analysis.analyzer import KenoAnalyzer


def main():
    # Initialize analyzer
    analyzer = KenoAnalyzer("historical")

    # Load historical data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "KenoPastYears")
    analyzer.load_csv_files(data_dir)

    # Analyze frequency
    frequency = analyzer.analyze_frequency()
    print("\nNumber Frequency Analysis:")
    for num, freq in sorted(frequency.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"Number {num}: {freq} times")

    # Analyze patterns
    patterns = analyzer.analyze_patterns(window=100, pick_size=20)
    print("\nPattern Analysis (last 100 draws):")
    print("Hot numbers:", patterns["hot_numbers"])
    print("Cold numbers:", patterns["cold_numbers"])

    # Analyze cyclic patterns
    cycles = analyzer.analyze_cyclic_patterns()
    print("\nCyclic Pattern Analysis:")
    for cycle_len, prob in sorted(cycles.items())[:5]:
        print(f"Cycle length {cycle_len}: {prob:.2%} probability")

    # Analyze due numbers
    due_numbers = analyzer.analyze_due_numbers()
    print("\nDue Numbers Analysis (top 10):")
    for num, score in due_numbers[:10]:
        print(f"Number {num}: {score:.2f} due score")

    # Make predictions
    print("\nPredictions:")
    methods = ["markov", "due"]
    for method in methods:
        prediction = analyzer.predict_next_draw(method, 10)
        print(f"{method.capitalize()} prediction (10 numbers):", prediction)


if __name__ == "__main__":
    main()
