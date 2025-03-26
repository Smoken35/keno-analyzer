#!/usr/bin/env python3
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from keno.data.data_processor import KenoDataProcessor
from keno.prediction.pattern_predictor import PatternPredictor


class PatternTestREPL:
    def __init__(self):
        self.setup_logging()
        self.data_processor = KenoDataProcessor()
        self.pattern_predictor = PatternPredictor()
        self.history = []

    def setup_logging(self):
        log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "keno_data")
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(log_dir, "pattern_test.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def load_data(self) -> pd.DataFrame:
        """Load the all-time results data."""
        try:
            base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "keno_data")
            all_time_file = os.path.join(base_dir, "keno_results_all.csv")
            if os.path.exists(all_time_file) and os.path.getsize(all_time_file) > 0:
                df = pd.read_csv(all_time_file)
                # Convert winning_numbers to list if it's a string
                if isinstance(df["winning_numbers"].iloc[0], str):
                    df["winning_numbers"] = df["winning_numbers"].apply(lambda x: eval(x))
                return df
            else:
                print("No data found. Please process some data first.")
                return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            print(f"Error loading data: {str(e)}")
            return pd.DataFrame()

    def test_pattern(self, numbers: List[int], data: pd.DataFrame) -> Dict[str, Any]:
        """Test a pattern against historical data."""
        try:
            results = self.pattern_predictor.analyze_pattern(numbers, data)
            return {
                "pattern": numbers,
                "frequency": results["frequency"],
                "confidence": results["confidence"],
                "historical_matches": results["historical_matches"],
            }
        except Exception as e:
            logging.error(f"Error testing pattern: {str(e)}")
            return {"pattern": numbers, "error": str(e)}

    def save_test_result(self, result: Dict[str, Any]):
        """Save a test result to history."""
        result["timestamp"] = datetime.now().isoformat()
        self.history.append(result)

        # Save to file
        base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "keno_data")
        history_file = os.path.join(base_dir, "pattern_test_history.json")

        try:
            import json

            with open(history_file, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving test history: {str(e)}")

    def display_result(self, result: Dict[str, Any]):
        """Display a test result in a formatted way."""
        print("\n=== Pattern Test Result ===")
        print(f"Pattern: {result['pattern']}")
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Frequency: {result['frequency']:.2f}%")
            print(f"Confidence: {result['confidence']:.2f}%")
            print("\nHistorical Matches:")
            for match in result["historical_matches"]:
                print(f"  Draw {match['draw_number']}: {match['draw_time']}")
                print(f"  Matches: {match['matches']}")
        print("========================\n")

    def run(self):
        """Run the REPL interface."""
        print("Welcome to the Keno Pattern Test REPL!")
        print("Enter 'help' for available commands.")
        print("Enter 'exit' to quit.")

        data = self.load_data()
        if data.empty:
            print("No data available. Please process some data first.")
            return

        while True:
            try:
                command = input("\nEnter command: ").strip().lower()

                if command == "exit":
                    break
                elif command == "help":
                    self.show_help()
                elif command == "test":
                    numbers = self.get_numbers_input()
                    if numbers:
                        result = self.test_pattern(numbers, data)
                        self.save_test_result(result)
                        self.display_result(result)
                elif command == "history":
                    self.show_history()
                elif command == "clear":
                    self.history = []
                    print("History cleared.")
                else:
                    print("Unknown command. Type 'help' for available commands.")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logging.error(f"Error in REPL: {str(e)}")
                print(f"Error: {str(e)}")

    def show_help(self):
        """Display help information."""
        print("\nAvailable Commands:")
        print("  test    - Test a pattern of numbers")
        print("  history - Show test history")
        print("  clear   - Clear test history")
        print("  help    - Show this help message")
        print("  exit    - Exit the program")

    def get_numbers_input(self) -> List[int]:
        """Get numbers input from user."""
        try:
            input_str = input("Enter numbers (comma-separated): ").strip()
            numbers = [int(n.strip()) for n in input_str.split(",")]

            # Validate numbers
            if not all(1 <= n <= 80 for n in numbers):
                print("Error: Numbers must be between 1 and 80")
                return None

            return numbers
        except ValueError:
            print("Error: Please enter valid numbers separated by commas")
            return None

    def show_history(self):
        """Display test history."""
        if not self.history:
            print("No test history available.")
            return

        print("\n=== Test History ===")
        for i, result in enumerate(self.history, 1):
            print(f"\nTest {i}:")
            print(f"Pattern: {result['pattern']}")
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Frequency: {result['frequency']:.2f}%")
                print(f"Confidence: {result['confidence']:.2f}%")
        print("\n==================\n")


def main():
    repl = PatternTestREPL()
    repl.run()


if __name__ == "__main__":
    main()


def parse_numbers(input_str: str) -> Optional[List[int]]:
    """Parse numbers from input string."""
    try:
        numbers = [int(x.strip()) for x in input_str.split(",")]
        if not all(1 <= x <= 80 for x in numbers):
            print("Error: All numbers must be between 1 and 80")
            return None
        return numbers
    except ValueError:
        print("Error: Invalid number format")
        return None


def validate_numbers(numbers: List[int]) -> List[int]:
    """Validate list of numbers."""
    if not numbers:
        print("Error: Empty number list")
        return []
    if not all(1 <= x <= 80 for x in numbers):
        print("Error: All numbers must be between 1 and 80")
        return []
    return numbers
