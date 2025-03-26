import json
import logging
import os
from datetime import datetime

import pandas as pd


class KenoDataProcessor:
    def __init__(self):
        self.setup_logging()
        self.setup_directories()

    def setup_logging(self):
        log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "keno_data")
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(log_dir, "processor.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def setup_directories(self):
        base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "keno_data")
        os.makedirs(os.path.join(base_dir, "json"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "csv"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "daily"), exist_ok=True)

    def process_uploaded_file(self, file_path):
        """Process an uploaded file and update the data structure."""
        try:
            # Determine file type and read accordingly
            file_ext = os.path.splitext(file_path)[1].lower()

            if file_ext == ".csv":
                # Read CSV with specific settings
                df = pd.read_csv(
                    file_path,
                    dtype={"draw_number": int, "draw_time": str, "winning_numbers": str},
                    parse_dates=["draw_time"],
                )
            elif file_ext == ".json":
                with open(file_path, "r") as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

            # Validate data structure
            required_columns = ["draw_number", "draw_time", "winning_numbers"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Convert winning_numbers to list if it's a string
            if isinstance(df["winning_numbers"].iloc[0], str):
                df["winning_numbers"] = df["winning_numbers"].apply(
                    lambda x: json.loads(x.replace("'", '"'))
                )

            # Validate data
            self._validate_data(df)

            # Save the processed data
            self.save_results(df)

            logging.info(f"Successfully processed {len(df)} records from {file_path}")
            return True, f"Successfully processed {len(df)} records"

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            return False, f"Error processing file: {str(e)}"

    def _validate_data(self, df):
        """Validate the data structure and content."""
        # Check draw numbers are unique
        if len(df["draw_number"].unique()) != len(df):
            raise ValueError("Duplicate draw numbers found")

        # Check winning numbers format
        for idx, row in df.iterrows():
            numbers = row["winning_numbers"]
            if not isinstance(numbers, list):
                raise ValueError(f"Invalid winning numbers format at row {idx}")
            if not all(isinstance(n, int) for n in numbers):
                raise ValueError(f"Non-integer winning numbers at row {idx}")
            if not all(1 <= n <= 80 for n in numbers):
                raise ValueError(f"Invalid number range at row {idx}")

    def save_results(self, df):
        """Save the processed results to both JSON and CSV formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "keno_data")

        # Save as JSON
        json_file = os.path.join(base_dir, "json", f"keno_results_{timestamp}.json")
        df.to_json(json_file, orient="records", indent=2)

        # Save as CSV
        csv_file = os.path.join(base_dir, "csv", f"keno_results_{timestamp}.csv")
        df.to_csv(csv_file, index=False)

        # Update all-time results
        all_time_file = os.path.join(base_dir, "keno_results_all.csv")
        try:
            if os.path.exists(all_time_file) and os.path.getsize(all_time_file) > 0:
                all_time_df = pd.read_csv(all_time_file)
                # Convert winning_numbers to list if it's a string
                if isinstance(all_time_df["winning_numbers"].iloc[0], str):
                    all_time_df["winning_numbers"] = all_time_df["winning_numbers"].apply(
                        lambda x: json.loads(x.replace("'", '"'))
                    )
                # Combine with new data and remove duplicates
                combined_df = pd.concat([all_time_df, df]).drop_duplicates(subset=["draw_number"])
                combined_df.to_csv(all_time_file, index=False)
            else:
                df.to_csv(all_time_file, index=False)
            logging.info(f"Updated all-time results file with {len(df)} new records")
        except Exception as e:
            logging.error(f"Error updating all-time results file: {str(e)}")
            # If there's an error, try to save just the new data
            df.to_csv(all_time_file, index=False)

    def get_latest_results(self, limit=10):
        """Get the most recent results from the all-time file."""
        try:
            base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "keno_data")
            all_time_file = os.path.join(base_dir, "keno_results_all.csv")
            if os.path.exists(all_time_file):
                df = pd.read_csv(all_time_file)
                return df.tail(limit).to_dict("records")
            return []
        except Exception as e:
            logging.error(f"Error getting latest results: {str(e)}")
            return []


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Process Keno data files")
    parser.add_argument("file", help="Path to the data file to process")
    args = parser.parse_args()

    processor = KenoDataProcessor()
    success, message = processor.process_uploaded_file(args.file)
    print(message)

    if success:
        print("\nLatest results:")
        latest = processor.get_latest_results()
        for result in latest:
            print(f"Draw {result['draw_number']}: {result['draw_time']}")


if __name__ == "__main__":
    main()
