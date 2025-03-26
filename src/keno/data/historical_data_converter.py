#!/usr/bin/env python3
import glob
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


class KenoHistoricalDataConverter:
    def __init__(self):
        self.setup_logging()
        self.setup_directories()

    def setup_logging(self):
        """Set up logging configuration."""
        log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "keno_data")
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(log_dir, "data_converter.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def setup_directories(self):
        """Set up necessary directories."""
        self.keno_data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "keno_data")
        os.makedirs(self.keno_data_dir, exist_ok=True)

    def convert_file(self, input_file: str) -> pd.DataFrame:
        """Convert a single historical data file to our format."""
        try:
            # Read the CSV file
            df = pd.read_csv(input_file)

            # Extract winning numbers from columns
            number_columns = [col for col in df.columns if "NUMBER DRAWN" in col]

            # Create winning_numbers list for each row
            df["winning_numbers"] = df[number_columns].apply(
                lambda x: sorted(x.dropna().astype(int).tolist()), axis=1
            )

            # Select and rename columns
            result_df = pd.DataFrame(
                {
                    "draw_number": df["DRAW NUMBER"],
                    "draw_time": pd.to_datetime(df["DRAW DATE"]),
                    "winning_numbers": df["winning_numbers"].apply(lambda x: str(x)),
                }
            )

            return result_df

        except Exception as e:
            logging.error(f"Error converting file {input_file}: {str(e)}")
            return pd.DataFrame()

    def process_directory(self, input_dir: str) -> pd.DataFrame:
        """Process all CSV files in a directory."""
        try:
            all_data = []

            # Get all CSV files
            csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
            total_files = len(csv_files)

            print(f"Found {total_files} CSV files to process")

            # Process each file
            for i, file in enumerate(sorted(csv_files), 1):
                print(f"Processing file {i}/{total_files}: {os.path.basename(file)}")
                df = self.convert_file(file)
                if not df.empty:
                    all_data.append(df)
                    print(f"Successfully processed {len(df)} records from {os.path.basename(file)}")
                else:
                    print(f"Failed to process {os.path.basename(file)}")

            if not all_data:
                print("No data was successfully processed")
                return pd.DataFrame()

            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)

            # Sort by draw time
            combined_df = combined_df.sort_values("draw_time")

            # Remove duplicates
            combined_df = combined_df.drop_duplicates(subset=["draw_number"], keep="first")

            return combined_df

        except Exception as e:
            logging.error(f"Error processing directory {input_dir}: {str(e)}")
            return pd.DataFrame()

    def save_data(self, df: pd.DataFrame, output_file: str):
        """Save the processed data to a CSV file."""
        try:
            df.to_csv(output_file, index=False)
            print(f"Successfully saved {len(df)} records to {output_file}")

        except Exception as e:
            logging.error(f"Error saving data to {output_file}: {str(e)}")

    def process_all_sources(self):
        """Process all available data sources."""
        try:
            data_sources = [
                "/Users/user/Desktop/KenoPastYears",
                "/Users/user/CascadeProjects/b2b_solution_finder/data/KenoPastYears",
            ]

            all_data = []

            # Process each data source
            for source in data_sources:
                if os.path.exists(source):
                    print(f"\nProcessing data source: {source}")
                    df = self.process_directory(source)
                    if not df.empty:
                        all_data.append(df)

            # Process current year file separately
            current_year_file = "/Users/user/Desktop/KenoCurrentYear.csv"
            if os.path.exists(current_year_file):
                print(f"\nProcessing current year data: {current_year_file}")
                df = self.convert_file(current_year_file)
                if not df.empty:
                    all_data.append(df)

            if not all_data:
                print("No data was successfully processed from any source")
                return

            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)

            # Sort by draw time
            combined_df = combined_df.sort_values("draw_time")

            # Remove duplicates
            combined_df = combined_df.drop_duplicates(subset=["draw_number"], keep="first")

            # Save the combined data
            output_file = os.path.join(self.keno_data_dir, "keno_results_all.csv")
            self.save_data(combined_df, output_file)

            print(f"\nTotal records processed: {len(combined_df)}")
            print(
                f"Date range: {combined_df['draw_time'].min()} to {combined_df['draw_time'].max()}"
            )

        except Exception as e:
            logging.error(f"Error processing all sources: {str(e)}")


def main():
    converter = KenoHistoricalDataConverter()
    converter.process_all_sources()


if __name__ == "__main__":
    main()
