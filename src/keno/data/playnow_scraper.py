"""
Enhanced PlayNow Keno Scraper with real browser support
"""

import argparse
import json
import logging
import os
import random
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import backoff
import pandas as pd
import requests
import schedule
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from urllib3.exceptions import NewConnectionError
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)


class PlayNowScraper:
    """Scrapes and processes Keno results from PlayNow."""

    def __init__(self, output_dir: str):
        """
        Initialize the scraper.

        Args:
            output_dir: Directory to store output files
        """
        self.output_dir = output_dir
        self.daily_dir = os.path.join(output_dir, "daily")
        self.csv_dir = os.path.join(output_dir, "csv")
        self.json_dir = os.path.join(output_dir, "json")

        # Create directories if they don't exist
        os.makedirs(self.daily_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)

    def parse_winning_numbers(self, html_content: str) -> List[Dict]:
        """
        Parse winning numbers from HTML content.

        Args:
            html_content: HTML string to parse

        Returns:
            List of dictionaries containing draw information
        """
        soup = BeautifulSoup(html_content, "html.parser")
        results = []
        containers = soup.find_all("div", class_="keno-result-container")

        for container in containers:
            date_div = container.find("div", class_="date")
            time_div = container.find("div", class_="time")
            draw_number_div = container.find("div", class_="draw-number")
            numbers = [int(n.text) for n in container.find_all("div", class_="keno-number")]

            if all([date_div, time_div, draw_number_div, numbers]):
                result = {
                    "date": self._standardize_date(date_div.text),
                    "time": time_div.text,
                    "draw_number": str(draw_number_div.text),
                    "winning_numbers": sorted(numbers),
                }
                results.append(result)

        return results

    def save_results(self, results: List[Dict]) -> Tuple[str, str, str]:
        """
        Save results to files.

        Args:
            results: List of draw results

        Returns:
            Tuple of (CSV path, JSON path, daily path)
        """
        if not results:
            raise ValueError("No results to save")

        # Create file paths
        date_str = results[0]["date"]
        daily_file = os.path.join(self.daily_dir, f"keno_results_{date_str}.csv")
        consolidated_path = os.path.join(self.csv_dir, "keno_results_all.csv")
        json_file = os.path.join(self.json_dir, "keno_results.json")

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Ensure draw_number is string type
        df["draw_number"] = df["draw_number"].astype(str)

        # Save daily CSV
        df.to_csv(daily_file, index=False)

        # Update consolidated CSV
        if os.path.exists(consolidated_path):
            existing_df = pd.read_csv(consolidated_path, dtype={"draw_number": str})
            # Remove duplicates based on date and draw number
            combined_df = pd.concat([existing_df, df]).drop_duplicates(
                subset=["date", "draw_number"]
            )
            combined_df.to_csv(consolidated_path, index=False)
        else:
            df.to_csv(consolidated_path, index=False)

        # Save JSON
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2)

        return consolidated_path, json_file, daily_file

    def convert_to_standard_format(self, csv_path: str) -> str:
        """
        Convert a CSV file to standard format.

        Args:
            csv_path: Path to the CSV file

        Returns:
            Path to the converted file
        """
        # Read CSV with draw_number as string
        df = pd.read_csv(csv_path, dtype={"draw_number": str})

        # Split winning_numbers into individual columns
        if "winning_numbers" in df.columns:
            # Convert string representation of list to actual list if needed
            if isinstance(df["winning_numbers"].iloc[0], str):
                df["winning_numbers"] = df["winning_numbers"].apply(eval)

            # Create individual number columns
            for i, num in enumerate(df["winning_numbers"].iloc[0], 1):
                df[f"number_drawn_{i}"] = df["winning_numbers"].apply(lambda x: x[i - 1])

            # Drop the original winning_numbers column
            df = df.drop("winning_numbers", axis=1)

        # Standardize column names
        df.columns = [col.upper() for col in df.columns]

        # Save standardized file
        output_path = csv_path.replace(".csv", "_standardized.csv")
        df.to_csv(output_path, index=False)

        return output_path

    def _standardize_date(self, date_str: str) -> str:
        """
        Convert date string to standard format (YYYY-MM-DD).

        Args:
            date_str: Date string in various formats

        Returns:
            Standardized date string
        """
        try:
            # Try different date formats
            for fmt in ["%Y-%m-%d", "%b %d, %Y", "%B %d, %Y", "%m/%d/%Y"]:
                try:
                    return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
                except ValueError:
                    continue
            raise ValueError(f"Could not parse date: {date_str}")
        except Exception as e:
            raise ValueError(f"Error standardizing date {date_str}: {str(e)}")

    def get_latest_results(self) -> Optional[List[Dict[str, Any]]]:
        """Fetch the latest Keno results from PlayNow.com."""
        try:
            url = "https://www.playnow.com/lottery/keno/winning-numbers"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            results = []

            # Find the results table
            table = soup.find("table", {"class": "winning-numbers-table"})
            if not table:
                logging.error("Could not find results table")
                return None

            # Parse each row
            for row in table.find_all("tr")[1:]:  # Skip header row
                cols = row.find_all("td")
                if len(cols) >= 3:
                    draw_number = str(cols[0].text.strip())
                    draw_time = cols[1].text.strip()
                    numbers = [int(n.strip()) for n in cols[2].text.strip().split(",")]

                    results.append(
                        {
                            "draw_number": draw_number,
                            "draw_time": draw_time,
                            "winning_numbers": numbers,
                        }
                    )

            return results

        except Exception as e:
            logging.error(f"Error fetching results: {str(e)}")
            return None

    def update_all_time_results(self, new_results: List[Dict[str, Any]]):
        """Update the all-time results file with new data."""
        try:
            all_time_file = os.path.join(self.csv_dir, "keno_results_all.csv")

            # Load existing results if file exists
            existing_results = []
            if os.path.exists(all_time_file):
                df = pd.read_csv(all_time_file)
                df["draw_number"] = df["draw_number"].astype(str)  # Ensure draw numbers are strings
                existing_results = df.to_dict("records")

            # Combine with new results
            all_results = existing_results + new_results

            # Remove duplicates based on draw_number
            seen = set()
            unique_results = []
            for result in all_results:
                if result["draw_number"] not in seen:
                    seen.add(result["draw_number"])
                    unique_results.append(result)

            # Save updated results
            df = pd.DataFrame(unique_results)
            df["draw_number"] = df["draw_number"].astype(str)  # Ensure draw numbers are strings
            df.to_csv(all_time_file, index=False)
            logging.info(f"Updated all-time results file with {len(new_results)} new entries")

        except Exception as e:
            logging.error(f"Error updating all-time results: {str(e)}")

    def run_daily_update(self):
        """Run a daily update of Keno results."""
        logging.info("Starting daily update")

        # Add random delay to avoid detection
        time.sleep(random.uniform(1, 3))

        results = self.get_latest_results()
        if not results:
            logging.error("Failed to fetch latest results")
            return False

        if self.save_results(results):
            self.update_all_time_results(results)
            logging.info("Daily update completed successfully")
            return True

        return False


def main():
    parser = argparse.ArgumentParser(description="PlayNow Keno Scraper")
    parser.add_argument("--data-dir", default="keno_data", help="Directory to store data files")
    parser.add_argument("--daily-update", action="store_true", help="Run a daily update of results")
    args = parser.parse_args()

    scraper = PlayNowScraper(output_dir=args.data_dir)

    if args.daily_update:
        if scraper.run_daily_update():
            print("Daily update completed successfully")
        else:
            print("Daily update failed. Check the log file for details.")
    else:
        results = scraper.get_latest_results()
        if results:
            if scraper.save_results(results):
                print("Results saved successfully")
            else:
                print("Failed to save results")
        else:
            print("Failed to fetch results")


if __name__ == "__main__":
    main()
