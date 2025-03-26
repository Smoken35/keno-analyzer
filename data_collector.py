import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("keno_data/collection.log"), logging.StreamHandler()],
)


class KenoDataCollector:
    def __init__(self):
        self.base_url = "https://www.keno.com.au/history"
        self.output_dir = Path("keno_data")
        self.output_dir.mkdir(exist_ok=True)

    def fetch_draw_data(self, date):
        """Fetch Keno draw data for a specific date."""
        try:
            params = {"date": date.strftime("%Y-%m-%d"), "draw": "all"}
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            draws = []

            # Parse the draw data from the HTML
            draw_elements = soup.find_all("div", class_="draw-result")
            for draw in draw_elements:
                draw_time = draw.find("div", class_="draw-time").text.strip()
                numbers = [int(num.text) for num in draw.find_all("div", class_="number")]
                draws.append({"date": date, "time": draw_time, "numbers": numbers})

            return draws

        except Exception as e:
            logging.error(f"Error fetching data for {date}: {str(e)}")
            return []

    def collect_historical_data(self, start_date, end_date):
        """Collect historical Keno data for a date range."""
        all_draws = []
        current_date = start_date

        while current_date <= end_date:
            logging.info(f"Fetching data for {current_date}")
            draws = self.fetch_draw_data(current_date)
            all_draws.extend(draws)

            # Add a delay to avoid overwhelming the server
            time.sleep(1)
            current_date += timedelta(days=1)

        # Convert to DataFrame and save
        df = pd.DataFrame(all_draws)
        output_file = (
            self.output_dir
            / f"keno_data_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        )
        df.to_csv(output_file, index=False)
        logging.info(f"Data saved to {output_file}")

        return df


def main():
    collector = KenoDataCollector()

    # Collect last 30 days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    df = collector.collect_historical_data(start_date, end_date)
    logging.info(f"Collected {len(df)} draws")


if __name__ == "__main__":
    main()
