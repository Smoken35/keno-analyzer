#!/usr/bin/env python3
"""
Keno Randomness Audit - Analyzes the randomness of Keno draws.
"""

import argparse
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from utils.stat_tests import KenoRandomnessTester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("randomness_audit.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class KenoRandomnessAuditor:
    """Main class for running Keno randomness audits."""

    def __init__(self, db_path: str):
        """Initialize the auditor with database path."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.tester = KenoRandomnessTester()
        self.connect()

    def connect(self):
        """Connect to the database."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def load_draw_data(self, start_date: str = None, end_date: str = None) -> tuple:
        """
        Load draw data from the database.

        Args:
            start_date: Optional start date in YYYY-MM-DD format
            end_date: Optional end date in YYYY-MM-DD format

        Returns:
            Tuple of (numbers, dates)
        """
        query = """
            SELECT numbers, draw_date
            FROM games
            WHERE 1=1
        """
        params = []

        if start_date:
            query += " AND draw_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND draw_date <= ?"
            params.append(end_date)

        query += " ORDER BY draw_date"

        self.cursor.execute(query, params)
        results = self.cursor.fetchall()

        numbers = []
        dates = []
        for row in results:
            numbers.append(json.loads(row[0]))
            dates.append(row[1])

        return numbers, dates

    def run_audit(self, start_date: str = None, end_date: str = None) -> Dict:
        """
        Run a complete randomness audit.

        Args:
            start_date: Optional start date in YYYY-MM-DD format
            end_date: Optional end date in YYYY-MM-DD format

        Returns:
            Dictionary containing audit results
        """
        logger.info("Starting randomness audit...")

        # Load data
        numbers, dates = self.load_draw_data(start_date, end_date)
        if not numbers:
            raise ValueError("No data found for the specified date range")

        logger.info(f"Analyzing {len(numbers)} draws...")

        # Run all tests
        results = self.tester.calculate_randomness_score(numbers, dates)

        # Add metadata
        results["metadata"] = {
            "total_draws": len(numbers),
            "start_date": dates[0] if dates else None,
            "end_date": dates[-1] if dates else None,
            "analysis_date": datetime.now().isoformat(),
        }

        logger.info("Audit completed successfully")
        return results

    def generate_report(self, results: Dict, output_file: str):
        """
        Generate a JSON report from the audit results.

        Args:
            results: Dictionary containing audit results
            output_file: Path to save the report
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save results
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Report saved to: {output_file}")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


def main():
    parser = argparse.ArgumentParser(description="Run Keno randomness audit")
    parser.add_argument("--db", default="keno.db", help="Database file path")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--output", default="reports/randomness_report.json", help="Output JSON file path"
    )

    args = parser.parse_args()

    auditor = KenoRandomnessAuditor(args.db)
    try:
        results = auditor.run_audit(args.start_date, args.end_date)
        auditor.generate_report(results, args.output)
    finally:
        auditor.close()


if __name__ == "__main__":
    main()
