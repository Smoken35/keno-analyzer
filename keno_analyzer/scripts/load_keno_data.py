#!/usr/bin/env python3
"""
Keno Data Loader - Loads and validates historical Keno game data into a database.
"""

import argparse
import csv
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('keno_data_loader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KenoDataLoader:
    """Handles loading and validation of Keno game data."""
    
    def __init__(self, db_path: str):
        """Initialize the data loader with database path."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.setup_database()
    
    def setup_database(self):
        """Create database tables if they don't exist."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Create games table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                game_id INTEGER PRIMARY KEY AUTOINCREMENT,
                draw_date TIMESTAMP,
                numbers TEXT,  -- JSON array of numbers
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create game_numbers table for efficient number-based queries
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS game_numbers (
                game_id INTEGER,
                number INTEGER,
                FOREIGN KEY (game_id) REFERENCES games(game_id),
                PRIMARY KEY (game_id, number)
            )
        ''')
        
        # Create indexes
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_draw_date ON games(draw_date)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_number ON game_numbers(number)')
        
        self.conn.commit()
    
    def validate_numbers(self, numbers: List[int]) -> bool:
        """Validate a list of Keno numbers."""
        if len(numbers) != 20:
            return False
        if not all(1 <= n <= 80 for n in numbers):
            return False
        if len(set(numbers)) != 20:  # Check for duplicates
            return False
        return True
    
    def parse_numbers(self, numbers_str: str) -> Optional[List[int]]:
        """Parse numbers from various string formats."""
        try:
            # Try parsing as JSON array
            if numbers_str.startswith('['):
                numbers = json.loads(numbers_str)
            else:
                # Try parsing as comma-separated values
                numbers = [int(n.strip()) for n in numbers_str.split(',')]
            
            # Validate the parsed numbers
            if self.validate_numbers(numbers):
                return sorted(numbers)  # Sort for consistency
            return None
        except (json.JSONDecodeError, ValueError):
            return None
    
    def load_csv(self, file_path: str, date_column: str = 'draw_date'):
        """Load data from a CSV file."""
        logger.info(f"Loading data from CSV: {file_path}")
        
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Parse draw date
                    draw_date = datetime.strptime(row[date_column], '%Y-%m-%d %H:%M:%S')
                    
                    # Parse numbers
                    numbers = self.parse_numbers(row['numbers'])
                    if not numbers:
                        logger.warning(f"Invalid numbers in row: {row}")
                        continue
                    
                    # Insert into database
                    self.cursor.execute('''
                        INSERT INTO games (draw_date, numbers, source)
                        VALUES (?, ?, ?)
                    ''', (draw_date, json.dumps(numbers), 'csv'))
                    
                    game_id = self.cursor.lastrowid
                    
                    # Insert individual numbers for indexing
                    for number in numbers:
                        self.cursor.execute('''
                            INSERT INTO game_numbers (game_id, number)
                            VALUES (?, ?)
                        ''', (game_id, number))
                    
                    self.conn.commit()
                    
                except (KeyError, ValueError) as e:
                    logger.error(f"Error processing row: {e}")
                    continue
    
    def load_json(self, file_path: str):
        """Load data from a JSON file."""
        logger.info(f"Loading data from JSON: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            for game in data:
                try:
                    draw_date = datetime.strptime(game['draw_date'], '%Y-%m-%d %H:%M:%S')
                    numbers = self.parse_numbers(game['numbers'])
                    
                    if not numbers:
                        logger.warning(f"Invalid numbers in game: {game}")
                        continue
                    
                    # Insert into database
                    self.cursor.execute('''
                        INSERT INTO games (draw_date, numbers, source)
                        VALUES (?, ?, ?)
                    ''', (draw_date, json.dumps(numbers), 'json'))
                    
                    game_id = self.cursor.lastrowid
                    
                    # Insert individual numbers for indexing
                    for number in numbers:
                        self.cursor.execute('''
                            INSERT INTO game_numbers (game_id, number)
                            VALUES (?, ?)
                        ''', (game_id, number))
                    
                    self.conn.commit()
                    
                except (KeyError, ValueError) as e:
                    logger.error(f"Error processing game: {e}")
                    continue
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

def main():
    parser = argparse.ArgumentParser(description='Load Keno game data into database')
    parser.add_argument('--input', required=True, help='Input file path')
    parser.add_argument('--format', choices=['csv', 'json'], required=True, help='Input file format')
    parser.add_argument('--db', default='keno.db', help='Database file path')
    parser.add_argument('--date-column', default='draw_date', help='Name of date column in CSV')
    
    args = parser.parse_args()
    
    loader = KenoDataLoader(args.db)
    try:
        if args.format == 'csv':
            loader.load_csv(args.input, args.date_column)
        else:
            loader.load_json(args.input)
    finally:
        loader.close()

if __name__ == '__main__':
    main() 