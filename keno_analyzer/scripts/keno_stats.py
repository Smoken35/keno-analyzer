#!/usr/bin/env python3
"""
Keno Stats Generator - Analyzes historical Keno game data and generates statistics.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('keno_stats.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KenoStatsGenerator:
    """Generates statistics from Keno game data."""
    
    def __init__(self, db_path: str):
        """Initialize the stats generator with database path."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.connect()
    
    def connect(self):
        """Connect to the database."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
    
    def get_basic_stats(self) -> Dict:
        """Get basic statistics about the dataset."""
        stats = {}
        
        # Total number of games
        self.cursor.execute('SELECT COUNT(*) FROM games')
        stats['total_games'] = self.cursor.fetchone()[0]
        
        # Date range
        self.cursor.execute('SELECT MIN(draw_date), MAX(draw_date) FROM games')
        min_date, max_date = self.cursor.fetchone()
        stats['first_game_date'] = min_date
        stats['last_game_date'] = max_date
        
        # Number frequency
        self.cursor.execute('''
            SELECT number, COUNT(*) as frequency
            FROM game_numbers
            GROUP BY number
            ORDER BY frequency DESC
        ''')
        number_freq = self.cursor.fetchall()
        
        # Most and least frequent numbers
        stats['most_frequent_numbers'] = [n for n, _ in number_freq[:5]]
        stats['least_frequent_numbers'] = [n for n, _ in number_freq[-5:]]
        
        # Average frequency per number
        total_occurrences = sum(freq for _, freq in number_freq)
        stats['average_frequency_per_number'] = total_occurrences / 80
        
        return stats
    
    def get_temporal_stats(self, interval: str = 'month') -> List[Dict]:
        """Get statistics over time intervals."""
        if interval == 'month':
            group_by = "strftime('%Y-%m', draw_date)"
        elif interval == 'week':
            group_by = "strftime('%Y-%W', draw_date)"
        else:
            raise ValueError("Interval must be 'month' or 'week'")
        
        self.cursor.execute(f'''
            SELECT 
                {group_by} as period,
                COUNT(*) as games_count,
                COUNT(DISTINCT game_id) as unique_games
            FROM games
            GROUP BY period
            ORDER BY period
        ''')
        
        return [
            {
                'period': row[0],
                'games_count': row[1],
                'unique_games': row[2]
            }
            for row in self.cursor.fetchall()
        ]
    
    def get_number_patterns(self) -> Dict:
        """Analyze patterns in number combinations."""
        patterns = {}
        
        # Most common pairs
        self.cursor.execute('''
            SELECT n1.number, n2.number, COUNT(*) as frequency
            FROM game_numbers n1
            JOIN game_numbers n2 ON n1.game_id = n2.game_id
            WHERE n1.number < n2.number
            GROUP BY n1.number, n2.number
            ORDER BY frequency DESC
            LIMIT 10
        ''')
        patterns['most_common_pairs'] = [
            {'numbers': (n1, n2), 'frequency': freq}
            for n1, n2, freq in self.cursor.fetchall()
        ]
        
        # Most common triplets
        self.cursor.execute('''
            SELECT n1.number, n2.number, n3.number, COUNT(*) as frequency
            FROM game_numbers n1
            JOIN game_numbers n2 ON n1.game_id = n2.game_id
            JOIN game_numbers n3 ON n1.game_id = n3.game_id
            WHERE n1.number < n2.number AND n2.number < n3.number
            GROUP BY n1.number, n2.number, n3.number
            ORDER BY frequency DESC
            LIMIT 10
        ''')
        patterns['most_common_triplets'] = [
            {'numbers': (n1, n2, n3), 'frequency': freq}
            for n1, n2, n3, freq in self.cursor.fetchall()
        ]
        
        return patterns
    
    def generate_report(self, output_file: str):
        """Generate a comprehensive report."""
        report = {
            'basic_stats': self.get_basic_stats(),
            'temporal_stats': {
                'monthly': self.get_temporal_stats('month'),
                'weekly': self.get_temporal_stats('week')
            },
            'patterns': self.get_number_patterns(),
            'generated_at': datetime.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report generated: {output_file}")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate Keno statistics report')
    parser.add_argument('--db', default='keno.db', help='Database file path')
    parser.add_argument('--output', default='keno_stats.json', help='Output JSON file path')
    
    args = parser.parse_args()
    
    generator = KenoStatsGenerator(args.db)
    try:
        generator.generate_report(args.output)
    finally:
        generator.close()

if __name__ == '__main__':
    main() 