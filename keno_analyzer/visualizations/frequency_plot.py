#!/usr/bin/env python3
"""
Keno Frequency Plotter - Creates visualizations of Keno number frequencies.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('frequency_plot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KenoVisualizer:
    """Creates visualizations of Keno data."""
    
    def __init__(self, db_path: str):
        """Initialize the visualizer with database path."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.connect()
    
    def connect(self):
        """Connect to the database."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
    
    def get_number_frequencies(self) -> pd.DataFrame:
        """Get frequency data for all numbers."""
        self.cursor.execute('''
            SELECT number, COUNT(*) as frequency
            FROM game_numbers
            GROUP BY number
            ORDER BY number
        ''')
        
        data = self.cursor.fetchall()
        df = pd.DataFrame(data, columns=['number', 'frequency'])
        return df
    
    def get_temporal_frequencies(self) -> pd.DataFrame:
        """Get frequency data over time."""
        self.cursor.execute('''
            SELECT 
                strftime('%Y-%m', draw_date) as month,
                number,
                COUNT(*) as frequency
            FROM games
            JOIN game_numbers ON games.game_id = game_numbers.game_id
            GROUP BY month, number
            ORDER BY month, number
        ''')
        
        data = self.cursor.fetchall()
        df = pd.DataFrame(data, columns=['month', 'number', 'frequency'])
        return df
    
    def plot_number_frequencies(self, output_file: str):
        """Create a bar plot of number frequencies."""
        df = self.get_number_frequencies()
        
        plt.figure(figsize=(15, 6))
        sns.barplot(data=df, x='number', y='frequency')
        plt.title('Keno Number Frequencies')
        plt.xlabel('Number')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(output_file)
        plt.close()
        logger.info(f"Frequency plot saved: {output_file}")
    
    def plot_temporal_heatmap(self, output_file: str):
        """Create a heatmap of number frequencies over time."""
        df = self.get_temporal_frequencies()
        
        # Pivot the data for the heatmap
        pivot_df = df.pivot(index='month', columns='number', values='frequency')
        
        plt.figure(figsize=(20, 10))
        sns.heatmap(pivot_df, cmap='YlOrRd', cbar_kws={'label': 'Frequency'})
        plt.title('Keno Number Frequencies Over Time')
        plt.xlabel('Number')
        plt.ylabel('Month')
        plt.tight_layout()
        
        plt.savefig(output_file)
        plt.close()
        logger.info(f"Temporal heatmap saved: {output_file}")
    
    def generate_visualizations(self, output_dir: str):
        """Generate all visualizations."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate frequency plot
        self.plot_number_frequencies(
            output_dir / 'number_frequencies.png'
        )
        
        # Generate temporal heatmap
        self.plot_temporal_heatmap(
            output_dir / 'temporal_heatmap.png'
        )
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate Keno visualizations')
    parser.add_argument('--db', default='keno.db', help='Database file path')
    parser.add_argument('--output-dir', default='visualizations', help='Output directory for plots')
    
    args = parser.parse_args()
    
    visualizer = KenoVisualizer(args.db)
    try:
        visualizer.generate_visualizations(args.output_dir)
    finally:
        visualizer.close()

if __name__ == '__main__':
    main() 