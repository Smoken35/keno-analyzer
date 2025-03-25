#!/usr/bin/env python3
"""
Cluster Analysis Script - Analyzes Keno number clusters using graph-based clustering.
"""

import argparse
import logging
import sqlite3
from typing import List, Set
from pathlib import Path
from datetime import datetime, timedelta

from ..utils.cluster_builder import ClusterBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cluster_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_transactions(db_path: str, 
                     start_date: str = None, 
                     end_date: str = None) -> List[Set[int]]:
    """
    Load Keno transactions from database.
    
    Args:
        db_path: Path to SQLite database
        start_date: Start date for analysis (YYYY-MM-DD)
        end_date: End date for analysis (YYYY-MM-DD)
        
    Returns:
        List of transaction sets
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Build query
    query = """
    SELECT DISTINCT n.number
    FROM games g
    JOIN game_numbers n ON g.id = n.game_id
    WHERE 1=1
    """
    params = []
    
    if start_date:
        query += " AND g.draw_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND g.draw_date <= ?"
        params.append(end_date)
    
    query += " ORDER BY g.draw_date, g.id"
    
    # Execute query
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    # Group numbers by game
    transactions = []
    current_game = []
    current_game_id = None
    
    for row in rows:
        game_id, number = row
        if current_game_id != game_id:
            if current_game:
                transactions.append(set(current_game))
            current_game = [number]
            current_game_id = game_id
        else:
            current_game.append(number)
    
    # Add last game
    if current_game:
        transactions.append(set(current_game))
    
    conn.close()
    return transactions

def main():
    """Main function to run cluster analysis."""
    parser = argparse.ArgumentParser(description='Analyze Keno number clusters')
    parser.add_argument('--db', required=True, help='Path to SQLite database')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--n-clusters', type=int, default=10, 
                       help='Number of clusters to create')
    parser.add_argument('--min-cohesion', type=float, default=0.3,
                       help='Minimum cluster cohesion threshold')
    parser.add_argument('--min-coverage', type=float, default=0.2,
                       help='Minimum cluster coverage threshold')
    parser.add_argument('--output', default='results/clusters.json',
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Load transactions
    logger.info("Loading transactions from database...")
    transactions = load_transactions(
        args.db,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    if not transactions:
        logger.error("No transactions found for specified date range")
        return
    
    logger.info(f"Loaded {len(transactions)} transactions")
    
    # Initialize and run cluster builder
    cluster_builder = ClusterBuilder(
        min_cohesion=args.min_cohesion,
        min_coverage=args.min_coverage
    )
    
    clusters = cluster_builder.build_clusters(
        transactions,
        n_clusters=args.n_clusters
    )
    
    # Save results
    cluster_builder.save_results(args.output)
    
    # Print summary
    logger.info(f"Analysis complete. Found {len(clusters)} clusters")
    logger.info(f"Results saved to {args.output}")
    
    # Print top clusters
    top_clusters = cluster_builder.get_top_clusters(n=5)
    logger.info("\nTop 5 clusters:")
    for i, cluster in enumerate(top_clusters, 1):
        logger.info(f"\nCluster {i}:")
        logger.info(f"Numbers: {cluster['numbers']}")
        logger.info(f"Quality Score: {cluster['quality_score']:.3f}")
        logger.info(f"Cohesion: {cluster['cohesion']:.3f}")
        logger.info(f"Coverage: {cluster['coverage']:.3f}")
        logger.info(f"Frequency: {cluster['frequency']:.3f}")

if __name__ == '__main__':
    main() 