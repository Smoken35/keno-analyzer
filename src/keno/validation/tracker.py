"""
Keno Validation and Tracking System.
Provides tools for validating predictions, tracking performance, and analyzing results.
"""

import os
import sqlite3
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class ValidationTracker:
    """
    Tracks and validates Keno predictions, providing performance analysis and reporting.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the validation tracker.
        
        Args:
            db_path: Optional path to SQLite database. If None, uses default path.
        """
        if db_path is None:
            # Default to user's home directory
            home = os.path.expanduser("~")
            db_dir = os.path.join(home, ".keno")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "validation.db")
        
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Drop existing tables
            cursor.execute("DROP TABLE IF EXISTS predictions")
            
            # Create predictions table with updated schema
            cursor.execute("""
                CREATE TABLE predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    method TEXT NOT NULL,
                    predicted_numbers TEXT NOT NULL,
                    draw_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    validated BOOLEAN DEFAULT FALSE,
                    actual_numbers TEXT,
                    matches INTEGER,
                    accuracy REAL
                )
            """)
            conn.commit()
    
    def record_prediction(
        self,
        method: str,
        predicted_numbers: List[int],
        draw_id: Optional[str] = None
    ) -> int:
        """
        Record a new prediction.
        
        Args:
            method: Prediction method used
            predicted_numbers: List of predicted numbers
            draw_id: Optional identifier for the draw
            
        Returns:
            ID of the recorded prediction
        """
        # Convert numpy types to Python native types
        predicted_numbers = [int(num) for num in predicted_numbers]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO predictions (method, predicted_numbers, draw_id)
                VALUES (?, ?, ?)
                """,
                (method, json.dumps(predicted_numbers), draw_id)
            )
            conn.commit()
            return cursor.lastrowid
    
    def validate_prediction(
        self,
        prediction_id: int,
        actual_numbers: List[int]
    ) -> Dict:
        """
        Validate a prediction against actual numbers.
        
        Args:
            prediction_id: ID of the prediction to validate
            actual_numbers: List of actual drawn numbers
            
        Returns:
            Dict containing validation results
        """
        # Convert numpy types to Python native types
        actual_numbers = [int(num) for num in actual_numbers]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get prediction
            cursor.execute(
                "SELECT method, predicted_numbers FROM predictions WHERE id = ?",
                (prediction_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                raise ValueError(f"No prediction found with ID {prediction_id}")
                
            method, predicted_numbers = result
            predicted_numbers = json.loads(predicted_numbers)
            
            # Calculate matches
            matches = len(set(predicted_numbers) & set(actual_numbers))
            accuracy = matches / len(predicted_numbers)
            
            # Update prediction
            cursor.execute(
                """
                UPDATE predictions
                SET validated = TRUE,
                    actual_numbers = ?,
                    matches = ?,
                    accuracy = ?
                WHERE id = ?
                """,
                (json.dumps(actual_numbers), matches, accuracy, prediction_id)
            )
            conn.commit()
            
            return {
                'matches': matches,
                'accuracy': accuracy,
                'method': method
            }
    
    def analyze_historical_performance(
        self,
        method: str,
        pick_size: int,
        analyzer: 'KenoAnalyzer',
        num_draws: int = 100
    ) -> Dict:
        """
        Analyze historical performance of a prediction method.
        
        Args:
            method: Prediction method to analyze
            pick_size: Number of numbers to pick
            analyzer: KenoAnalyzer instance
            num_draws: Number of historical draws to analyze
            
        Returns:
            Dict containing performance metrics
        """
        matches_list = []
        
        # Use only the last num_draws draws
        historical_draws = analyzer.data[-num_draws:]
        
        for i in range(len(historical_draws) - 1):
            # Make prediction using data up to current draw
            analyzer.data = historical_draws[:i+1]
            prediction = analyzer.predict_next_draw(method)[:pick_size]
            
            # Compare with next draw
            actual = historical_draws[i+1]
            matches = len(set(prediction) & set(actual))
            matches_list.append(matches)
        
        # Calculate statistics
        avg_matches = np.mean(matches_list)
        p_value = self._calculate_significance(avg_matches, len(matches_list))
        
        return {
            'avg_matches': avg_matches,
            'p_value': p_value,
            'num_draws': len(matches_list)
        }
    
    def compare_methods(
        self,
        analyzer: 'KenoAnalyzer',
        pick_size: int = 4,
        num_draws: int = 100
    ) -> pd.DataFrame:
        """
        Compare performance of different prediction methods.
        
        Args:
            analyzer: KenoAnalyzer instance
            pick_size: Number of numbers to pick
            num_draws: Number of historical draws to analyze
            
        Returns:
            DataFrame with comparison results
        """
        methods = ['frequency', 'patterns', 'markov', 'due']
        results = []
        
        for method in methods:
            validation = self.analyze_historical_performance(
                method,
                pick_size,
                analyzer,
                num_draws
            )
            results.append({
                'method': method,
                'avg_matches': validation['avg_matches'],
                'p_value': validation['p_value']
            })
            
        return pd.DataFrame(results)
    
    def _calculate_significance(
        self,
        avg_matches: float,
        n: int
    ) -> float:
        """
        Calculate statistical significance of prediction results.
        
        Args:
            avg_matches: Average number of matches
            n: Number of predictions
            
        Returns:
            P-value from binomial test
        """
        # Use binomtest instead of deprecated binom_test
        result = stats.binomtest(
            k=int(avg_matches * n),  # Total successes
            n=n * 20,  # Total trials (20 numbers per prediction)
            p=20/80,  # Expected probability
            alternative='greater'
        )
        return result.pvalue
    
    def get_method_performance(
        self,
        method: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get performance statistics for prediction methods.
        
        Args:
            method: Optional method name to filter by
            
        Returns:
            DataFrame with performance statistics
        """
        query = """
            SELECT 
                p.method,
                COUNT(*) as predictions,
                AVG(p.matches) as avg_matches,
                AVG(p.accuracy) as avg_accuracy,
                MIN(p.matches) as min_matches,
                MAX(p.matches) as max_matches
            FROM predictions p
            WHERE p.validated = TRUE
        """
        
        if method:
            query += f" AND p.method = '{method}'"
        
        query += " GROUP BY p.method"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn)
        
        # Calculate additional statistics
        if not df.empty:
            df['accuracy_pct'] = df['avg_accuracy'] * 100
            
            # Calculate p-values
            df['p_value'] = df.apply(
                lambda row: self._calculate_significance(
                    row['avg_matches'],
                    row['predictions']
                ),
                axis=1
            )
            
            df['significant'] = df['p_value'] < 0.05
        
        return df
    
    def get_prediction_history(
        self,
        method: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get prediction history with validation results.
        
        Args:
            method: Optional method name to filter by
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with prediction history
        """
        query = """
            SELECT 
                p.id,
                p.method,
                p.predicted_numbers,
                p.draw_id,
                p.timestamp,
                p.matches,
                p.accuracy,
                v.actual_numbers,
                v.validation_date
            FROM predictions p
            LEFT JOIN validation_results v ON p.id = v.prediction_id
            WHERE p.validated = TRUE
        """
        
        if method:
            query += f" AND p.method = '{method}'"
        
        query += f" ORDER BY p.timestamp DESC LIMIT {limit}"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn)
        
        # Convert JSON strings to lists
        df['predicted_numbers'] = df['predicted_numbers'].apply(json.loads)
        df['actual_numbers'] = df['actual_numbers'].apply(json.loads)
        
        return df
    
    def plot_method_comparison(
        self,
        comparison_df: pd.DataFrame,
        filename: str
    ) -> None:
        """
        Create a visualization comparing method performance.
        
        Args:
            comparison_df: DataFrame from compare_methods()
            filename: Output file path
        """
        plt.figure(figsize=(12, 6))
        
        # Create bar plot
        ax = sns.barplot(
            data=comparison_df,
            x='method',
            y='accuracy_pct',
            palette='viridis'
        )
        
        # Add significance markers
        for i, significant in enumerate(comparison_df['significant']):
            color = 'green' if significant else 'red'
            marker = '★' if significant else '✗'
            plt.text(i, comparison_df['accuracy_pct'].iloc[i], marker,
                    color=color, ha='center', va='bottom')
        
        plt.title('Prediction Method Comparison')
        plt.xlabel('Method')
        plt.ylabel('Accuracy (%)')
        plt.xticks(rotation=45)
        
        # Add baseline
        baseline = (20/80) * 100  # Expected accuracy by chance
        plt.axhline(y=baseline, color='r', linestyle='--', alpha=0.5,
                   label='Random Chance')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    
    def plot_prediction_history(
        self,
        method: str,
        filename: str,
        limit: int = 50
    ) -> None:
        """
        Create a visualization of prediction history.
        
        Args:
            method: Prediction method to plot
            filename: Output file path
            limit: Number of predictions to include
        """
        history_df = self.get_prediction_history(method=method, limit=limit)
        
        plt.figure(figsize=(12, 6))
        
        # Plot accuracy over time
        plt.plot(range(len(history_df)), history_df['accuracy'],
                marker='o', linestyle='-', alpha=0.6)
        
        # Add trend line
        z = np.polyfit(range(len(history_df)), history_df['accuracy'], 1)
        p = np.poly1d(z)
        plt.plot(range(len(history_df)), p(range(len(history_df))),
                "r--", alpha=0.8, label='Trend')
        
        plt.title(f'Prediction Accuracy History: {method}')
        plt.xlabel('Prediction Number')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    
    def generate_validation_report(
        self,
        output_dir: str = "validation_reports"
    ) -> str:
        """
        Generate a comprehensive validation report.
        
        Args:
            output_dir: Directory for report files
            
        Returns:
            str: Path to report directory
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(output_dir, f"report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Get performance data
        performance_df = self.get_method_performance()
        performance_df.to_csv(
            os.path.join(report_dir, "method_performance.csv"),
            index=False
        )
        
        # Create performance plot
        self.plot_method_comparison(
            performance_df,
            os.path.join(report_dir, "method_comparison.png")
        )
        
        # Generate history plots for each method
        for method in performance_df['method']:
            self.plot_prediction_history(
                method=method,
                filename=os.path.join(report_dir, f"{method}_history.png")
            )
        
        # Create summary report
        summary = {
            'generated_at': datetime.now().isoformat(),
            'total_predictions': int(performance_df['predictions'].sum()),
            'best_method': performance_df.loc[
                performance_df['avg_accuracy'].idxmax(),
                'method'
            ],
            'method_stats': performance_df.to_dict('records')
        }
        
        with open(os.path.join(report_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        return report_dir 