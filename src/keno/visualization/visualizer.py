"""
Visualization module for Keno data and predictions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import os

class KenoVisualizer:
    """Creates visualizations for Keno data and predictions."""
    
    def __init__(self, analyzer):
        """
        Initialize the visualizer.
        
        Args:
            analyzer: KenoAnalyzer instance
        """
        self.analyzer = analyzer
        self.save_path = os.path.expanduser('~/.keno/visualizations')
        os.makedirs(self.save_path, exist_ok=True)
        
        # Set style using matplotlib's built-in styles
        plt.style.use('seaborn-v0_8')  # Using a built-in seaborn-compatible style
        sns.set_palette("husl")
        
    def plot_frequency_heatmap(self, output_path: str):
        """
        Create a heatmap of number frequencies.
        
        Args:
            output_path: Path to save the plot
        """
        freq = self.analyzer.analyze_frequency()
        
        # Reshape data into 8x10 grid
        grid_data = np.zeros((8, 10))
        for num, count in freq.items():
            row = (num - 1) // 10
            col = (num - 1) % 10
            grid_data[row][col] = count
            
        plt.figure(figsize=(15, 10))
        sns.heatmap(grid_data, annot=True, fmt='.0f', cmap='YlOrRd')
        plt.title('Keno Number Frequency Heatmap')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.savefig(output_path)
        plt.close()
        
    def plot_pattern_analysis(self, output_path: str):
        """
        Plot pattern analysis results.
        
        Args:
            output_path: Path to save the plot
        """
        patterns = self.analyzer.analyze_patterns()
        frequencies = self.analyzer.analyze_frequency()
        
        # Ensure all numbers have frequency values
        for num in range(1, 81):
            if num not in frequencies:
                frequencies[num] = 0
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot hot numbers
        x = np.arange(len(patterns['hot_numbers']))
        width = 0.35
        
        ax1.bar(x - width/2, [frequencies[n] for n in patterns['hot_numbers']],
                width, label='Hot Numbers')
        ax1.set_title('Hot Numbers Frequency')
        ax1.set_xlabel('Number')
        ax1.set_ylabel('Frequency')
        ax1.set_xticks(x)
        ax1.set_xticklabels(patterns['hot_numbers'])
        ax1.legend()
        
        # Plot cold numbers
        x = np.arange(len(patterns['cold_numbers']))
        ax2.bar(x - width/2, [frequencies[n] for n in patterns['cold_numbers']],
                width, label='Cold Numbers')
        ax2.set_title('Cold Numbers Frequency')
        ax2.set_xlabel('Number')
        ax2.set_ylabel('Frequency')
        ax2.set_xticks(x)
        ax2.set_xticklabels(patterns['cold_numbers'])
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    def plot_prediction_comparison(self, pick_size: int, num_draws: int, output_path: str):
        """
        Plot comparison of different prediction methods.
        
        Args:
            pick_size: Number of numbers to pick
            num_draws: Number of draws to analyze
            output_path: Path to save the plot
        """
        methods = ['frequency', 'patterns', 'markov', 'due']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, method in enumerate(methods):
            # Get prediction
            prediction = self.analyzer.predict_next_draw(method, pick_size)
            
            # Get actual numbers from last draw
            actual = self.analyzer.data[-1]
            
            # Calculate matches and create matched numbers array
            matched = np.zeros(len(prediction))
            for j, num in enumerate(prediction):
                if num in actual:
                    matched[j] = num
            
            # Plot numbers
            x = np.arange(len(prediction))
            axes[i].bar(x, prediction, alpha=0.5, label='Predicted')
            axes[i].bar(x, matched, alpha=0.5, label='Matched')
            
            matches = len(set(prediction) & set(actual))
            axes[i].set_title(f'{method.capitalize()} Method\n{matches} matches')
            axes[i].set_xlabel('Index')
            axes[i].set_ylabel('Number')
            axes[i].legend()
            
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    def plot_validation_results(self,
                              validation_tracker,
                              filename: str) -> None:
        """
        Visualize validation results.
        
        Args:
            validation_tracker: ValidationTracker instance
            filename: Name of output file
        """
        if validation_tracker.history.empty:
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Accuracy over time
        for method in validation_tracker.history['method'].unique():
            method_data = validation_tracker.history[
                validation_tracker.history['method'] == method
            ]
            ax1.plot(method_data.index, method_data['accuracy'] * 100,
                    label=method, marker='o')
            
        ax1.set_title('Prediction Accuracy Over Time')
        ax1.set_xlabel('Prediction Number')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True)
        
        # Method comparison boxplot
        sns.boxplot(data=validation_tracker.history,
                   x='method', y='accuracy',
                   ax=ax2)
        ax2.set_title('Accuracy Distribution by Method')
        ax2.set_xlabel('Method')
        ax2.set_ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, filename))
        plt.close()
        
    def plot_expected_value_analysis(self,
                                   payout_table: str,
                                   filename: str) -> None:
        """
        Visualize expected value analysis.
        
        Args:
            payout_table: Name of payout table to analyze
            filename: Name of output file
        """
        pick_sizes = sorted(self.analyzer.payout_tables[payout_table].keys())
        methods = ['frequency', 'cycles', 'markov', 'due', 'ensemble']
        
        results = []
        for pick_size in pick_sizes:
            for method in methods:
                ev = self.analyzer.calculate_expected_value(
                    pick_size, payout_table, method
                )
                results.append({
                    'pick_size': pick_size,
                    'method': method,
                    'expected_value': ev
                })
                
        df = pd.DataFrame(results)
        
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=df, x='pick_size', y='expected_value',
                    hue='method', marker='o')
        plt.title('Expected Value by Pick Size and Method')
        plt.xlabel('Pick Size')
        plt.ylabel('Expected Value ($)')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, filename))
        plt.close()
        
    def create_dashboard(self, validation_tracker) -> None:
        """Create a complete analysis dashboard."""
        # Frequency analysis
        self.plot_frequency_heatmap(os.path.join(self.save_path, 'frequency_heatmap.png'))
        
        # Pattern analysis
        self.plot_pattern_analysis(os.path.join(self.save_path, 'pattern_analysis.png'))
        
        # Prediction comparison
        self.plot_prediction_comparison(4, 100, os.path.join(self.save_path, 'prediction_comparison.png'))
        
        # Validation results
        if not validation_tracker.history.empty:
            self.plot_validation_results(validation_tracker, 'validation_results.png')
            
        # Expected value analysis
        if self.analyzer.payout_tables:
            self.plot_expected_value_analysis(
                list(self.analyzer.payout_tables.keys())[0],
                'expected_value_analysis.png'
            ) 