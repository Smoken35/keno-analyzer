#!/usr/bin/env python3
"""
Keno Randomness Visualization - Creates plots for randomness audit results.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('randomness_plots.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KenoRandomnessVisualizer:
    """Class for creating visualizations of Keno randomness audit results."""
    
    def __init__(self, report_path: str):
        """Initialize the visualizer with the report path."""
        self.report_path = Path(report_path)
        self.results = self._load_report()
        self.output_dir = self.report_path.parent / 'plots'
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def _load_report(self) -> Dict:
        """Load the audit report from JSON."""
        with open(self.report_path, 'r') as f:
            return json.load(f)
    
    def plot_entropy_distribution(self):
        """Plot the distribution of Shannon entropy values."""
        plt.figure(figsize=(10, 6))
        
        # Create histogram of entropy values
        sns.histplot(data=self.results['entropy']['window_entropies'],
                    bins=30, kde=True)
        
        plt.title('Distribution of Shannon Entropy Values')
        plt.xlabel('Entropy')
        plt.ylabel('Count')
        
        # Add vertical line for theoretical maximum
        max_entropy = np.log2(80)  # Maximum possible entropy for 80 numbers
        plt.axvline(x=max_entropy, color='r', linestyle='--',
                   label=f'Theoretical Max: {max_entropy:.2f}')
        
        plt.legend()
        plt.savefig(self.output_dir / 'entropy_distribution.png')
        plt.close()
    
    def plot_chi_square_results(self):
        """Plot chi-square test results over time."""
        plt.figure(figsize=(12, 6))
        
        # Create time series of p-values
        dates = pd.to_datetime(self.results['chi_square']['dates'])
        p_values = self.results['chi_square']['p_values']
        
        plt.plot(dates, p_values, marker='o')
        
        # Add significance threshold
        plt.axhline(y=0.05, color='r', linestyle='--',
                   label='Significance Threshold (0.05)')
        
        plt.title('Chi-Square Test P-Values Over Time')
        plt.xlabel('Date')
        plt.ylabel('P-Value')
        plt.xticks(rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'chi_square_trend.png')
        plt.close()
    
    def plot_temporal_drift(self):
        """Plot temporal drift analysis results."""
        plt.figure(figsize=(12, 6))
        
        # Create heatmap of drift scores
        drift_data = pd.DataFrame(self.results['temporal_drift']['drift_scores'])
        drift_data.index = pd.to_datetime(drift_data.index)
        
        sns.heatmap(drift_data, cmap='RdYlBu_r', center=0,
                   xticklabels=True, yticklabels=True)
        
        plt.title('Number Frequency Drift Over Time')
        plt.xlabel('Keno Number')
        plt.ylabel('Date')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal_drift.png')
        plt.close()
    
    def plot_autocorrelation(self):
        """Plot autocorrelation analysis results."""
        plt.figure(figsize=(10, 6))
        
        # Create heatmap of autocorrelation values
        acf_data = pd.DataFrame(self.results['autocorrelation']['acf_values'])
        
        sns.heatmap(acf_data, cmap='RdYlBu_r', center=0,
                   xticklabels=True, yticklabels=True)
        
        plt.title('Number Autocorrelation Analysis')
        plt.xlabel('Lag')
        plt.ylabel('Keno Number')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'autocorrelation.png')
        plt.close()
    
    def plot_overall_score(self):
        """Plot the overall randomness score and its components."""
        plt.figure(figsize=(10, 6))
        
        # Extract component scores
        components = {
            'Entropy': self.results['entropy']['score'],
            'Chi-Square': self.results['chi_square']['score'],
            'Temporal Drift': self.results['temporal_drift']['score'],
            'Autocorrelation': self.results['autocorrelation']['score']
        }
        
        # Create bar plot
        plt.bar(components.keys(), components.values())
        
        plt.title('Randomness Score Components')
        plt.xlabel('Component')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Add overall score as horizontal line
        overall_score = self.results['overall_score']
        plt.axhline(y=overall_score, color='r', linestyle='--',
                   label=f'Overall Score: {overall_score:.2f}')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'overall_score.png')
        plt.close()
    
    def generate_all_plots(self):
        """Generate all visualization plots."""
        logger.info("Generating visualization plots...")
        
        self.plot_entropy_distribution()
        self.plot_chi_square_results()
        self.plot_temporal_drift()
        self.plot_autocorrelation()
        self.plot_overall_score()
        
        logger.info(f"Plots saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate Keno randomness visualization plots')
    parser.add_argument('--report', required=True,
                       help='Path to the randomness audit report JSON file')
    
    args = parser.parse_args()
    
    visualizer = KenoRandomnessVisualizer(args.report)
    visualizer.generate_all_plots()

if __name__ == '__main__':
    main() 