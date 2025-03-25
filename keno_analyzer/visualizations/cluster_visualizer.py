#!/usr/bin/env python3
"""
Cluster Visualizer - Creates visualizations for Keno number clusters.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cluster_visualization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ClusterVisualizer:
    """Visualizes Keno number clusters using various plots."""
    
    def __init__(self, results_file: str):
        """
        Initialize the visualizer.
        
        Args:
            results_file: Path to cluster analysis results JSON file
        """
        self.results_file = results_file
        self.results = self._load_results()
        self.clusters = self.results['clusters']
        self.metadata = self.results['metadata']
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def _load_results(self) -> Dict:
        """Load cluster analysis results from JSON file."""
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def plot_cluster_heatmap(self, output_file: str = 'results/cluster_heatmap.png'):
        """
        Create a heatmap showing cluster quality metrics.
        
        Args:
            output_file: Path to save the plot
        """
        # Prepare data
        n_clusters = len(self.clusters)
        metrics = ['quality_score', 'cohesion', 'coverage', 'frequency']
        data = np.zeros((n_clusters, len(metrics)))
        
        for i, cluster in enumerate(self.clusters):
            for j, metric in enumerate(metrics):
                data[i, j] = cluster[metric]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(data, 
                   xticklabels=metrics,
                   yticklabels=[f'Cluster {i+1}' for i in range(n_clusters)],
                   cmap='YlOrRd',
                   annot=True,
                   fmt='.3f',
                   cbar_kws={'label': 'Score'})
        
        plt.title('Cluster Quality Metrics Heatmap')
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved cluster heatmap to {output_file}")
    
    def plot_cluster_sizes(self, output_file: str = 'results/cluster_sizes.png'):
        """
        Create a bar plot showing cluster sizes.
        
        Args:
            output_file: Path to save the plot
        """
        # Prepare data
        sizes = [len(cluster['numbers']) for cluster in self.clusters]
        cluster_labels = [f'Cluster {i+1}' for i in range(len(self.clusters))]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Create bar plot
        bars = plt.bar(cluster_labels, sizes)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.title('Number of Elements in Each Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Elements')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file)
        plt.close()
        
        logger.info(f"Saved cluster sizes plot to {output_file}")
    
    def plot_cluster_network(self, output_file: str = 'results/cluster_network.png'):
        """
        Create a network visualization of clusters.
        
        Args:
            output_file: Path to save the plot
        """
        # Create graph
        G = nx.Graph()
        
        # Add nodes for each number
        for cluster in self.clusters:
            for number in cluster['numbers']:
                G.add_node(number, cluster=cluster['numbers'])
        
        # Add edges between numbers in the same cluster
        for cluster in self.clusters:
            numbers = cluster['numbers']
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    G.add_edge(numbers[i], numbers[j], 
                              weight=cluster['cohesion'])
        
        # Create figure
        plt.figure(figsize=(15, 15))
        
        # Set up layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                             node_color='lightblue',
                             node_size=500)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, 
                             alpha=0.5,
                             width=1)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, 
                              font_size=8,
                              font_weight='bold')
        
        plt.title('Keno Number Cluster Network')
        plt.axis('off')
        
        # Save plot
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved cluster network plot to {output_file}")
    
    def plot_metric_distributions(self, output_file: str = 'results/metric_distributions.png'):
        """
        Create violin plots showing distributions of cluster metrics.
        
        Args:
            output_file: Path to save the plot
        """
        # Prepare data
        metrics = ['quality_score', 'cohesion', 'coverage', 'frequency']
        data = {metric: [cluster[metric] for cluster in self.clusters] 
                for metric in metrics}
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Create violin plots
        sns.violinplot(data=data)
        
        plt.title('Distribution of Cluster Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file)
        plt.close()
        
        logger.info(f"Saved metric distributions plot to {output_file}")

def main():
    """Main function to run cluster visualization."""
    parser = argparse.ArgumentParser(description='Visualize Keno number clusters')
    parser.add_argument('--results', required=True,
                       help='Path to cluster analysis results JSON file')
    parser.add_argument('--output-dir', default='results',
                       help='Directory to save visualization files')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = ClusterVisualizer(args.results)
    
    # Create visualizations
    output_dir = Path(args.output_dir)
    
    visualizer.plot_cluster_heatmap(output_dir / 'cluster_heatmap.png')
    visualizer.plot_cluster_sizes(output_dir / 'cluster_sizes.png')
    visualizer.plot_cluster_network(output_dir / 'cluster_network.png')
    visualizer.plot_metric_distributions(output_dir / 'metric_distributions.png')
    
    logger.info("Visualization complete")

if __name__ == '__main__':
    main() 