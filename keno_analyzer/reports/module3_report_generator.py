#!/usr/bin/env python3
"""
Module 3 Report Generator
Generates stakeholder-ready HTML and PDF reports for Keno pattern analysis results.
"""

import argparse
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from datetime import datetime
from pathlib import Path
from base64 import b64encode
from io import BytesIO
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('module3_report.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Module3ReportGenerator:
    def __init__(self, results_file: str, output_dir: str, logo_path: str = None):
        """Initialize the report generator.
        
        Args:
            results_file: Path to the JSON results file
            output_dir: Directory to save the generated reports
            logo_path: Optional path to company logo
        """
        self.results_file = Path(results_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logo_path = Path(logo_path) if logo_path else None
        
        # Load results
        with open(self.results_file, 'r') as f:
            self.results = json.load(f)
        
        # Setup Jinja2 environment
        template_dir = Path(__file__).parent / 'templates'
        self.env = Environment(loader=FileSystemLoader(str(template_dir)))
        
        # Configure matplotlib style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # Brand colors
        self.colors = {
            'primary': '#1a237e',
            'secondary': '#0d47a1',
            'accent': '#ff4081',
            'success': '#4caf50',
            'warning': '#ff9800',
            'danger': '#f44336'
        }
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary metrics and insights."""
        summary = {
            'total_patterns': len(self.results['patterns']),
            'total_rules': len(self.results['rules']),
            'total_clusters': len(self.results['clusters']),
            'key_insights': []
        }
        
        # Pattern insights
        pattern_supports = [p['support'] for p in self.results['patterns']]
        summary['key_insights'].append({
            'category': 'Patterns',
            'insights': [
                f"Found {summary['total_patterns']} significant patterns",
                f"Average support: {np.mean(pattern_supports):.3f}",
                f"Strongest pattern support: {max(pattern_supports):.3f}"
            ]
        })
        
        # Rule insights
        rule_lifts = [r['lift'] for r in self.results['rules']]
        summary['key_insights'].append({
            'category': 'Association Rules',
            'insights': [
                f"Generated {summary['total_rules']} association rules",
                f"Average lift: {np.mean(rule_lifts):.3f}",
                f"Strongest rule lift: {max(rule_lifts):.3f}"
            ]
        })
        
        # Cluster insights
        cluster_qualities = [c['quality_score'] for c in self.results['clusters']]
        summary['key_insights'].append({
            'category': 'Number Clusters',
            'insights': [
                f"Identified {summary['total_clusters']} number clusters",
                f"Average cluster quality: {np.mean(cluster_qualities):.3f}",
                f"Best cluster quality: {max(cluster_qualities):.3f}"
            ]
        })
        
        return summary
    
    def _generate_summary_metrics(self) -> Dict[str, Any]:
        """Generate detailed summary metrics from the results."""
        metrics = {
            'patterns': {
                'total': len(self.results['patterns']),
                'avg_support': np.mean([p['support'] for p in self.results['patterns']]),
                'max_support': max([p['support'] for p in self.results['patterns']]),
                'min_support': min([p['support'] for p in self.results['patterns']]),
                'avg_size': np.mean([len(p['items']) for p in self.results['patterns']])
            },
            'rules': {
                'total': len(self.results['rules']),
                'avg_lift': np.mean([r['lift'] for r in self.results['rules']]),
                'max_lift': max([r['lift'] for r in self.results['rules']]),
                'avg_confidence': np.mean([r['confidence'] for r in self.results['rules']]),
                'avg_size': np.mean([len(r['antecedent']) + len(r['consequent']) 
                                   for r in self.results['rules']])
            },
            'clusters': {
                'total': len(self.results['clusters']),
                'avg_quality': np.mean([c['quality_score'] for c in self.results['clusters']]),
                'max_quality': max([c['quality_score'] for c in self.results['clusters']]),
                'avg_size': np.mean([len(c['numbers']) for c in self.results['clusters']]),
                'avg_cohesion': np.mean([c['cohesion'] for c in self.results['clusters']])
            }
        }
        return metrics
    
    def _generate_pattern_plots(self) -> Dict[str, str]:
        """Generate enhanced plots for pattern analysis."""
        plots = {}
        
        # Support distribution with KDE
        plt.figure(figsize=(10, 6))
        supports = [p['support'] for p in self.results['patterns']]
        sns.histplot(supports, bins=30, kde=True, color=self.colors['primary'])
        plt.title('Pattern Support Distribution', pad=20)
        plt.xlabel('Support')
        plt.ylabel('Count')
        plots['support_dist'] = self._plot_to_base64()
        
        # Pattern size distribution
        plt.figure(figsize=(10, 6))
        sizes = [len(p['items']) for p in self.results['patterns']]
        sns.histplot(sizes, bins=30, color=self.colors['secondary'])
        plt.title('Pattern Size Distribution', pad=20)
        plt.xlabel('Number of Items')
        plt.ylabel('Count')
        plots['size_dist'] = self._plot_to_base64()
        
        # Pattern heatmap (top 20 patterns)
        plt.figure(figsize=(12, 8))
        top_patterns = sorted(self.results['patterns'], 
                            key=lambda x: x['support'], 
                            reverse=True)[:20]
        pattern_matrix = np.zeros((20, 80))
        for i, pattern in enumerate(top_patterns):
            for num in pattern['items']:
                pattern_matrix[i, num-1] = pattern['support']
        
        sns.heatmap(pattern_matrix, cmap='YlOrRd', xticklabels=10)
        plt.title('Top 20 Patterns Heatmap', pad=20)
        plt.xlabel('Keno Numbers')
        plt.ylabel('Pattern Rank')
        plots['pattern_heatmap'] = self._plot_to_base64()
        
        return plots
    
    def _generate_rule_plots(self) -> Dict[str, str]:
        """Generate enhanced plots for association rules analysis."""
        plots = {}
        
        # Lift vs Confidence scatter plot with size
        plt.figure(figsize=(10, 6))
        lifts = [r['lift'] for r in self.results['rules']]
        confidences = [r['confidence'] for r in self.results['rules']]
        sizes = [len(r['antecedent']) + len(r['consequent']) for r in self.results['rules']]
        
        plt.scatter(confidences, lifts, s=sizes, alpha=0.5, c=self.colors['primary'])
        plt.title('Rule Lift vs Confidence', pad=20)
        plt.xlabel('Confidence')
        plt.ylabel('Lift')
        plots['lift_conf'] = self._plot_to_base64()
        
        # Rule network graph (top 20 rules)
        plt.figure(figsize=(12, 8))
        G = nx.DiGraph()
        top_rules = sorted(self.results['rules'], 
                          key=lambda x: x['lift'], 
                          reverse=True)[:20]
        
        for rule in top_rules:
            ant = tuple(sorted(rule['antecedent']))
            cons = tuple(sorted(rule['consequent']))
            G.add_edge(ant, cons, weight=rule['lift'])
        
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color=self.colors['primary'],
                node_size=1000, font_size=8, font_weight='bold')
        plt.title('Top 20 Rules Network', pad=20)
        plots['rule_network'] = self._plot_to_base64()
        
        return plots
    
    def _generate_cluster_plots(self) -> Dict[str, str]:
        """Generate enhanced plots for cluster analysis."""
        plots = {}
        
        # Cluster quality distribution with KDE
        plt.figure(figsize=(10, 6))
        qualities = [c['quality_score'] for c in self.results['clusters']]
        sns.histplot(qualities, bins=30, kde=True, color=self.colors['primary'])
        plt.title('Cluster Quality Distribution', pad=20)
        plt.xlabel('Quality Score')
        plt.ylabel('Count')
        plots['quality_dist'] = self._plot_to_base64()
        
        # Cluster size vs quality with color gradient
        plt.figure(figsize=(10, 6))
        sizes = [len(c['numbers']) for c in self.results['clusters']]
        qualities = [c['quality_score'] for c in self.results['clusters']]
        cohesions = [c['cohesion'] for c in self.results['clusters']]
        
        scatter = plt.scatter(sizes, qualities, c=cohesions, 
                            cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Cohesion')
        plt.title('Cluster Size vs Quality', pad=20)
        plt.xlabel('Number of Items')
        plt.ylabel('Quality Score')
        plots['size_quality'] = self._plot_to_base64()
        
        # Cluster heatmap (top 20 clusters)
        plt.figure(figsize=(12, 8))
        top_clusters = sorted(self.results['clusters'], 
                            key=lambda x: x['quality_score'], 
                            reverse=True)[:20]
        cluster_matrix = np.zeros((20, 80))
        for i, cluster in enumerate(top_clusters):
            for num in cluster['numbers']:
                cluster_matrix[i, num-1] = cluster['quality_score']
        
        sns.heatmap(cluster_matrix, cmap='YlOrRd', xticklabels=10)
        plt.title('Top 20 Clusters Heatmap', pad=20)
        plt.xlabel('Keno Numbers')
        plt.ylabel('Cluster Rank')
        plots['cluster_heatmap'] = self._plot_to_base64()
        
        return plots
    
    def _plot_to_base64(self) -> str:
        """Convert the current matplotlib figure to base64 string."""
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.close()
        return b64encode(buf.getvalue()).decode('utf-8')
    
    def _get_top_items(self, n: int = 5) -> Dict[str, List[Dict]]:
        """Get top items for each category with enhanced metrics."""
        top_items = {
            'patterns': sorted(
                self.results['patterns'],
                key=lambda x: x['support'],
                reverse=True
            )[:n],
            'rules': sorted(
                self.results['rules'],
                key=lambda x: x['lift'],
                reverse=True
            )[:n],
            'clusters': sorted(
                self.results['clusters'],
                key=lambda x: x['quality_score'],
                reverse=True
            )[:n]
        }
        return top_items
    
    def generate_html_report(self) -> Path:
        """Generate the HTML report with enhanced visualizations."""
        logger.info("Generating HTML report...")
        
        # Prepare template data
        template_data = {
            'executive_summary': self._generate_executive_summary(),
            'metrics': self._generate_summary_metrics(),
            'pattern_plots': self._generate_pattern_plots(),
            'rule_plots': self._generate_rule_plots(),
            'cluster_plots': self._generate_cluster_plots(),
            'top_items': self._get_top_items(),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'colors': self.colors,
            'has_logo': self.logo_path is not None
        }
        
        # Render template
        template = self.env.get_template('module3_report.html')
        html_content = template.render(**template_data)
        
        # Save HTML report
        output_file = self.output_dir / 'module3_report.html'
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {output_file}")
        return output_file
    
    def generate_pdf_report(self) -> Path:
        """Generate the PDF report from the HTML report."""
        logger.info("Generating PDF report...")
        
        # Generate HTML report first
        html_file = self.generate_html_report()
        
        # Convert HTML to PDF
        output_file = self.output_dir / 'module3_report.pdf'
        HTML(html_file).write_pdf(output_file)
        
        logger.info(f"PDF report generated: {output_file}")
        return output_file

def main():
    parser = argparse.ArgumentParser(description='Generate Module 3 analysis reports')
    parser.add_argument('results_file', help='Path to the JSON results file')
    parser.add_argument('--output-dir', default='reports', help='Directory to save reports')
    parser.add_argument('--format', choices=['html', 'pdf', 'both'], default='both',
                      help='Report format to generate')
    parser.add_argument('--logo', help='Path to company logo')
    
    args = parser.parse_args()
    
    try:
        generator = Module3ReportGenerator(args.results_file, args.output_dir, args.logo)
        
        if args.format in ['html', 'both']:
            generator.generate_html_report()
        
        if args.format in ['pdf', 'both']:
            generator.generate_pdf_report()
        
        logger.info("Report generation completed successfully")
        
    except Exception as e:
        logger.error(f"Error generating reports: {str(e)}")
        raise

if __name__ == '__main__':
    main() 