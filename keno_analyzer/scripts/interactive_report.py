"""
Interactive HTML report generation for Keno strategy comparison.
"""

import os
import json
from typing import Dict, List, Any
import numpy as np
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go

def calculate_trend(
    data: List[float], 
    window: int = 10, 
    threshold: float = 0.05
) -> Dict[str, Any]:
    """Calculate trend from time series data."""
    if len(data) < 2 * window:
        return {
            'icon': '➖',
            'color': 'gray',
            'change_percent': 0,
            'tooltip': 'Insufficient data for trend'
        }
    
    recent = np.mean(data[-window:])
    previous = np.mean(data[-2*window:-window])
    
    if previous == 0:
        change_percent = 0
    else:
        change_percent = ((recent - previous) / previous) * 100
    
    if abs(change_percent) < threshold:
        return {
            'icon': '➖',
            'color': 'gray',
            'change_percent': change_percent,
            'tooltip': f'Stable ({change_percent:.1f}% change)'
        }
    elif change_percent > 0:
        return {
            'icon': '📈',
            'color': 'green',
            'change_percent': change_percent,
            'tooltip': f'Improving (+{change_percent:.1f}%)'
        }
    else:
        return {
            'icon': '📉',
            'color': 'red',
            'change_percent': change_percent,
            'tooltip': f'Declining ({change_percent:.1f}%)'
        }

def calculate_insights(results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Calculate insights from strategy results."""
    insights = []
    
    # Calculate quality scores
    quality_scores = {}
    for strategy, data in results.items():
        # Normalize metrics to [0, 1] range
        max_hits = max(r['hits'] for r in data['detailed_results'])
        hits_norm = data['average_hits'] / max_hits
        conf_norm = data['average_confidence']
        rate_norm = data['hit_rate']
        std_dev = np.std([r['hits'] for r in data['detailed_results']])
        max_std = np.sqrt(max_hits)
        std_norm = 1 - (std_dev / max_std)
        
        # Calculate quality score (0-100)
        quality_score = (
            0.4 * hits_norm +
            0.3 * rate_norm +
            0.2 * conf_norm +
            0.1 * std_norm
        ) * 100
        
        quality_scores[strategy] = quality_score
        
    # Generate rolling quality scores for sparklines
    rolling_scores = {}
    for strategy, data in results.items():
        scores = []
        for i in range(len(data['detailed_results'])):
            window = data['detailed_results'][max(0, i-9):i+1]
            avg_hits = np.mean([r['hits'] for r in window])
            avg_conf = np.mean([r['confidence'] for r in window])
            hit_rate = sum(1 for r in window if r['hits'] > 0) / len(window)
            std_dev = np.std([r['hits'] for r in window])
            
            # Calculate normalized score
            hits_norm = avg_hits / max_hits
            std_norm = 1 - (std_dev / max_std)
            score = (
                0.4 * hits_norm +
                0.3 * hit_rate +
                0.2 * avg_conf +
                0.1 * std_norm
            ) * 100
            scores.append(score)
        rolling_scores[strategy] = scores
    
    # Top Overall Strategy
    best_strategy = max(quality_scores.items(), key=lambda x: x[1])[0]
    trend = calculate_trend(rolling_scores[best_strategy])
    insights.append({
        'title': 'Top Overall Strategy',
        'description': f'{best_strategy} (Score: {quality_scores[best_strategy]:.1f})',
        'icon': '🏆',
        'data': rolling_scores[best_strategy],
        'trend': trend
    })
    
    # Best Performing Strategy (Average Hits)
    best_hits = max(results.items(), key=lambda x: x[1]['average_hits'])[0]
    hits_data = [r['hits'] for r in results[best_hits]['detailed_results']]
    trend = calculate_trend(hits_data)
    insights.append({
        'title': 'Best Performing Strategy',
        'description': f'{best_hits} (Avg: {results[best_hits]["average_hits"]:.1f} hits)',
        'icon': '🎯',
        'data': hits_data,
        'trend': trend
    })
    
    # Most Confident Strategy
    most_confident = max(results.items(), key=lambda x: x[1]['average_confidence'])[0]
    conf_data = [r['confidence'] for r in results[most_confident]['detailed_results']]
    trend = calculate_trend(conf_data)
    insights.append({
        'title': 'Most Confident Strategy',
        'description': f'{most_confident} ({results[most_confident]["average_confidence"]:.1%})',
        'icon': '🎲',
        'data': conf_data,
        'trend': trend
    })
    
    # Most Consistent Strategy (Lowest StdDev)
    stdevs = {name: np.std([r['hits'] for r in data['detailed_results']])
              for name, data in results.items()}
    most_consistent = min(stdevs.items(), key=lambda x: x[1])[0]
    consist_data = [r['hits'] for r in results[most_consistent]['detailed_results']]
    trend = calculate_trend(consist_data)
    insights.append({
        'title': 'Most Consistent Strategy',
        'description': f'{most_consistent} (σ: {stdevs[most_consistent]:.2f})',
        'icon': '📊',
        'data': consist_data,
        'trend': trend
    })
    
    # Highest Hit Rate
    best_rate = max(results.items(), key=lambda x: x[1]['hit_rate'])[0]
    rate_data = [1 if r['hits'] > 0 else 0 
                 for r in results[best_rate]['detailed_results']]
    trend = calculate_trend(rate_data)
    insights.append({
        'title': 'Highest Hit Rate',
        'description': f'{best_rate} ({results[best_rate]["hit_rate"]:.1%})',
        'icon': '🎪',
        'data': rate_data,
        'trend': trend
    })
    
    # Best Worst-Case Performance
    min_hits = {name: min(r['hits'] for r in data['detailed_results'])
                for name, data in results.items()}
    best_min = max(min_hits.items(), key=lambda x: x[1])[0]
    worst_data = [r['hits'] for r in results[best_min]['detailed_results']]
    trend = calculate_trend(worst_data)
    insights.append({
        'title': 'Best Worst-Case Performance',
        'description': f'{best_min} (Min: {min_hits[best_min]} hits)',
        'icon': '🛡️',
        'data': worst_data,
        'trend': trend
    })
    
    return insights

def generate_interactive_report(
    results: Dict[str, Dict[str, Any]],
    plot_paths: Dict[str, str],
    csv_path: str,
    output_dir: str
) -> str:
    """
    Generate an interactive HTML report comparing different Keno strategies.
    
    Args:
        results: Dictionary containing strategy results
        plot_paths: Dictionary mapping plot names to file paths
        csv_path: Path to the summary CSV file
        output_dir: Directory to save the report
        
    Returns:
        Path to the generated HTML report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read summary data
    summary_df = pd.read_csv(csv_path)
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Interactive Keno Strategy Comparison Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .plot {{ margin: 20px 0; }}
            .summary {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Keno Strategy Comparison Report</h1>
            
            <div class="summary">
                <h2>Summary Statistics</h2>
                <table border="1">
                    <tr>
                        <th>Strategy</th>
                        <th>Total Predictions</th>
                        <th>Average Hits</th>
                        <th>Hit Rate</th>
                        <th>Average Confidence</th>
                    </tr>
    """
    
    # Add summary statistics
    for strategy, data in results.items():
        html_content += f"""
                    <tr>
                        <td>{strategy}</td>
                        <td>{data['total_predictions']}</td>
                        <td>{data['average_hits']:.2f}</td>
                        <td>{data['hit_rate']:.2%}</td>
                        <td>{data['average_confidence']:.2f}</td>
                    </tr>
        """
    
    html_content += """
                </table>
            </div>
            
            <div class="plot">
                <h2>Strategy Performance Comparison</h2>
                <img src="basic_comparison.png" alt="Basic Comparison">
            </div>
            
            <div class="plot">
                <h2>Time Series Analysis</h2>
                <img src="time_series.png" alt="Time Series">
            </div>
            
            <div class="plot">
                <h2>Hit Distribution</h2>
                <img src="hit_distribution.png" alt="Hit Distribution">
            </div>
            
            <div class="plot">
                <h2>Strategy Overlap</h2>
                <img src="strategy_overlap.png" alt="Strategy Overlap">
            </div>
            
            <div class="plot">
                <h2>Confidence vs Hits</h2>
                <img src="confidence_vs_hits.png" alt="Confidence vs Hits">
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save the report
    html_path = os.path.join(output_dir, "interactive_report.html")
    with open(html_path, "w") as f:
        f.write(html_content)
    
    return html_path 