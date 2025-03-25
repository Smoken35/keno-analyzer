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
    
    # Best Overall Strategy
    best_overall = max(results.items(), 
                      key=lambda x: (x[1]['hit_rate'], x[1]['average_hits']))
    insights.append({
        'title': 'Best Overall Strategy',
        'description': f'{best_overall[0]} (Hit Rate: {best_overall[1]["hit_rate"]:.1%})',
        'icon': '🏆',
        'data': [r['hits'] for r in best_overall[1]['detailed_results']],
        'trend': calculate_trend([r['hits'] for r in best_overall[1]['detailed_results']])
    })
    
    # Most Consistent Strategy
    consistency_scores = {
        name: np.std([r['hits'] for r in data['detailed_results']])
        for name, data in results.items()
    }
    most_consistent = min(consistency_scores.items(), key=lambda x: x[1])
    insights.append({
        'title': 'Most Consistent Strategy',
        'description': f'{most_consistent[0]} (Std Dev: {most_consistent[1]:.2f})',
        'icon': '🎯',
        'data': [r['hits'] for r in results[most_consistent[0]]['detailed_results']],
        'trend': calculate_trend([r['hits'] for r in results[most_consistent[0]]['detailed_results']])
    })
    
    # Highest Hit Rate
    best_rate = max(results.items(), key=lambda x: x[1]['hit_rate'])[0]
    rate_data = [1 if r['hits'] > 0 else 0 
                 for r in results[best_rate]['detailed_results']]
    trend = calculate_trend(rate_data)
    insights.append({
        'title': 'Highest Hit Rate',
        'description': f'{best_rate} {results[best_rate]["hit_rate"]:.1%}',
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
    if not results:
        raise ValueError("Results dictionary cannot be empty")
    
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
                        <td>{data['hit_rate']:.2f}</td>
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