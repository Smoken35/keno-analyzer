"""
Interactive HTML report generation for Keno strategy comparison.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def calculate_trend(numbers: List[float], window: int = 5) -> List[float]:
    """Calculate moving average trend."""
    return [
        sum(numbers[max(0, i - window) : i + 1]) / min(i + 1, window) for i in range(len(numbers))
    ]


def analyze_patterns(data: List[int], window: int = 10) -> Dict[str, float]:
    """Analyze number patterns."""
    patterns = {}
    for i in range(len(data) - window + 1):
        pattern = tuple(data[i : i + window])
        patterns[pattern] = patterns.get(pattern, 0) + 1
    return {str(k): v / len(data) for k, v in patterns.items()}


def calculate_insights(
    results: Dict[str, Dict[str, Union[float, List[Dict[str, float]]]]]
) -> List[Dict[str, Union[str, List[float], Dict[str, Union[str, float]]]]]:
    """Calculate insights from strategy results."""
    insights = []

    # Best Overall Strategy
    best_overall = max(results.items(), key=lambda x: (x[1]["hit_rate"], x[1]["average_hits"]))
    insights.append(
        {
            "title": "Best Overall Strategy",
            "description": f'{best_overall[0]} (Hit Rate: {best_overall[1]["hit_rate"]:.1f})',
            "icon": "🏆",
            "data": [r["hits"] for r in best_overall[1]["detailed_results"]],
            "trend": calculate_trend(
                [float(r["hits"]) for r in best_overall[1]["detailed_results"]]
            ),
        }
    )

    # Most Consistent Strategy
    consistency_scores = {
        name: np.std([r["hits"] for r in data["detailed_results"]])
        for name, data in results.items()
    }
    most_consistent = min(consistency_scores.items(), key=lambda x: x[1])
    insights.append(
        {
            "title": "Most Consistent Strategy",
            "description": f"{most_consistent[0]} (Std Dev: {most_consistent[1]:.2f})",
            "icon": "🎯",
            "data": [r["hits"] for r in results[most_consistent[0]]["detailed_results"]],
            "trend": calculate_trend(
                [float(r["hits"]) for r in results[most_consistent[0]]["detailed_results"]]
            ),
        }
    )

    # Highest Hit Rate
    best_rate = max(results.items(), key=lambda x: x[1]["hit_rate"])
    insights.append(
        {
            "title": "Highest Hit Rate",
            "description": f'{best_rate[0]} ({best_rate[1]["hit_rate"]:.1f})',
            "icon": "🎪",
            "data": [1 if r["hits"] > 0 else 0 for r in best_rate[1]["detailed_results"]],
            "trend": calculate_trend(
                [float(1 if r["hits"] > 0 else 0) for r in best_rate[1]["detailed_results"]]
            ),
        }
    )

    # Best Worst-Case Performance
    min_hits = {
        name: min(r["hits"] for r in data["detailed_results"]) for name, data in results.items()
    }
    best_min = max(min_hits.items(), key=lambda x: x[1])
    insights.append(
        {
            "title": "Best Worst-Case Performance",
            "description": f"{best_min[0]} (Min: {best_min[1]} hits)",
            "icon": "🛡️",
            "data": [r["hits"] for r in results[best_min[0]]["detailed_results"]],
            "trend": calculate_trend(
                [float(r["hits"]) for r in results[best_min[0]]["detailed_results"]]
            ),
        }
    )

    return insights


def generate_interactive_report(
    results: Dict[str, Dict[str, Union[float, List[Dict[str, float]]]]],
    plot_paths: Dict[str, str],
    csv_path: str,
    output_dir: str,
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

    insights = calculate_insights(results)
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Keno Strategy Comparison Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .plot {
                margin: 20px 0;
            }
            .summary {
                margin: 20px 0;
            }
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


def generate_report(data: List[int], output_file: str = "report.html") -> None:
    """Generate interactive report."""
    if not data:
        raise ValueError("Data cannot be empty")

    if any(n < 1 or n > 80 for n in data):
        raise ValueError("All numbers must be between 1 and 80")

    df = pd.DataFrame({"numbers": data})

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Number Distribution",
            "Moving Average Trend",
            "Pattern Analysis",
            "Frequency Analysis",
        ),
    )

    # Number distribution
    hist = go.Histogram(x=data, nbinsx=20, name="Distribution")
    fig.add_trace(hist, row=1, col=1)

    # Moving average trend
    trend = calculate_trend([float(x) for x in data])
    line = go.Scatter(x=list(range(len(trend))), y=trend, name="Trend")
    fig.add_trace(line, row=1, col=2)

    # Pattern analysis
    patterns = analyze_patterns(data)
    bar = go.Bar(x=list(patterns.keys())[:10], y=list(patterns.values())[:10], name="Patterns")
    fig.add_trace(bar, row=2, col=1)

    # Frequency analysis
    freq = pd.Series(data).value_counts().sort_index()
    scatter = go.Scatter(x=freq.index, y=freq.values, mode="markers", name="Frequency")
    fig.add_trace(scatter, row=2, col=2)

    # Update layout
    fig.update_layout(height=800, showlegend=True, title_text="Keno Number Analysis Report")

    # Save report
    fig.write_html(output_file)


if __name__ == "__main__":
    # Example usage
    import random

    data = [random.randint(1, 80) for _ in range(100)]
    generate_report(data)
