"""
Tests for the interactive report generator.
"""

import os
import pytest
import numpy as np
from pathlib import Path
import sys

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from keno.scripts.interactive_report import (
    calculate_trend,
    calculate_insights,
    generate_interactive_report
)

@pytest.fixture
def sample_results():
    """Create sample results for testing."""
    return {
        "PatternStrategy": {
            "total_predictions": 100,
            "average_hits": 8.2,
            "average_confidence": 0.73,
            "hit_rate": 0.62,
            "detailed_results": [{"hits": 8, "confidence": 0.7}] * 50 + [{"hits": 9, "confidence": 0.75}] * 50
        },
        "RuleStrategy": {
            "total_predictions": 100,
            "average_hits": 7.5,
            "average_confidence": 0.65,
            "hit_rate": 0.55,
            "detailed_results": [{"hits": 7, "confidence": 0.6}] * 50 + [{"hits": 8, "confidence": 0.7}] * 50
        }
    }

@pytest.fixture
def sample_plot_paths(tmp_path):
    """Create sample plot paths for testing."""
    plot_paths = {}
    for plot_name in ["basic_comparison", "time_series", "hit_distribution", "strategy_overlap", "confidence_vs_hits"]:
        plot_path = tmp_path / f"{plot_name}.png"
        plot_path.write_text("fake image data")
        plot_paths[plot_name] = str(plot_path)
    return plot_paths

@pytest.fixture
def sample_csv_path(tmp_path, sample_results):
    """Create a sample CSV file for testing."""
    csv_path = tmp_path / "summary.csv"
    with open(csv_path, "w") as f:
        f.write("strategy,total_predictions,average_hits,hit_rate,average_confidence\n")
        for strategy, data in sample_results.items():
            f.write(f"{strategy},{data['total_predictions']},{data['average_hits']},{data['hit_rate']},{data['average_confidence']}\n")
    return str(csv_path)

class TestInteractiveReport:
    """Test cases for interactive report generation."""
    
    def test_calculate_trend(self):
        """Test trend calculation function."""
        # Test stable trend
        data = [1.0] * 20
        trend = calculate_trend(data)
        assert trend['icon'] == '➖'
        assert trend['color'] == 'gray'
        assert abs(trend['change_percent']) < 0.05
        
        # Test improving trend
        data = [1.0] * 10 + [1.2] * 10
        trend = calculate_trend(data)
        assert trend['icon'] == '📈'
        assert trend['color'] == 'green'
        assert trend['change_percent'] > 0
        
        # Test declining trend
        data = [1.2] * 10 + [1.0] * 10
        trend = calculate_trend(data)
        assert trend['icon'] == '📉'
        assert trend['color'] == 'red'
        assert trend['change_percent'] < 0
        
        # Test insufficient data
        data = [1.0] * 5
        trend = calculate_trend(data)
        assert trend['icon'] == '➖'
        assert trend['color'] == 'gray'
        assert trend['change_percent'] == 0

    def test_calculate_insights(self, sample_results):
        """Test insights calculation function."""
        insights = calculate_insights(sample_results)
        
        # Check that we get the expected number of insights
        assert len(insights) == 4  # Four key insights
        
        # Check insight structure
        for insight in insights:
            assert 'title' in insight
            assert 'description' in insight
            assert 'icon' in insight
            assert 'data' in insight
            assert 'trend' in insight
        
        # Check specific insights
        titles = [insight['title'] for insight in insights]
        assert 'Best Overall Strategy' in titles
        assert 'Most Consistent Strategy' in titles
        assert 'Highest Hit Rate' in titles
        assert 'Best Worst-Case Performance' in titles
        
        # Check that insights are properly calculated
        hit_rate_insight = next(i for i in insights if i['title'] == 'Highest Hit Rate')
        assert 'PatternStrategy' in hit_rate_insight['description']
        assert '62.0' in hit_rate_insight['description']  # Without % symbol

    def test_generate_interactive_report(self, tmp_path, sample_results, sample_plot_paths, sample_csv_path):
        """Test interactive report generation."""
        output_dir = str(tmp_path)
        
        # Generate the report
        html_path = generate_interactive_report(
            results=sample_results,
            plot_paths=sample_plot_paths,
            csv_path=sample_csv_path,
            output_dir=output_dir
        )
        
        # Check that report was generated
        assert os.path.exists(html_path)
        assert html_path.endswith('.html')
        
        # Check report content
        with open(html_path, 'r') as f:
            content = f.read()
            
            # Check for required elements
            assert 'Keno Strategy Comparison Report' in content
            assert 'PatternStrategy' in content
            assert 'RuleStrategy' in content
            assert 'Total Predictions' in content
            assert 'Average Hits' in content
            assert 'Hit Rate' in content
            assert 'Average Confidence' in content
            
            # Check for plot references
            for plot_name in sample_plot_paths.keys():
                assert f'{plot_name}.png' in content
            
            # Check for data values
            assert '8.2' in content  # PatternStrategy average hits
            assert '0.62' in content  # PatternStrategy hit rate
            assert '7.5' in content  # RuleStrategy average hits
            assert '0.55' in content  # RuleStrategy hit rate

    def test_generate_interactive_report_with_invalid_data(self, tmp_path):
        """Test report generation with invalid data."""
        with pytest.raises(ValueError, match="Results dictionary cannot be empty"):
            generate_interactive_report(
                results={},  # Empty results
                plot_paths={},  # Empty plot paths
                csv_path="nonexistent.csv",  # Invalid CSV path
                output_dir=str(tmp_path)
            )

    def test_generate_interactive_report_with_missing_plots(self, tmp_path, sample_results, sample_csv_path):
        """Test report generation with missing plot files."""
        # Create a plot paths dict with missing files
        plot_paths = {
            "basic_comparison": "nonexistent.png",
            "time_series": "nonexistent.png",
            "hit_distribution": "nonexistent.png",
            "strategy_overlap": "nonexistent.png",
            "confidence_vs_hits": "nonexistent.png"
        }
        
        # Report should still generate even with missing plots
        html_path = generate_interactive_report(
            results=sample_results,
            plot_paths=plot_paths,
            csv_path=sample_csv_path,
            output_dir=str(tmp_path)
        )
        
        assert os.path.exists(html_path)
        assert html_path.endswith('.html') 