#!/usr/bin/env bash
# Keno Prediction Tool - Example Usage Scripts
# These scripts demonstrate common usage patterns for the Keno CLI.

# Exit on error
set -e

# Create output directories
mkdir -p ./outputs
mkdir -p ./visualizations

echo "===== Keno CLI Usage Examples ====="
echo

# Example 1: Basic Data and Analysis
example1() {
    echo "Example 1: Basic Data Analysis"
    echo "------------------------------"
    
    # Scrape data from BCLC (or use sample if you prefer)
    echo "1. Scraping data..."
    ./keno-cli.py data scrape --source sample --output ./outputs/sample_data.csv
    
    # Perform basic analysis
    echo "2. Analyzing frequency patterns..."
    ./keno-cli.py analyze --type frequency --data ./outputs/sample_data.csv --output ./outputs/frequency_analysis.json
    
    # Show hot numbers
    echo "3. Top 10 hot numbers:"
    cat ./outputs/frequency_analysis.json | jq '.frequency.hot_numbers[:10]'
    
    echo "Example 1 completed."
    echo
}

# Example 2: Prediction Comparison
example2() {
    echo "Example 2: Prediction Comparison"
    echo "--------------------------------"
    
    # Generate predictions using different methods
    echo "1. Comparing prediction methods..."
    ./keno-cli.py predict --data ./outputs/sample_data.csv --compare --picks 4 --output ./outputs/predictions.json
    
    # Simulate performance
    echo "2. Simulating prediction performance..."
    ./keno-cli.py predict --data ./outputs/sample_data.csv --compare --picks 4 --simulate --draws 20 --output ./outputs/prediction_performance.json
    
    # Show best method
    echo "3. Best performing method:"
    cat ./outputs/prediction_performance.json | jq -r 'to_entries | 
        .[] | select(.key == "performance") | 
        .value | to_entries | 
        sort_by(.value.roi) | 
        reverse | 
        .[0] | 
        "\(.key): ROI=\(.value.roi)%, Profit=\(.value.net_profit)"'
    
    echo "Example 2 completed."
    echo
}

# Example 3: Visualization
example3() {
    echo "Example 3: Visualization"
    echo "------------------------"
    
    # Create frequency heatmap
    echo "1. Creating frequency heatmap..."
    ./keno-cli.py visualize --data ./outputs/sample_data.csv --type frequency --output-dir ./visualizations
    
    # Create pattern visualizations
    echo "2. Creating pattern visualizations..."
    ./keno-cli.py visualize --data ./outputs/sample_data.csv --type patterns --output-dir ./visualizations
    
    # Create overdue number visualization
    echo "3. Creating overdue number visualization..."
    ./keno-cli.py visualize --data ./outputs/sample_data.csv --type overdue --output-dir ./visualizations
    
    # Create prediction comparison
    echo "4. Creating prediction comparison..."
    ./keno-cli.py visualize --data ./outputs/sample_data.csv --type predictions --picks 4 --output-dir ./visualizations
    
    # Create dashboard
    echo "5. Creating comprehensive dashboard..."
    ./keno-cli.py visualize --data ./outputs/sample_data.csv --dashboard --dashboard-file ./visualizations/keno_dashboard.pdf
    
    echo "Visualizations created in ./visualizations directory."
    echo "Example 3 completed."
    echo
}

# Example 4: Strategy Optimization
example4() {
    echo "Example 4: Strategy Optimization"
    echo "--------------------------------"
    
    # Find optimal pick size
    echo "1. Finding optimal pick size..."
    ./keno-cli.py strategy --data ./outputs/sample_data.csv --method ensemble --simulations 10 --output ./outputs/strategy.json
    
    # Compare methods
    echo "2. Comparing strategy methods..."
    ./keno-cli.py strategy --data ./outputs/sample_data.csv --method all --picks 4 --simulations 10 --output ./outputs/strategy_comparison.json
    
    # Show best strategy
    echo "3. Best strategy:"
    cat ./outputs/strategy_comparison.json | jq '.best_strategy'
    
    echo "Example 4 completed."
    echo
}

# Example 5: Complete Workflow
example5() {
    echo "Example 5: Complete Analysis Workflow"
    echo "-------------------------------------"
    
    # Scrape data
    echo "1. Scraping latest data..."
    ./keno-cli.py data scrape --source sample --output ./outputs/latest_data.csv
    
    # Run comprehensive analysis
    echo "2. Running comprehensive analysis..."
    ./keno-cli.py analyze --data ./outputs/latest_data.csv --type all --output ./outputs/full_analysis.json
    
    # Generate predictions using ensemble method
    echo "3. Generating predictions..."
    ./keno-cli.py predict --data ./outputs/latest_data.csv --method ensemble --picks 4 --simulate --output ./outputs/latest_predictions.json
    
    # Generate visualizations
    echo "4. Creating visualizations..."
    ./keno-cli.py visualize --data ./outputs/latest_data.csv --type all --output-dir ./visualizations
    
    # Create dashboard
    echo "5. Creating dashboard..."
    ./keno-cli.py visualize --data ./outputs/latest_data.csv --dashboard --dashboard-file ./visualizations/latest_dashboard.pdf
    
    # Find optimal strategy
    echo "6. Optimizing strategy..."
    ./keno-cli.py strategy --data ./outputs/latest_data.csv --method all --simulations 20 --output ./outputs/optimal_strategy.json
    
    echo "Complete workflow completed. Results in ./outputs and ./visualizations directories."
    echo
}

# Run all examples or a specific one
if [[ -z "$1" ]]; then
    echo "Running all examples..."
    example1
    example2
    example3
    example4
    example5
else
    # Run specific example
    example$1
fi

echo "All examples completed." 