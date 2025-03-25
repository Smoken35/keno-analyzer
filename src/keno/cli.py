#!/usr/bin/env python3
"""
Command Line Interface for the Keno Prediction Tool.
Provides comprehensive access to analysis, prediction, and visualization features.
"""

import argparse
import json
import yaml
import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union

from .analyzer import KenoAnalyzer
from .visualization.visualizer import KenoVisualizer

def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_data(analyzer: KenoAnalyzer, args: argparse.Namespace) -> bool:
    """
    Load data from specified source into analyzer.
    
    Args:
        analyzer: KenoAnalyzer instance
        args: Command line arguments
        
    Returns:
        bool: True if data loaded successfully
    """
    try:
        if args.csv:
            logging.info(f"Loading data from CSV: {args.csv}")
            analyzer.load_data_from_csv(args.csv)
        else:
            logging.info(f"Scraping data from {args.source}")
            analyzer.scrape_data(source=args.source)
            
            if args.output:
                analyzer.export_data(args.output)
                logging.info(f"Data exported to {args.output}")
        
        return True
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return False

def set_payout_tables(analyzer: KenoAnalyzer) -> None:
    """Set up standard payout tables."""
    # BCLC payouts
    bclc_payouts = {
        2: {2: 11, 1: 0, 0: 0},
        3: {3: 27, 2: 2, 1: 0, 0: 0},
        4: {4: 72, 3: 4, 2: 1, 1: 0, 0: 0},
        5: {5: 410, 4: 18, 3: 2, 2: 1, 1: 0, 0: 0},
        6: {6: 1100, 5: 50, 4: 8, 3: 1, 2: 0, 1: 0, 0: 0},
        7: {7: 4500, 6: 100, 5: 20, 4: 3, 3: 1, 2: 0, 1: 0, 0: 0},
        8: {8: 10000, 7: 500, 6: 75, 5: 10, 4: 2, 3: 0, 2: 0, 1: 0, 0: 0},
        9: {9: 40000, 8: 4250, 7: 200, 6: 50, 5: 6, 4: 1, 3: 0, 2: 0, 1: 0, 0: 0},
        10: {10: 100000, 9: 5000, 8: 500, 7: 80, 6: 15, 5: 2, 4: 0, 3: 0, 2: 0, 1: 0, 0: 0}
    }
    analyzer.set_payout_table("BCLC", bclc_payouts)

def handle_data_command(args: argparse.Namespace) -> Dict:
    """Handle data-related commands."""
    analyzer = KenoAnalyzer()
    
    if not load_data(analyzer, args):
        return {'error': 'Failed to load data'}
    
    result = {
        'data_source': args.source if not args.csv else args.csv,
        'num_draws': len(analyzer.data),
        'date_range': {
            'start': analyzer.data['date'].min().strftime('%Y-%m-%d'),
            'end': analyzer.data['date'].max().strftime('%Y-%m-%d')
        }
    }
    
    if args.output:
        result['output_file'] = args.output
    
    return result

def handle_analyze_command(args: argparse.Namespace) -> Dict:
    """Handle analysis commands."""
    analyzer = KenoAnalyzer()
    
    if not load_data(analyzer, args):
        return {'error': 'Failed to load data'}
    
    set_payout_tables(analyzer)
    results = {}
    
    if args.type in ['all', 'frequency']:
        freq_df = analyzer.analyze_frequency()
        hot_numbers = analyzer.find_hot_numbers(20)
        cold_numbers = analyzer.find_cold_numbers(10)
        results['frequency'] = {
            'hot_numbers': hot_numbers['number'].tolist(),
            'cold_numbers': cold_numbers['number'].tolist()
        }
    
    if args.type in ['all', 'patterns']:
        skip_analysis = analyzer.analyze_skip_and_hit_patterns()
        if skip_analysis is not None:
            results['patterns'] = {
                'overdue_numbers': skip_analysis.head(10)['number'].tolist(),
                'overdue_factors': skip_analysis.head(10)['overdue_factor'].tolist()
            }
    
    if args.type in ['all', 'cycles']:
        cycles = analyzer.detect_cyclic_patterns()
        if cycles and cycles['cyclic_numbers']:
            results['cycles'] = {
                'cyclic_numbers': [
                    {
                        'number': num,
                        'avg_interval': data['avg_interval'],
                        'confidence': data['confidence']
                    }
                    for num, data in cycles['cyclic_numbers'].items()
                ]
            }
    
    if args.type in ['all', 'strategy']:
        ev_comparison = analyzer.compare_play_types("BCLC")
        if ev_comparison:
            results['strategy'] = {
                'best_play': max(
                    ev_comparison.items(),
                    key=lambda x: x[1]['return_percentage']
                )[0]
            }
    
    return results

def handle_predict_command(args: argparse.Namespace) -> Dict:
    """Handle prediction commands."""
    analyzer = KenoAnalyzer()
    
    if not load_data(analyzer, args):
        return {'error': 'Failed to load data'}
    
    set_payout_tables(analyzer)
    results = {'predictions': {}}
    
    methods = {
        'frequency': analyzer.predict_next_draw,
        'pattern': lambda: analyzer.predict_next_draw(method='pattern'),
        'cycles': analyzer.predict_using_cycles,
        'markov': analyzer.predict_using_markov_chains,
        'due': analyzer.predict_using_due_theory,
        'ensemble': analyzer.predict_using_ensemble
    }
    
    if args.method == 'all' or args.compare:
        for method_name, predict_func in methods.items():
            try:
                predictions = predict_func(args.picks)[:args.picks]
                results['predictions'][method_name] = predictions
            except Exception as e:
                logging.warning(f"Method {method_name} failed: {e}")
    else:
        if args.method in methods:
            predictions = methods[args.method](args.picks)[:args.picks]
            results['predictions'][args.method] = predictions
    
    if args.simulate:
        results['performance'] = {}
        for method_name, numbers in results['predictions'].items():
            try:
                sim = analyzer.simulate_strategy(
                    "BCLC", len(numbers), "custom",
                    num_draws=args.draws,
                    bet_amount=1,
                    custom_numbers=numbers
                )
                if sim:
                    results['performance'][method_name] = {
                        'roi': sim['roi'],
                        'net_profit': sim['net_profit'],
                        'match_distribution': sim['match_distribution']
                    }
            except Exception as e:
                logging.warning(f"Simulation failed for {method_name}: {e}")
    
    return results

def handle_visualize_command(args: argparse.Namespace) -> Dict:
    """Handle visualization commands."""
    analyzer = KenoAnalyzer()
    
    if not load_data(analyzer, args):
        return {'error': 'Failed to load data'}
    
    set_payout_tables(analyzer)
    visualizer = KenoVisualizer(analyzer)
    
    # Set output directory
    if args.output_dir:
        visualizer.save_path = args.output_dir
        os.makedirs(visualizer.save_path, exist_ok=True)
    
    results = {'visualizations': []}
    
    if args.type in ['all', 'frequency']:
        visualizer.plot_frequency_heatmap("frequency_heatmap.png")
        results['visualizations'].append('frequency_heatmap.png')
    
    if args.type in ['all', 'patterns']:
        visualizer.plot_patterns("pattern_analysis.png")
        results['visualizations'].append('pattern_analysis.png')
    
    if args.type in ['all', 'overdue']:
        visualizer.plot_overdue_numbers("overdue_numbers.png")
        results['visualizations'].append('overdue_numbers.png')
    
    if args.type in ['all', 'cycles']:
        visualizer.plot_cyclic_patterns("cyclic_patterns.png")
        results['visualizations'].append('cyclic_patterns.png')
    
    if args.type in ['all', 'pairs']:
        visualizer.plot_pair_analysis("pair_analysis.png")
        results['visualizations'].append('pair_analysis.png')
    
    if args.type in ['all', 'history']:
        hot_numbers = analyzer.find_hot_numbers(5)['number'].tolist()
        visualizer.plot_number_history(hot_numbers, "hot_numbers_history.png")
        results['visualizations'].append('hot_numbers_history.png')
    
    if args.type in ['all', 'predictions']:
        methods = {
            'Frequency': analyzer.predict_next_draw(method='frequency')[:args.picks],
            'Pattern': analyzer.predict_next_draw(method='pattern')[:args.picks],
            'Cycle': analyzer.predict_using_cycles(args.picks),
            'Markov': analyzer.predict_using_markov_chains(args.picks),
            'Due': analyzer.predict_using_due_theory(args.picks),
            'Ensemble': analyzer.predict_using_ensemble(args.picks)
        }
        visualizer.plot_prediction_comparison(methods, "prediction_comparison.png")
        results['visualizations'].append('prediction_comparison.png')
    
    if args.dashboard:
        dashboard_path = visualizer.create_dashboard(args.dashboard_file)
        results['dashboard'] = dashboard_path
    
    return results

def handle_strategy_command(args: argparse.Namespace) -> Dict:
    """Handle strategy optimization commands."""
    analyzer = KenoAnalyzer()
    
    if not load_data(analyzer, args):
        return {'error': 'Failed to load data'}
    
    set_payout_tables(analyzer)
    results = {}
    
    # Compare strategies
    comparison = analyzer.compare_advanced_strategies(
        "BCLC",
        bet_amount=1,
        num_draws=args.simulations
    )
    
    if comparison is not None and not comparison.empty:
        # Convert DataFrame to dict for JSON serialization
        results['strategies'] = comparison.to_dict('records')
        
        # Find best strategy
        best_strategy = comparison.iloc[0]
        results['best_strategy'] = {
            'method': best_strategy['Method'],
            'pick_size': best_strategy['Pick Size'],
            'roi': best_strategy['ROI %'],
            'net_profit': best_strategy['Net Profit'],
            'selected_numbers': best_strategy['Selected Numbers']
        }
    
    return results

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Keno Prediction Tool CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Data command
    data_parser = subparsers.add_parser('data', help='Data management commands')
    data_parser.add_argument(
        '--source',
        default='sample',
        help="Data source (sample, bclc, or URL)"
    )
    data_parser.add_argument(
        '--csv',
        help="Load data from CSV file"
    )
    data_parser.add_argument(
        '--output',
        help="Output file for scraped data"
    )
    
    # Analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Analysis commands')
    analyze_parser.add_argument(
        '--type',
        choices=['all', 'frequency', 'patterns', 'cycles', 'strategy'],
        default='all',
        help="Type of analysis to perform"
    )
    analyze_parser.add_argument(
        '--source',
        default='sample',
        help="Data source"
    )
    analyze_parser.add_argument(
        '--csv',
        help="Load data from CSV file"
    )
    analyze_parser.add_argument(
        '--output',
        help="Output file for analysis results"
    )
    
    # Prediction command
    predict_parser = subparsers.add_parser('predict', help='Prediction commands')
    predict_parser.add_argument(
        '--method',
        choices=['all', 'frequency', 'pattern', 'cycles', 'markov', 'due', 'ensemble'],
        default='ensemble',
        help="Prediction method to use"
    )
    predict_parser.add_argument(
        '--picks',
        type=int,
        default=4,
        help="Number of numbers to pick"
    )
    predict_parser.add_argument(
        '--compare',
        action='store_true',
        help="Compare all prediction methods"
    )
    predict_parser.add_argument(
        '--simulate',
        action='store_true',
        help="Simulate prediction performance"
    )
    predict_parser.add_argument(
        '--draws',
        type=int,
        default=100,
        help="Number of draws to simulate"
    )
    predict_parser.add_argument(
        '--source',
        default='sample',
        help="Data source"
    )
    predict_parser.add_argument(
        '--csv',
        help="Load data from CSV file"
    )
    predict_parser.add_argument(
        '--output',
        help="Output file for predictions"
    )
    
    # Visualization command
    viz_parser = subparsers.add_parser('visualize', help='Visualization commands')
    viz_parser.add_argument(
        '--type',
        choices=['all', 'frequency', 'patterns', 'overdue', 'cycles', 'pairs', 'history', 'predictions'],
        default='all',
        help="Type of visualization to create"
    )
    viz_parser.add_argument(
        '--picks',
        type=int,
        default=4,
        help="Number of picks for prediction visualization"
    )
    viz_parser.add_argument(
        '--source',
        default='sample',
        help="Data source"
    )
    viz_parser.add_argument(
        '--csv',
        help="Load data from CSV file"
    )
    viz_parser.add_argument(
        '--output-dir',
        default='keno_visualizations',
        help="Output directory for visualizations"
    )
    viz_parser.add_argument(
        '--dashboard',
        action='store_true',
        help="Create comprehensive dashboard"
    )
    viz_parser.add_argument(
        '--dashboard-file',
        default='keno_dashboard.pdf',
        help="Filename for dashboard PDF"
    )
    
    # Strategy command
    strategy_parser = subparsers.add_parser('strategy', help='Strategy optimization commands')
    strategy_parser.add_argument(
        '--method',
        choices=['all', 'frequency', 'pattern', 'cycles', 'markov', 'due', 'ensemble'],
        default='all',
        help="Strategy method to optimize"
    )
    strategy_parser.add_argument(
        '--picks',
        type=int,
        default=4,
        help="Number of picks to optimize for"
    )
    strategy_parser.add_argument(
        '--simulations',
        type=int,
        default=100,
        help="Number of simulations to run"
    )
    strategy_parser.add_argument(
        '--source',
        default='sample',
        help="Data source"
    )
    strategy_parser.add_argument(
        '--csv',
        help="Load data from CSV file"
    )
    strategy_parser.add_argument(
        '--output',
        help="Output file for strategy results"
    )
    
    return parser.parse_args()

def main() -> int:
    """Main CLI function."""
    args = parse_args()
    setup_logging(args.verbose)
    
    # Execute command
    try:
        if args.command == 'data':
            results = handle_data_command(args)
        elif args.command == 'analyze':
            results = handle_analyze_command(args)
        elif args.command == 'predict':
            results = handle_predict_command(args)
        elif args.command == 'visualize':
            results = handle_visualize_command(args)
        elif args.command == 'strategy':
            results = handle_strategy_command(args)
        else:
            print("Please specify a command. Use --help for usage information.")
            return 1
        
        # Output results
        if args.output and args.command != 'visualize':
            # Write to file
            with open(args.output, 'w') as f:
                if args.output.endswith('.yaml'):
                    yaml.dump(results, f, default_flow_style=False)
                else:
                    json.dump(results, f, indent=2)
            logging.info(f"Results written to {args.output}")
        else:
            # Print to console
            print(json.dumps(results, indent=2))
        
        return 0
        
    except Exception as e:
        logging.error(f"Error executing command: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 