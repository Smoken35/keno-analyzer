# Keno Strategy Analyzer

A Python package for analyzing and optimizing Keno strategies using advanced statistical methods and machine learning techniques.

## Features

- Strategy comparison and evaluation
- Interactive HTML reports with trend analysis
- Performance metrics and visualizations
- Real-time strategy optimization

## Installation

### Development Installation

```bash
git clone https://github.com/yourusername/keno_analyzer.git
cd keno_analyzer
pip install -e ".[dev]"
```

### User Installation

```bash
pip install keno_analyzer
```

## Usage

```python
from keno_analyzer import KenoAnalyzer

# Initialize analyzer
analyzer = KenoAnalyzer(data_source="sample")

# Analyze patterns
patterns = analyzer.analyze_patterns()

# Make predictions
prediction = analyzer.predict_next_draw(method="patterns", picks=4)

# Generate interactive report
from keno_analyzer.scripts.interactive_report import generate_interactive_report
report_path = generate_interactive_report(results, plot_paths, csv_path, output_dir)
```

## Testing

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request 