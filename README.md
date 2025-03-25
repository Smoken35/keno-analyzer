# Keno Strategy Analyzer

[![Test](https://github.com/Smoken35/keno-analyzer/actions/workflows/test.yml/badge.svg)](https://github.com/Smoken35/keno-analyzer/actions/workflows/test.yml)
[![Coverage](https://img.shields.io/badge/coverage-0%25-red.svg)](https://smoken35.github.io/keno-analyzer/coverage_html/index.html)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/Smoken35/keno-analyzer)](LICENSE)
[![PyPI version](https://badge.fury.io/py/keno-analyzer.svg)](https://badge.fury.io/py/keno-analyzer)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python package for analyzing and optimizing Keno strategies using advanced statistical methods and machine learning techniques.

## Features

- Strategy comparison and evaluation
- Interactive HTML reports with trend analysis
- Performance metrics and visualizations
- Real-time strategy optimization

## Installation

### Development Installation

```bash
git clone https://github.com/Smoken35/keno-analyzer.git
cd keno-analyzer
pip install -e ".[dev]"
```

### User Installation

```bash
pip install keno-analyzer
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

## Development

### Code Style

This project uses:
- [Black](https://github.com/psf/black) for code formatting
- [isort](https://github.com/pycqa/isort) for import sorting
- [flake8](https://flake8.pycqa.org/) for linting

To format code:
```bash
black .
isort .
```

### Pre-commit Hooks

Install pre-commit hooks:
```bash
pre-commit install
```

This will run formatting and linting checks before each commit.

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src/keno --cov-report=html

# Run specific test file
pytest tests/test_analyzer.py
```

### Building Documentation

```bash
# Build HTML documentation
sphinx-build -b html docs/source docs/build/html
```

## Coverage

![Coverage](https://img.shields.io/badge/coverage-0%25-red.svg)

Detailed coverage reports are available at [Coverage Report](https://smoken35.github.io/keno-analyzer/coverage_html/index.html). 