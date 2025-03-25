# Keno Prediction Tool Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### 1. Package Installation Fails
```
ERROR: Could not find a version that satisfies the requirement keno-prediction-tool
```

**Solution:**
1. Verify Python version:
```bash
python --version  # Should be 3.8 or higher
```

2. Update pip and setuptools:
```bash
python -m pip install --upgrade pip setuptools wheel
```

3. Try installing with specific version:
```bash
pip install keno-prediction-tool==0.4.0
```

#### 2. Dependency Conflicts
```
ERROR: Cannot install keno-prediction-tool due to package conflicts
```

**Solution:**
1. Create a fresh virtual environment:
```bash
python -m venv .venv-keno
source .venv-keno/bin/activate
```

2. Install with isolated environment:
```bash
pip install --no-cache-dir keno-prediction-tool
```

### Data Scraping Issues

#### 1. Connection Errors
```python
requests.exceptions.ConnectionError: Failed to establish connection
```

**Solutions:**
1. Check internet connection
2. Verify proxy settings:
```python
analyzer = KenoAnalyzer()
analyzer.scrape_data(
    source="bclc",
    proxy={
        'http': 'http://proxy.example.com:8080',
        'https': 'https://proxy.example.com:8080'
    }
)
```

3. Use rate limiting:
```python
analyzer.scrape_data(source="bclc", rate_limit=2)  # 2 seconds between requests
```

#### 2. Data Format Issues
```python
ValueError: Invalid data format in scraped content
```

**Solutions:**
1. Use sample data for testing:
```python
analyzer.scrape_data(source="sample")
```

2. Validate source format:
```python
analyzer.validate_source_format("bclc")
```

### Analysis Issues

#### 1. Memory Errors
```python
MemoryError: Unable to allocate array with shape (1000000, 80)
```

**Solutions:**
1. Reduce data size:
```python
analyzer.set_max_historical_draws(1000)
```

2. Use chunked processing:
```python
analyzer.enable_chunked_processing(chunk_size=100)
```

#### 2. Performance Issues
```python
WARNING: Analysis taking longer than expected
```

**Solutions:**
1. Enable caching:
```python
analyzer.enable_caching(cache_dir="~/.keno/cache")
```

2. Optimize calculations:
```python
analyzer.set_optimization_level("high")
```

### Prediction Issues

#### 1. Invalid Predictions
```python
ValueError: Predictions contain invalid numbers
```

**Solutions:**
1. Validate input data:
```python
analyzer.validate_data()
```

2. Check prediction parameters:
```python
analyzer.validate_prediction_params(
    method="ensemble",
    num_picks=4
)
```

#### 2. Inconsistent Results
```python
WARNING: Prediction results show high variance
```

**Solutions:**
1. Increase sample size:
```python
predictions = analyzer.predict_using_ensemble(
    num_picks=4,
    num_simulations=1000
)
```

2. Use validation:
```python
tracker = ValidationTracker()
tracker.validate_predictions(predictions, confidence_level=0.95)
```

### Validation Issues

#### 1. Database Errors
```python
sqlite3.OperationalError: database is locked
```

**Solutions:**
1. Reset database connection:
```python
tracker.reset_connection()
```

2. Use different database file:
```python
tracker = ValidationTracker(db_path="~/keno_new.db")
```

#### 2. Data Integrity Issues
```python
IntegrityError: UNIQUE constraint failed
```

**Solutions:**
1. Clean validation data:
```python
tracker.clean_validation_data()
```

2. Reset validation history:
```python
tracker.reset_validation_history()
```

### Visualization Issues

#### 1. Display Problems
```python
matplotlib.pyplot.error: Cannot connect to display
```

**Solutions:**
1. Use non-interactive backend:
```python
import matplotlib
matplotlib.use('Agg')
```

2. Save to file instead:
```python
visualizer.save_plot("output.png")
```

#### 2. Memory Issues with Large Plots
```python
MemoryError: Failed to allocate memory for plot
```

**Solutions:**
1. Reduce plot resolution:
```python
visualizer.set_dpi(72)
```

2. Use chunked plotting:
```python
visualizer.plot_in_chunks(chunk_size=100)
```

## Advanced Troubleshooting

### Performance Profiling

1. Enable profiling:
```python
from keno.utils.profiling import profile_execution

with profile_execution() as profiler:
    analyzer.predict_using_ensemble(4)
    
profiler.print_stats()
```

2. Memory profiling:
```python
from keno.utils.profiling import memory_profile

memory_profile(analyzer.analyze_game)
```

### Logging and Debugging

1. Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. Log to file:
```python
handler = logging.FileHandler('keno_debug.log')
logging.getLogger().addHandler(handler)
```

### Error Reporting

1. Enable automatic error reporting:
```python
from keno.utils.error_reporting import setup_error_reporting

setup_error_reporting(
    enabled=True,
    report_path="~/.keno/error_reports"
)
```

2. Generate error report:
```python
from keno.utils.error_reporting import generate_report

generate_report()
```

## System Checks

### Environment Check
```python
from keno.utils.system import check_environment

check_environment()
```

### Dependency Check
```python
from keno.utils.system import check_dependencies

check_dependencies()
```

### Performance Check
```python
from keno.utils.system import benchmark_system

benchmark_system()
```

## Getting Help

### Community Support
- GitHub Issues: [Link to issues]
- Documentation: [Link to docs]
- Stack Overflow: Tag [keno-prediction-tool]

### Reporting Bugs
1. Enable verbose logging
2. Generate system report
3. Create minimal reproduction case
4. Submit issue with details

### Contributing Fixes
1. Fork repository
2. Create branch
3. Add tests
4. Submit pull request 