# Keno Prediction Tool Deployment Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the Tool](#running-the-tool)
5. [Troubleshooting](#troubleshooting)
6. [Production Deployment](#production-deployment)

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 4GB RAM
- 1GB free disk space
- Internet connection (for data scraping)

### Recommended Requirements
- Python 3.10 or higher
- 8GB RAM
- 2GB free disk space
- Stable internet connection
- GPU (optional, for deep learning features)

### Operating System Support
- Linux (Ubuntu 20.04+, CentOS 8+)
- macOS (10.15+)
- Windows 10/11

## Installation

### 1. Basic Installation
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# Install from PyPI
pip install keno-prediction-tool

# For development installation
git clone https://github.com/yourusername/keno-prediction-tool.git
cd keno-prediction-tool
pip install -e .
```

### 2. Full Installation (with all features)
```bash
pip install keno-prediction-tool[full]
```

### 3. Dependencies Check
```bash
python -c "from keno.analyzer import KenoAnalyzer; print('Installation successful!')"
```

## Configuration

### 1. Basic Configuration
Create a configuration file at `~/.keno/config.yaml`:

```yaml
data:
  source: "bclc"  # or "sample" for testing
  cache_dir: "~/.keno/cache"
  
analysis:
  default_picks: 4
  simulation_draws: 1000
  
validation:
  db_path: "~/.keno/validation.db"
  min_confidence: 0.95
  
visualization:
  save_path: "~/keno_visualizations"
  style: "seaborn"
```

### 2. Payout Tables
Configure payout tables in `~/.keno/payouts.yaml`:

```yaml
BCLC:
  4:
    4: 100
    3: 5
    2: 1
    1: 0
    0: 0
  5:
    5: 500
    4: 15
    3: 2
    2: 0
    1: 0
    0: 0
  # ... add more as needed
```

### 3. Environment Variables
Set up required environment variables:

```bash
# Linux/macOS
export KENO_CONFIG_PATH=~/.keno/config.yaml
export KENO_DATA_PATH=~/.keno/data

# Windows
set KENO_CONFIG_PATH=%USERPROFILE%\.keno\config.yaml
set KENO_DATA_PATH=%USERPROFILE%\.keno\data
```

## Running the Tool

### 1. Command Line Interface
```bash
# Basic usage
keno predict --method ensemble --picks 4

# Data scraping
keno data scrape --source bclc --output daily_data.csv

# Analysis
keno analyze --type frequency --data daily_data.csv

# Visualization
keno visualize --data daily_data.csv --dashboard
```

### 2. Python API
```python
from keno import KenoAnalyzer, ValidationTracker

# Initialize
analyzer = KenoAnalyzer()
analyzer.scrape_data(source="bclc")

# Make predictions
predictions = analyzer.predict_using_ensemble(num_picks=4)

# Validate
tracker = ValidationTracker()
performance = tracker.analyze_historical_performance(
    'ensemble',
    4,
    analyzer,
    num_draws=100
)
```

### 3. Automated Scripts
Set up cron jobs or scheduled tasks:

```bash
# Linux/macOS crontab
0 */4 * * * /path/to/venv/bin/python /path/to/keno/scripts/daily_predictions.py

# Windows Task Scheduler
schtasks /create /tn "Keno Daily Predictions" /tr "python C:\path\to\daily_predictions.py" /sc daily /st 00:00
```

## Troubleshooting

### Common Issues

1. **Installation Errors**
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install build dependencies
pip install wheel setuptools

# If SSL errors occur
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org keno-prediction-tool
```

2. **Data Scraping Issues**
- Check internet connection
- Verify source URL is accessible
- Check rate limiting settings
- Use sample data for testing

3. **Database Errors**
```bash
# Reset validation database
rm ~/.keno/validation.db
keno validate init

# Check permissions
chmod 644 ~/.keno/validation.db
```

4. **Memory Issues**
- Reduce simulation size
- Clear cache: `rm -rf ~/.keno/cache/*`
- Increase Python memory limit: `export PYTHONMEM=4096`

### Logging

Configure logging in `~/.keno/logging.yaml`:

```yaml
version: 1
formatters:
  standard:
    format: '%(asctime)s - %(levelname)s - %(message)s'
handlers:
  file:
    class: logging.FileHandler
    filename: ~/.keno/keno.log
    formatter: standard
root:
  level: INFO
  handlers: [file]
```

### Error Reporting

Enable automatic error reporting:

```python
from keno.utils.error_reporting import setup_error_reporting

setup_error_reporting(
    enabled=True,
    report_path="~/.keno/error_reports"
)
```

## Production Deployment

### 1. Security Considerations

- Use secure configuration storage
- Enable SSL for data scraping
- Set up proper file permissions
- Use environment variables for sensitive data

### 2. Performance Optimization

```bash
# Cache optimization
python -m keno.utils.optimize_cache

# Database optimization
python -m keno.utils.optimize_db

# Pre-generate common predictions
python -m keno.utils.generate_prediction_cache
```

### 3. Monitoring

Set up monitoring using standard tools:

```bash
# Prometheus metrics
keno metrics --port 8000

# Health checks
curl http://localhost:8000/health

# Performance metrics
curl http://localhost:8000/metrics
```

### 4. Backup and Recovery

```bash
# Automated backup script
#!/bin/bash
backup_dir="/path/to/backups"
date_str=$(date +%Y%m%d)

# Backup configuration
cp -r ~/.keno "${backup_dir}/config_${date_str}"

# Backup database
sqlite3 ~/.keno/validation.db ".backup '${backup_dir}/validation_${date_str}.db'"

# Compress
tar -czf "${backup_dir}/keno_backup_${date_str}.tar.gz" \
    "${backup_dir}/config_${date_str}" \
    "${backup_dir}/validation_${date_str}.db"
```

### 5. Scaling

For high-volume usage:

- Use Redis for caching
- Implement load balancing
- Set up database replication
- Use distributed task queues

Example Redis configuration:
```python
from keno.utils.cache import setup_cache

setup_cache(
    backend="redis",
    host="localhost",
    port=6379,
    db=0
)
``` 