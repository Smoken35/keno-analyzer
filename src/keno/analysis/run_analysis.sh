#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the combination analyzer
python3 src/keno/analysis/combination_analyzer.py 