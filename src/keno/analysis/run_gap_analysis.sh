#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the gap analyzer
python3 src/keno/analysis/gap_analyzer.py 