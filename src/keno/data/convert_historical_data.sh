#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the historical data converter
python3 src/keno/data/historical_data_converter.py 