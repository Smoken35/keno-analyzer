#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the super-ensemble predictor
python3 src/keno/prediction/super_ensemble.py 