#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add the project root to PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR/../..:$PYTHONPATH"

# Run the pattern test REPL
python3 "$SCRIPT_DIR/pattern_test_repl.py" 