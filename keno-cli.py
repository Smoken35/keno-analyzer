#!/usr/bin/env python3
"""
Keno CLI executable wrapper.
"""

import sys

from keno.cli import main

if __name__ == "__main__":
    sys.exit(main())
