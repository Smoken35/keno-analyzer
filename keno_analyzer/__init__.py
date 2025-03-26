"""
Keno Strategy Analyzer package.
"""

print("Loading keno_analyzer/__init__.py")
from .core.analyzer import KenoAnalyzer

print("Successfully imported KenoAnalyzer from core.analyzer")

__version__ = "0.1.0"
__all__ = ["KenoAnalyzer"]
