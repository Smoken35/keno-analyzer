"""
Base class for Keno prediction strategies.
Defines the interface that all prediction strategies must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class BasePredictionStrategy(ABC):
    """Abstract base class for Keno prediction strategies."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """Initialize the prediction strategy.
        
        Args:
            name: Name of the strategy
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self._validate_config()
        
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the strategy configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def fit(self, historical_data: List[List[int]]) -> None:
        """Fit the strategy to historical data.
        
        Args:
            historical_data: List of historical Keno draws
        """
        pass
    
    @abstractmethod
    def predict(self, draw_index: int, num_picks: int = 20) -> List[int]:
        """Generate predictions for a specific draw.
        
        Args:
            draw_index: Index of the draw to predict
            num_picks: Number of numbers to predict (default: 20)
            
        Returns:
            List of predicted numbers
        """
        pass
    
    @abstractmethod
    def get_confidence(self, prediction: List[int]) -> float:
        """Get confidence score for a prediction.
        
        Args:
            prediction: List of predicted numbers
            
        Returns:
            Confidence score between 0 and 1
        """
        pass
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the strategy.
        
        Returns:
            Dictionary containing strategy information
        """
        return {
            'name': self.name,
            'config': self.config,
            'type': self.__class__.__name__
        }
    
    def _validate_prediction(self, prediction: List[int], num_picks: int) -> None:
        """Validate a prediction.
        
        Args:
            prediction: List of predicted numbers
            num_picks: Expected number of picks
            
        Raises:
            ValueError: If prediction is invalid
        """
        if not isinstance(prediction, list):
            raise ValueError("Prediction must be a list")
        
        if len(prediction) != num_picks:
            raise ValueError(f"Prediction must contain exactly {num_picks} numbers")
        
        if not all(isinstance(n, int) for n in prediction):
            raise ValueError("All predictions must be integers")
        
        if not all(1 <= n <= 80 for n in prediction):
            raise ValueError("All predictions must be between 1 and 80")
        
        if len(set(prediction)) != num_picks:
            raise ValueError("Predictions must be unique") 