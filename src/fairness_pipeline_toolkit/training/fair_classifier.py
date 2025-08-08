"""Base fair classifier interface."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class FairClassifier(BaseEstimator, ClassifierMixin, ABC):
    """Abstract base class for fairness-aware classifiers."""
    
    def __init__(self, sensitive_features: Optional[list] = None):
        """Initialize fair classifier.
        
        Args:
            sensitive_features: List of sensitive feature column names
        """
        self.sensitive_features = sensitive_features or []
        self.is_fitted_ = False
        self.classes_ = None
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            sensitive_features: Optional[pd.DataFrame] = None) -> 'FairClassifier':
        """Fit the fair classifier to training data.
        
        Args:
            X: Feature matrix
            y: Target labels
            sensitive_features: Sensitive attribute values
            
        Returns:
            Self for method chaining
        """
        pass
        
    @abstractmethod
    def predict(self, X: pd.DataFrame, 
                sensitive_features: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Make predictions using the fair classifier.
        
        Args:
            X: Feature matrix
            sensitive_features: Sensitive attribute values
            
        Returns:
            Predicted labels
        """
        pass
        
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame, 
                     sensitive_features: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix
            sensitive_features: Sensitive attribute values
            
        Returns:
            Class probabilities
        """
        pass
        
    def _validate_input(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Validate input data format."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
            
        if y is not None and not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("y must be a pandas Series or numpy array")
            
    def _check_is_fitted(self) -> None:
        """Check if classifier has been fitted."""
        if not self.is_fitted_:
            raise ValueError("Classifier must be fitted before making predictions")
            
    def get_fairness_info(self) -> Dict[str, Any]:
        """Get information about fairness constraints and methods."""
        return {
            'sensitive_features': self.sensitive_features,
            'is_fitted': self.is_fitted_,
            'classes': self.classes_.tolist() if self.classes_ is not None else None
        }