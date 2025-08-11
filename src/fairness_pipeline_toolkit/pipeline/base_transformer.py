"""Base transformer interface for data debiasing."""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BaseTransformer(BaseEstimator, TransformerMixin, ABC):
    """Abstract base class for fairness-aware data transformers."""
    
    def __init__(self, sensitive_features: Optional[list] = None):
        """Initialize transformer with sensitive feature specifications."""
        self.sensitive_features = sensitive_features or []
        self.is_fitted_ = False
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BaseTransformer':
        """Fit the transformer to the data.
        
        Args:
            X: Input features DataFrame
            y: Target variable (optional)
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input data to reduce bias.
        
        Args:
            X: Input features DataFrame
            
        Returns:
            Transformed DataFrame with reduced bias
        """
        pass
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit transformer and transform data in one step."""
        return self.fit(X, y).transform(X)
    
    def _validate_input(self, X: pd.DataFrame) -> None:
        """Validate input data format and required columns."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        missing_features = set(self.sensitive_features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing sensitive features: {missing_features}")
    
    def _check_is_fitted(self) -> None:
        """Check if transformer has been fitted."""
        if not self.is_fitted_:
            raise ValueError("Transformer must be fitted before transforming data")