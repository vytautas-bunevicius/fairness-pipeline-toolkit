"""Defines the abstract base class for all fairness-aware classifiers.

This module provides a standardized interface for classifiers that incorporate
fairness considerations, ensuring they are compatible with scikit-learn pipelines
and other tools within the toolkit.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class FairClassifier(BaseEstimator, ClassifierMixin, ABC):
    """Defines the common interface for all fairness-aware classifier models.

    This abstract class ensures that any fairness-enhancing classifier implements
    the essential `fit`, `predict`, and `predict_proba` methods, making them
    interchangeable. It also provides common helper methods for input validation
    and state checking, promoting consistency across different fairness techniques.
    """

    def __init__(self, sensitive_features: Optional[list] = None):
        """Initializes the fair classifier.

        Args:
            sensitive_features: A list of column names that identify sensitive attributes
                (e.g., race, gender). These are used by subclasses to apply fairness logic.
        """
        self.sensitive_features = sensitive_features or []
        self.is_fitted_ = False
        self.classes_ = None

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_features: Optional[pd.DataFrame] = None,
    ) -> "FairClassifier":
        """Trains a fairness-aware model.

        Subclasses must implement this method to train their underlying model while
        incorporating fairness logic based on the sensitive features.

        Args:
            X: The training input samples.
            y: The target values.
            sensitive_features: A DataFrame containing the sensitive feature values.
                If not provided, subclasses should handle its absence gracefully,
                typically by using the features specified during initialization.

        Returns:
            The fitted classifier instance.
        """
        pass

    @abstractmethod
    def predict(
        self, X: pd.DataFrame, sensitive_features: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """Generates predictions from the trained fair model.

        Subclasses must implement this method to return predictions, ensuring that
        any fairness adjustments learned during `fit` are applied.

        Args:
            X: The input samples for prediction.
            sensitive_features: A DataFrame with sensitive feature values, which may be
                required by some models to make fair predictions.

        Returns:
            An array of predicted class labels.
        """
        pass

    @abstractmethod
    def predict_proba(
        self, X: pd.DataFrame, sensitive_features: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """Predicts class probabilities from the trained fair model.

        Subclasses must implement this method. It's essential for models where
        prediction confidence is important.

        Args:
            X: The input samples for prediction.
            sensitive_features: A DataFrame with sensitive feature values.

        Returns:
            An array of class probabilities.
        """
        pass

    def _validate_input(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Ensures that input data `X` and `y` adhere to the expected types."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if y is not None and not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("y must be a pandas Series or numpy array")

    def _check_is_fitted(self) -> None:
        """Verifies that the classifier has been trained before calling `predict` or `predict_proba`."""
        if not self.is_fitted_:
            raise ValueError("Classifier must be fitted before making predictions")

    def get_fairness_info(self) -> Dict[str, Any]:
        """Retrieves basic information about the classifier's fairness configuration."""
        return {
            "sensitive_features": self.sensitive_features,
            "is_fitted": self.is_fitted_,
            "classes": self.classes_.tolist() if self.classes_ is not None else None,
        }