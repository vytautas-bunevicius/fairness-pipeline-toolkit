"""Fairness-constrained classifier using constraint optimization."""

from typing import Any, Optional, Dict
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    
from .fair_classifier import FairClassifier


class FairnessConstrainedClassifier(FairClassifier):
    """Classifier that enforces fairness constraints during training."""
    
    def __init__(self, 
                 base_estimator: Optional[Any] = None,
                 constraint: str = "demographic_parity",
                 sensitive_features: Optional[list] = None,
                 random_state: int = 42):
        """Initialize fairness-constrained classifier.
        
        Args:
            base_estimator: Base ML model to use
            constraint: Fairness constraint ("demographic_parity" or "equalized_odds")
            sensitive_features: List of sensitive feature column names
            random_state: Random state for reproducibility
        """
        super().__init__(sensitive_features)
        
        if not FAIRLEARN_AVAILABLE:
            # Fallback implementation when fairlearn is not available
            self.use_fairlearn = False
            self.base_estimator = base_estimator or LogisticRegression(random_state=random_state)
        else:
            self.use_fairlearn = True
            self.base_estimator = base_estimator or LogisticRegression(random_state=random_state)
            
            # Set up constraint
            if constraint == "demographic_parity":
                self.constraint = DemographicParity()
            elif constraint == "equalized_odds":
                self.constraint = EqualizedOdds()
            else:
                raise ValueError(f"Unknown constraint: {constraint}")
            
            self.mitigator = ExponentiatedGradient(
                estimator=self.base_estimator,
                constraints=self.constraint,
                max_iter=50,
                nu=1e-6,
                eta0=2.0
            )
        
        self.constraint_name = constraint
        self.random_state = random_state
        
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            sensitive_features: Optional[pd.DataFrame] = None) -> 'FairnessConstrainedClassifier':
        """Fit the fair classifier to training data.
        
        Args:
            X: Feature matrix
            y: Target labels  
            sensitive_features: Sensitive attribute values
            
        Returns:
            Self for method chaining
        """
        self._validate_input(X, y)
        
        # Extract sensitive features from X if not provided separately
        if sensitive_features is None:
            if not self.sensitive_features:
                raise ValueError("sensitive_features must be provided")
            sensitive_features = X[self.sensitive_features]
        
        self.classes_ = np.unique(y)
        
        if self.use_fairlearn:
            # Use Fairlearn implementation
            self.mitigator.fit(X, y, sensitive_features=sensitive_features)
        else:
            # Fallback: simple fair classifier using post-processing
            self.base_estimator.fit(X, y)
            self._fit_threshold_optimization(X, y, sensitive_features)
        
        self.is_fitted_ = True
        return self
    
    def _fit_threshold_optimization(self, X: pd.DataFrame, y: pd.Series, 
                           sensitive_features: pd.DataFrame) -> None:
        """Optimize decision thresholds for fairness (fallback method)."""
        # Simple post-processing approach
        base_predictions = self.base_estimator.predict_proba(X)[:, 1]
        
        # Calculate group-specific thresholds to achieve fairness
        self.group_thresholds_ = {}
        
        for sensitive_col in sensitive_features.columns:
            thresholds = {}
            for group in sensitive_features[sensitive_col].unique():
                mask = sensitive_features[sensitive_col] == group
                group_probs = base_predictions[mask]
                group_labels = y[mask]
                
                # Find threshold that balances fairness and accuracy
                best_threshold = 0.5
                if len(group_probs) > 0:
                    sorted_probs = np.sort(group_probs)
                    pos_rate_target = y.mean()  # Overall positive rate
                    
                    # Find threshold closest to achieving target positive rate
                    for threshold in sorted_probs:
                        pred_pos_rate = (group_probs >= threshold).mean()
                        if abs(pred_pos_rate - pos_rate_target) < abs((group_probs >= best_threshold).mean() - pos_rate_target):
                            best_threshold = threshold
                
                thresholds[group] = best_threshold
            
            self.group_thresholds_[sensitive_col] = thresholds
    
    def predict(self, X: pd.DataFrame, 
                sensitive_features: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Make predictions using the fair classifier.
        
        Args:
            X: Feature matrix
            sensitive_features: Sensitive attribute values
            
        Returns:
            Predicted labels
        """
        self._check_is_fitted()
        self._validate_input(X)
        
        if sensitive_features is None:
            sensitive_features = X[self.sensitive_features]
        
        if self.use_fairlearn:
            # Fairlearn ExponentiatedGradient predict method doesn't accept sensitive_features
            return self.mitigator.predict(X)
        else:
            # Fallback prediction with post-processing
            base_probs = self.base_estimator.predict_proba(X)[:, 1]
            predictions = np.zeros(len(X))
            
            # Apply group-specific thresholds
            for sensitive_col in sensitive_features.columns:
                for group in sensitive_features[sensitive_col].unique():
                    if group in self.group_thresholds_[sensitive_col]:
                        mask = sensitive_features[sensitive_col] == group
                        threshold = self.group_thresholds_[sensitive_col][group]
                        predictions[mask] = (base_probs[mask] >= threshold).astype(int)
            
            return predictions
    
    def predict_proba(self, X: pd.DataFrame, 
                     sensitive_features: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix
            sensitive_features: Sensitive attribute values
            
        Returns:
            Class probabilities
        """
        self._check_is_fitted()
        self._validate_input(X)
        
        if self.use_fairlearn:
            # Fairlearn doesn't always support predict_proba, so we approximate
            predictions = self.predict(X, sensitive_features)
            n_samples = len(predictions)
            n_classes = len(self.classes_)
            probas = np.zeros((n_samples, n_classes))
            
            for i, pred in enumerate(predictions):
                probas[i, pred] = 0.8  # High confidence for predicted class
                probas[i, 1-pred] = 0.2  # Low confidence for other class
            
            return probas
        else:
            return self.base_estimator.predict_proba(X)
    
    def get_fairness_info(self) -> Dict[str, Any]:
        """Get information about fairness constraints and methods."""
        info = super().get_fairness_info()
        info.update({
            'constraint': self.constraint_name,
            'base_estimator': str(self.base_estimator),
            'uses_fairlearn': self.use_fairlearn
        })
        return info