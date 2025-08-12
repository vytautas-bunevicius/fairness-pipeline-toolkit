"""Fairness-constrained classifier using constraint optimization."""

from typing import Any, Optional, Dict
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
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
            self.use_fairlearn = False
            self.base_estimator = base_estimator if base_estimator is not None else LogisticRegression(random_state=random_state)
        else:
            self.use_fairlearn = True
            self.base_estimator = base_estimator if base_estimator is not None else LogisticRegression(random_state=random_state)
            
            if constraint == "demographic_parity":
                self.constraint = DemographicParity()
            elif constraint == "equalized_odds":
                self.constraint = EqualizedOdds()
            else:
                raise ValueError(f"Unknown constraint: {constraint}")
            
            self.mitigator = ExponentiatedGradient(
                estimator=self.base_estimator,
                constraints=self.constraint,
                max_iter=100,
                nu=1e-6,
                eta0=2.0,
                eps=0.01
            )
        
        self.constraint_name = constraint
        self.random_state = random_state
        self.categorical_encoders_ = {}
        self.categorical_features_ = []
        
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
        
        if sensitive_features is None:
            if not self.sensitive_features:
                raise ValueError("sensitive_features must be provided")
            sensitive_features = X[self.sensitive_features]
        
        # Identify and encode categorical features
        X_encoded = X.copy()
        for col in X.columns:
            if X[col].dtype in ['object', 'category']:
                self.categorical_features_.append(col)
                encoder = LabelEncoder()
                X_encoded[col] = encoder.fit_transform(X[col].astype(str))
                self.categorical_encoders_[col] = encoder
        
        # Encode categorical features in sensitive_features too
        sensitive_features_encoded = sensitive_features.copy()
        for col in sensitive_features.columns:
            if sensitive_features[col].dtype in ['object', 'category']:
                if col not in self.categorical_encoders_:
                    encoder = LabelEncoder()
                    sensitive_features_encoded[col] = encoder.fit_transform(sensitive_features[col].astype(str))
                    self.categorical_encoders_[col] = encoder
                else:
                    sensitive_features_encoded[col] = self.categorical_encoders_[col].transform(sensitive_features[col].astype(str))
        
        self.classes_ = np.unique(y)
        self.original_columns_ = list(X_encoded.columns)  # Store original column order
        
        if self.use_fairlearn:
            self.mitigator.fit(X_encoded, y, sensitive_features=sensitive_features_encoded)
        else:
            self.base_estimator.fit(X_encoded, y)
            self._fit_threshold_optimization(X_encoded, y, sensitive_features_encoded)
        
        self.is_fitted_ = True
        return self
    
    def _fit_threshold_optimization(self, X: pd.DataFrame, y: pd.Series, 
                           sensitive_features: pd.DataFrame) -> None:
        """Optimize decision thresholds for fairness (fallback method)."""
        base_predictions = self.base_estimator.predict_proba(X)[:, 1]
        
        self.group_thresholds_ = {}
        
        for sensitive_col in sensitive_features.columns:
            thresholds = {}
            for group in sensitive_features[sensitive_col].unique():
                mask = sensitive_features[sensitive_col] == group
                group_probs = base_predictions[mask]
                
                best_threshold = 0.5
                if len(group_probs) > 0:
                    sorted_probs = np.sort(group_probs)
                    pos_rate_target = y.mean()
                    
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
        
        missing_cols = set(self.original_columns_) - set(X.columns)
        if missing_cols:
            X_combined = X.copy()
            for col in missing_cols:
                if col in sensitive_features.columns:
                    X_combined[col] = sensitive_features[col]
            X = X_combined[self.original_columns_]
        
        X_encoded = X.copy()
        for col in self.categorical_features_:
            if col in X_encoded.columns:
                encoder = self.categorical_encoders_[col]
                try:
                    X_encoded[col] = encoder.transform(X_encoded[col].astype(str))
                except ValueError:
                    # Handle unseen categories gracefully
                    known_classes = set(encoder.classes_)
                    X_encoded[col] = X_encoded[col].apply(
                        lambda x: encoder.transform([str(x)])[0] if str(x) in known_classes 
                        else encoder.transform([encoder.classes_[0]])[0]
                    )
        
        # Encode sensitive features
        sensitive_features_encoded = sensitive_features.copy()
        for col in sensitive_features.columns:
            if col in self.categorical_encoders_:
                encoder = self.categorical_encoders_[col]
                try:
                    sensitive_features_encoded[col] = encoder.transform(sensitive_features[col].astype(str))
                except ValueError:
                    known_classes = set(encoder.classes_)
                    sensitive_features_encoded[col] = sensitive_features[col].apply(
                        lambda x: encoder.transform([str(x)])[0] if str(x) in known_classes 
                        else encoder.transform([encoder.classes_[0]])[0]
                    )
        
        if self.use_fairlearn:
            return self.mitigator.predict(X_encoded)
        else:
            base_probs = self.base_estimator.predict_proba(X_encoded)[:, 1]
            predictions = np.zeros(len(X_encoded))
            
            for sensitive_col in sensitive_features_encoded.columns:
                for group in sensitive_features_encoded[sensitive_col].unique():
                    if group in self.group_thresholds_[sensitive_col]:
                        mask = sensitive_features_encoded[sensitive_col] == group
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
        
        # Encode features for prediction
        X_encoded = X.copy()
        for col in self.categorical_features_:
            if col in X_encoded.columns:
                encoder = self.categorical_encoders_[col]
                try:
                    X_encoded[col] = encoder.transform(X_encoded[col].astype(str))
                except ValueError:
                    known_classes = set(encoder.classes_)
                    X_encoded[col] = X_encoded[col].apply(
                        lambda x: encoder.transform([str(x)])[0] if str(x) in known_classes 
                        else encoder.transform([encoder.classes_[0]])[0]
                    )
        
        if self.use_fairlearn:
            predictions = self.predict(X, sensitive_features)
            n_samples = len(predictions)
            n_classes = len(self.classes_)
            probas = np.zeros((n_samples, n_classes))
            
            for i, pred in enumerate(predictions):
                probas[i, pred] = 0.8
                probas[i, 1-pred] = 0.2
            
            return probas
        else:
            return self.base_estimator.predict_proba(X_encoded)
    
    def get_fairness_info(self) -> Dict[str, Any]:
        """Get information about fairness constraints and methods."""
        info = super().get_fairness_info()
        info.update({
            'constraint': self.constraint_name,
            'base_estimator': str(self.base_estimator),
            'uses_fairlearn': self.use_fairlearn
        })
        return info