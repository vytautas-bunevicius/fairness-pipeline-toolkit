"""Bias mitigation transformer for reducing disparate impact in datasets."""

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .base_transformer import BaseTransformer


class BiasMitigationTransformer(BaseTransformer):
    """Mitigate bias by adjusting feature distributions across demographic groups."""
    
    def __init__(self, sensitive_features: Optional[list] = None, 
                 repair_level: float = 1.0, random_state: int = 42):
        """Initialize bias mitigation transformer.
        
        Args:
            sensitive_features: List of sensitive feature column names
            repair_level: Level of repair to apply (0.0 to 1.0)
            random_state: Random state for reproducibility
        """
        super().__init__(sensitive_features)
        self.repair_level = repair_level
        self.random_state = random_state
        self.scaler_ = StandardScaler()
        self.feature_means_ = {}
        self.group_stats_ = {}
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BiasMitigationTransformer':
        """Fit the transformer by computing group statistics.
        
        Args:
            X: Input features DataFrame
            y: Target variable (optional)
            
        Returns:
            Self for method chaining
        """
        self._validate_input(X)
        
        # Identify non-sensitive features
        self.non_sensitive_features_ = [col for col in X.columns 
                                      if col not in self.sensitive_features]
        
        if not self.non_sensitive_features_:
            raise ValueError("At least one non-sensitive feature is required")
        
        # Fit scaler on non-sensitive features
        X_nonsensitive = X[self.non_sensitive_features_]
        self.scaler_.fit(X_nonsensitive)
        
        # Compute group statistics for each sensitive attribute
        for sensitive_attr in self.sensitive_features:
            group_stats = {}
            for group_value in X[sensitive_attr].unique():
                mask = X[sensitive_attr] == group_value
                group_data = X_nonsensitive[mask]
                
                if len(group_data) > 0:
                    group_stats[group_value] = {
                        'mean': group_data.mean().to_dict(),
                        'size': len(group_data)
                    }
            
            self.group_stats_[sensitive_attr] = group_stats
        
        # Compute overall means
        self.feature_means_ = X_nonsensitive.mean().to_dict()
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data to reduce disparate impact.
        
        Args:
            X: Input features DataFrame
            
        Returns:
            Transformed DataFrame with reduced bias
        """
        self._check_is_fitted()
        self._validate_input(X)
        
        X_transformed = X.copy()
        
        # Apply bias mitigation to non-sensitive features
        for sensitive_attr in self.sensitive_features:
            for group_value in X[sensitive_attr].unique():
                if group_value not in self.group_stats_[sensitive_attr]:
                    continue
                    
                mask = X[sensitive_attr] == group_value
                group_means = self.group_stats_[sensitive_attr][group_value]['mean']
                
                # Apply repair by moving group means towards overall means
                for feature in self.non_sensitive_features_:
                    if feature in group_means and feature in self.feature_means_:
                        group_mean = group_means[feature]
                        overall_mean = self.feature_means_[feature]
                        
                        # Calculate mitigation adjustment
                        adjustment = self.repair_level * (overall_mean - group_mean)
                        X_transformed.loc[mask, feature] += adjustment
        
        return X_transformed
    
    def get_mitigation_details(self) -> Dict[str, Any]:
        """Get information about the bias mitigation process."""
        self._check_is_fitted()
        
        info = {
            'repair_level': self.repair_level,
            'sensitive_features': self.sensitive_features,
            'non_sensitive_features': self.non_sensitive_features_,
            'group_statistics': self.group_stats_,
            'overall_means': self.feature_means_
        }
        
        return info