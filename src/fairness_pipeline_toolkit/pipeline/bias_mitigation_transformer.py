"""
Bias mitigation approaches for fairness-aware machine learning.

This module implements preprocessing transformations that reduce statistical disparities
between demographic groups while preserving the underlying data relationships essential
for accurate predictions. The transformer addresses the challenge of biased historical
data by making feature distributions more equitable across protected groups.
"""

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.covariance import EmpiricalCovariance
from sklearn.base import BaseEstimator, TransformerMixin


class BiasMitigationTransformer(BaseEstimator, TransformerMixin):
    """
    Reduces disparate impact by equalizing feature distributions across demographic groups.

    Traditional bias mitigation often destroys feature correlations, leading to poor model
    performance. This transformer offers multiple strategies that balance fairness improvements
    with predictive utility by preserving statistical relationships in the data.
    """

    def __init__(
        self,
        sensitive_features: Optional[list] = None,
        repair_level: float = 1.0,
        random_state: int = 42,
        method: str = "mean_matching",
    ):
        """
        Initialize with configurable bias mitigation strategy.

        The repair_level controls the fairness-accuracy tradeoff: higher values increase
        fairness at the potential cost of predictive performance. Different methods
        preserve different aspects of the original data structure.
        """
        self.sensitive_features = sensitive_features or []
        self.is_fitted_ = False
        self.repair_level = repair_level
        self.random_state = random_state
        self.method = method
        self.scaler_ = StandardScaler()
        self.categorical_encoders_ = {}
        self.numerical_features_ = []
        self.categorical_features_ = []
        self.feature_means_ = {}
        self.group_stats_ = {}
        self.covariance_matrices_ = {}
        self.overall_cov_ = None
        np.random.seed(random_state)

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

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "BiasMitigationTransformer":
        """
        Learn demographic group statistics to enable fair transformations.

        Statistical disparities between groups indicate potential bias sources. By measuring
        these differences during fitting, we can later apply corrections that move group
        distributions toward equity while maintaining the mathematical relationships
        necessary for accurate prediction.
        """
        self._validate_input(X)

        self.non_sensitive_features_ = [
            col for col in X.columns if col not in self.sensitive_features
        ]

        if not self.non_sensitive_features_:
            raise ValueError("At least one non-sensitive feature is required")

        X_nonsensitive = X[self.non_sensitive_features_]
        for col in X_nonsensitive.columns:
            if X_nonsensitive[col].dtype in ["object", "category"]:
                self.categorical_features_.append(col)
            else:
                self.numerical_features_.append(col)

        X_encoded = X_nonsensitive.copy()
        for col in self.categorical_features_:
            encoder = LabelEncoder()
            X_encoded[col] = encoder.fit_transform(X_nonsensitive[col].astype(str))
            self.categorical_encoders_[col] = encoder

        if len(X_encoded.columns) > 0:
            self.scaler_.fit(X_encoded)
        for sensitive_attr in self.sensitive_features:
            group_stats = {}
            group_covs = {}

            for group_value in X[sensitive_attr].unique():
                mask = X[sensitive_attr] == group_value
                group_data = X_encoded[mask]

                if len(group_data) > 1:
                    group_stats[group_value] = {
                        "mean": group_data.mean().to_dict(),
                        "size": len(group_data),
                        "std": group_data.std().to_dict(),
                    }

                    if self.method in ["covariance_matching", "multivariate_repair"]:
                        try:
                            cov_est = EmpiricalCovariance().fit(group_data)
                            group_covs[group_value] = cov_est.covariance_
                        except (ValueError, np.linalg.LinAlgError):
                            group_covs[group_value] = np.diag(group_data.var())

            self.group_stats_[sensitive_attr] = group_stats
            self.covariance_matrices_[sensitive_attr] = group_covs

        self.feature_means_ = X_encoded.mean().to_dict()
        if self.method in ["covariance_matching", "multivariate_repair"]:
            try:
                overall_cov_est = EmpiricalCovariance().fit(X_encoded)
                self.overall_cov_ = overall_cov_est.covariance_
            except (ValueError, np.linalg.LinAlgError):
                self.overall_cov_ = np.diag(X_encoded.var())

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned corrections to reduce demographic disparities.

        The transformation adjusts feature values to make group distributions more equitable.
        By interpolating between original group means and the overall population mean,
        we reduce bias while controlling the degree of change through repair_level.
        """
        self._check_is_fitted()
        self._validate_input(X)

        X_transformed = X.copy()

        for col in self.categorical_features_:
            if col in X_transformed.columns:
                encoder = self.categorical_encoders_[col]
                try:
                    X_transformed[col] = encoder.transform(
                        X_transformed[col].astype(str)
                    )
                except ValueError:
                    known_classes = set(encoder.classes_)
                    X_transformed[col] = X_transformed[col].apply(
                        lambda x: encoder.transform([str(x)])[0]
                        if str(x) in known_classes
                        else encoder.transform([encoder.classes_[0]])[0]
                    )

        if self.method == "mean_matching":
            X_transformed = self._apply_mean_matching(X_transformed)
        elif self.method == "covariance_matching":
            X_transformed = self._apply_covariance_matching(X_transformed)
        elif self.method == "multivariate_repair":
            X_transformed = self._apply_multivariate_repair(X_transformed)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        for col in self.categorical_features_:
            if col in X_transformed.columns:
                encoder = self.categorical_encoders_[col]
                encoded_values = X_transformed[col].round().astype(int)
                encoded_values = np.clip(encoded_values, 0, len(encoder.classes_) - 1)
                X_transformed[col] = encoder.inverse_transform(encoded_values)

        return X_transformed

    def _apply_mean_matching(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Equalize group means while preserving individual variation patterns.

        This approach directly addresses statistical parity by moving group centroids
        toward the population average. It's computationally efficient but may not
        preserve complex multivariate relationships between features.
        """
        X_transformed = X.copy()

        for sensitive_attr in self.sensitive_features:
            for group_value in X[sensitive_attr].unique():
                if group_value not in self.group_stats_[sensitive_attr]:
                    continue

                mask = X[sensitive_attr] == group_value
                group_means = self.group_stats_[sensitive_attr][group_value]["mean"]

                for feature in self.non_sensitive_features_:
                    if feature in group_means and feature in self.feature_means_:
                        if feature in self.categorical_features_:
                            continue

                        group_mean = group_means[feature]
                        overall_mean = self.feature_means_[feature]

                        adjustment = self.repair_level * (overall_mean - group_mean)
                        original_dtype = X_transformed[feature].dtype
                        new_values = X_transformed.loc[mask, feature] + adjustment
                        if pd.api.types.is_integer_dtype(original_dtype):
                            X_transformed.loc[mask, feature] = (
                                new_values.round().astype(original_dtype)
                            )
                        else:
                            X_transformed.loc[mask, feature] = new_values

        return X_transformed

    def _apply_covariance_matching(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust means while attempting to preserve feature correlation structure.

        Beyond simple mean equalization, this method tries to maintain the covariance
        relationships that are often crucial for prediction accuracy. It's more complex
        but potentially preserves model performance better than mean matching alone.
        """
        X_transformed = X.copy()

        for sensitive_attr in self.sensitive_features:
            group_covs = self.covariance_matrices_[sensitive_attr]

            for group_value in X[sensitive_attr].unique():
                if (
                    group_value not in self.group_stats_[sensitive_attr]
                    or group_value not in group_covs
                ):
                    continue

                mask = X[sensitive_attr] == group_value
                group_data = X_transformed.loc[
                    mask, self.non_sensitive_features_
                ].values

                if len(group_data) == 0:
                    continue

                group_mean = np.array(
                    [
                        self.group_stats_[sensitive_attr][group_value]["mean"][f]
                        for f in self.non_sensitive_features_
                    ]
                )
                overall_mean = np.array(
                    [self.feature_means_[f] for f in self.non_sensitive_features_]
                )
                group_cov = group_covs[group_value]

                try:
                    centered_data = group_data - group_mean

                    mean_adjustment = self.repair_level * (overall_mean - group_mean)
                    adjusted_data = centered_data + group_mean + mean_adjustment

                    if len(adjusted_data) > 1:
                        current_cov = np.cov(adjusted_data.T)
                        if (
                            np.linalg.det(current_cov) > 1e-8
                            and np.linalg.det(group_cov) > 1e-8
                        ):
                            scale_factor = np.sqrt(
                                np.diag(group_cov) / (np.diag(current_cov) + 1e-8)
                            )
                            scale_factor = np.clip(scale_factor, 0.5, 2.0)
                            adjusted_data = (
                                adjusted_data - overall_mean
                            ) * scale_factor + overall_mean

                    for i, feature in enumerate(self.non_sensitive_features_):
                        original_dtype = X_transformed[feature].dtype
                        if pd.api.types.is_integer_dtype(original_dtype):
                            X_transformed.loc[mask, feature] = (
                                adjusted_data[:, i].round().astype(original_dtype)
                            )
                        else:
                            X_transformed.loc[mask, feature] = adjusted_data[:, i]

                except (np.linalg.LinAlgError, ValueError):
                    self._apply_mean_matching_single_group(
                        X_transformed, mask, group_value, sensitive_attr
                    )

        return X_transformed

    def _apply_multivariate_repair(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply multivariate bias repair that considers feature relationships."""
        X_transformed = X.copy()

        for sensitive_attr in self.sensitive_features:
            for group_value in X[sensitive_attr].unique():
                if group_value not in self.group_stats_[sensitive_attr]:
                    continue

                mask = X[sensitive_attr] == group_value
                group_data = X_transformed.loc[
                    mask, self.non_sensitive_features_
                ].values

                if len(group_data) == 0:
                    continue

                group_mean = np.array(
                    [
                        self.group_stats_[sensitive_attr][group_value]["mean"][f]
                        for f in self.non_sensitive_features_
                    ]
                )
                overall_mean = np.array(
                    [self.feature_means_[f] for f in self.non_sensitive_features_]
                )

                try:
                    centered_data = group_data - group_mean

                    target_mean = group_mean + self.repair_level * (
                        overall_mean - group_mean
                    )

                    if len(group_data) > 1:
                        group_std = np.array(
                            [
                                self.group_stats_[sensitive_attr][group_value]["std"][f]
                                for f in self.non_sensitive_features_
                            ]
                        )
                        noise_scale = (1 - self.repair_level) * 0.1
                        noise = np.random.normal(
                            0, group_std * noise_scale, centered_data.shape
                        )
                        repaired_data = centered_data + target_mean + noise
                    else:
                        repaired_data = centered_data + target_mean

                    for i, feature in enumerate(self.non_sensitive_features_):
                        original_dtype = X_transformed[feature].dtype
                        if pd.api.types.is_integer_dtype(original_dtype):
                            X_transformed.loc[mask, feature] = (
                                repaired_data[:, i].round().astype(original_dtype)
                            )
                        else:
                            X_transformed.loc[mask, feature] = repaired_data[:, i]

                except (ValueError, IndexError):
                    self._apply_mean_matching_single_group(
                        X_transformed, mask, group_value, sensitive_attr
                    )

        return X_transformed

    def _apply_mean_matching_single_group(
        self,
        X_transformed: pd.DataFrame,
        mask: pd.Series,
        group_value: Any,
        sensitive_attr: str,
    ) -> None:
        """Helper method to apply mean matching to a single group."""
        group_means = self.group_stats_[sensitive_attr][group_value]["mean"]

        for feature in self.non_sensitive_features_:
            if feature in group_means and feature in self.feature_means_:
                if feature in self.categorical_features_:
                    continue

                group_mean = group_means[feature]
                overall_mean = self.feature_means_[feature]
                adjustment = self.repair_level * (overall_mean - group_mean)
                original_dtype = X_transformed[feature].dtype
                new_values = X_transformed.loc[mask, feature] + adjustment
                if pd.api.types.is_integer_dtype(original_dtype):
                    X_transformed.loc[mask, feature] = new_values.round().astype(
                        original_dtype
                    )
                else:
                    X_transformed.loc[mask, feature] = new_values

    def get_mitigation_details(self) -> Dict[str, Any]:
        """Get information about the bias mitigation process."""
        self._check_is_fitted()

        info = {
            "method": self.method,
            "repair_level": self.repair_level,
            "sensitive_features": self.sensitive_features,
            "non_sensitive_features": self.non_sensitive_features_,
            "group_statistics": self.group_stats_,
            "overall_means": self.feature_means_,
        }

        if self.method in ["covariance_matching", "multivariate_repair"]:
            info["covariance_matrices"] = {
                attr: {
                    group: cov.tolist() if isinstance(cov, np.ndarray) else cov
                    for group, cov in group_covs.items()
                }
                for attr, group_covs in self.covariance_matrices_.items()
            }
            if self.overall_cov_ is not None:
                info["overall_covariance"] = self.overall_cov_.tolist()

        return info
