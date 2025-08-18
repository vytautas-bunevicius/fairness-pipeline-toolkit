"""A classifier that enforces fairness during training using Fairlearn's reductions.

It uses Exponentiated Gradient to apply fairness constraints like Demographic Parity.
If Fairlearn is not installed, it falls back to a simpler threshold optimization
as a post-processing step to mitigate bias.
"""

from typing import Any, Optional, Dict
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

try:
    from fairlearn.reductions import (
        ExponentiatedGradient,
        DemographicParity,
        EqualizedOdds,
    )

    FAIRLEARN_AVAILABLE = True

except ImportError:
    # Define lightweight stubs to satisfy type checkers when fairlearn isn't available
    class ExponentiatedGradient:  # type: ignore
        pass

    class DemographicParity:  # type: ignore
        pass

    class EqualizedOdds:  # type: ignore
        pass

    FAIRLEARN_AVAILABLE = False

from .fair_classifier import FairClassifier


class FairnessConstrainedClassifier(FairClassifier):
    """Enforces fairness constraints on a classifier during training.

    This wrapper uses the Exponentiated Gradient algorithm from Fairlearn to apply
    fairness constraints. If Fairlearn is unavailable, it uses a post-training
    threshold adjustment as a fallback to ensure basic fairness mitigation.
    """

    def __init__(
        self,
        base_estimator: Optional[Any] = None,
        constraint: str = "demographic_parity",
        sensitive_features: Optional[list] = None,
        random_state: int = 42,
    ):
        """Initializes the classifier and its fairness strategy.

        Sets up the classifier to use Fairlearn's Exponentiated Gradient for in-processing
        fairness mitigation. If Fairlearn is not installed, the classifier is configured
        to use a simpler post-processing approach based on threshold optimization.
        This ensures the classifier remains functional even without the optional
        Fairlearn dependency.

        Args:
            base_estimator: The underlying estimator (e.g., LogisticRegression) to be made fair.
                If not provided, a default LogisticRegression instance is used.
            constraint: The fairness constraint to enforce, either "demographic_parity"
                or "equalized_odds". This is only used if Fairlearn is available.
            sensitive_features: The names of columns that contain sensitive attributes
                and should be used for fairness calculations.
            random_state: Seed for reproducibility of the base estimator.
        """
        super().__init__(sensitive_features)

        if not FAIRLEARN_AVAILABLE:
            self.use_fairlearn = False
            self.base_estimator = (
                base_estimator
                if base_estimator is not None
                else LogisticRegression(random_state=random_state)
            )
        else:
            self.use_fairlearn = True
            self.base_estimator = (
                base_estimator
                if base_estimator is not None
                else LogisticRegression(random_state=random_state)
            )

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
                eps=0.01,
            )

        self.constraint_name = constraint
        self.random_state = random_state
        self.categorical_encoders_ = {}
        self.categorical_features_ = []
        # Initialized here to avoid attributes being created outside __init__
        self.original_columns_ = []
        self.group_thresholds_ = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_features: Optional[pd.DataFrame] = None,
    ) -> "FairnessConstrainedClassifier":
        """Trains the classifier while enforcing fairness constraints.

        This method first preprocesses the data by label-encoding categorical features.
        It then applies the selected fairness mitigation strategy. If using Fairlearn,
        it fits the Exponentiated Gradient mitigator. Otherwise, it trains the base
        estimator and then calculates optimal decision thresholds for each sensitive
        group to approximate the desired fairness constraint.

        Args:
            X: The training input samples.
            y: The target values.
            sensitive_features: A DataFrame containing the sensitive feature values for the
                training data. If not provided, the columns specified during
                initialization will be used from X.

        Returns:
            The fitted classifier.
        """
        self._validate_input(X, y)

        if sensitive_features is None:
            if not self.sensitive_features:
                raise ValueError("sensitive_features must be provided")
            sensitive_features = X[self.sensitive_features]

        X_encoded = X.copy()
        for col in X.columns:
            if X[col].dtype in ["object", "category"]:
                self.categorical_features_.append(col)
                encoder = LabelEncoder()
                X_encoded[col] = encoder.fit_transform(X[col].astype(str))
                self.categorical_encoders_[col] = encoder

        sensitive_features_encoded = sensitive_features.copy()
        for col in sensitive_features.columns:
            if sensitive_features[col].dtype in ["object", "category"]:
                if col not in self.categorical_encoders_:
                    encoder = LabelEncoder()
                    sensitive_features_encoded[col] = encoder.fit_transform(
                        sensitive_features[col].astype(str)
                    )
                    self.categorical_encoders_[col] = encoder
                else:
                    sensitive_features_encoded[col] = self.categorical_encoders_[
                        col
                    ].transform(sensitive_features[col].astype(str))

        self.classes_ = np.unique(y)
        self.original_columns_ = list(X_encoded.columns)

        if self.use_fairlearn:
            self.mitigator.fit(
                X_encoded, y, sensitive_features=sensitive_features_encoded
            )
        else:
            self.base_estimator.fit(X_encoded, y)
            self._fit_threshold_optimization(X_encoded, y, sensitive_features_encoded)

        self.is_fitted_ = True
        return self

    def _fit_threshold_optimization(
        self, X: pd.DataFrame, y: pd.Series, sensitive_features: pd.DataFrame
    ) -> None:
        """Calculates group-specific thresholds as a fallback fairness method.

        This post-processing technique is used when Fairlearn is unavailable. It adjusts
        the decision threshold for each sensitive attribute group to align their
        positive prediction rate with the overall positive rate, approximating
        demographic parity. This is a simpler alternative to in-processing methods.
        """
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
                        if abs(pred_pos_rate - pos_rate_target) < abs(
                            (group_probs >= best_threshold).mean() - pos_rate_target
                        ):
                            best_threshold = threshold

                thresholds[group] = best_threshold

            self.group_thresholds_[sensitive_col] = thresholds

    def predict(
        self, X: pd.DataFrame, sensitive_features: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """Generates predictions adjusted for fairness.

        If Fairlearn was used for training, this method uses the mitigated model's
        `predict` method. If the fallback threshold optimization was used, it first
        gets prediction probabilities from the base model and then applies the
        group-specific thresholds to determine the final predictions. This ensures
        the fairness logic applied during `fit` is also used at inference time.

        Args:
            X: The input samples for prediction.
            sensitive_features: A DataFrame with sensitive feature values. If not provided,
                the columns specified during initialization will be used from X.

        Returns:
            An array of predicted class labels.
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
                    known_classes = set(encoder.classes_)
                    X_encoded[col] = X_encoded[col].apply(
                        lambda x: encoder.transform([str(x)])[0]
                        if str(x) in known_classes
                        else encoder.transform([encoder.classes_[0]])[0]
                    )

        sensitive_features_encoded = sensitive_features.copy()
        for col in sensitive_features.columns:
            if col in self.categorical_encoders_:
                encoder = self.categorical_encoders_[col]
                try:
                    sensitive_features_encoded[col] = encoder.transform(
                        sensitive_features[col].astype(str)
                    )
                except ValueError:
                    known_classes = set(encoder.classes_)
                    sensitive_features_encoded[col] = sensitive_features[col].apply(
                        lambda x: encoder.transform([str(x)])[0]
                        if str(x) in known_classes
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

    def predict_proba(
        self, X: pd.DataFrame, sensitive_features: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """Predicts class probabilities, with a notable caveat for the Fairlearn path.

        When using the fallback method, this returns the true probabilities from the
        base estimator. However, when using Fairlearn's Exponentiated Gradient,
        obtaining direct probabilities from the mitigated model is not straightforward.
        As a workaround, this method returns fixed high/low probability values
        (e.g., 0.8/0.2) based on the final prediction. This provides a probability-like
        output but does not reflect the model's true confidence.

        Args:
            X: The input samples for prediction.
            sensitive_features: A DataFrame with sensitive feature values.

        Returns:
            An array of class probabilities.
        """
        self._check_is_fitted()
        self._validate_input(X)

        X_encoded = X.copy()
        for col in self.categorical_features_:
            if col in X_encoded.columns:
                encoder = self.categorical_encoders_[col]
                try:
                    X_encoded[col] = encoder.transform(X_encoded[col].astype(str))
                except ValueError:
                    known_classes = set(encoder.classes_)
                    X_encoded[col] = X_encoded[col].apply(
                        lambda x: encoder.transform([str(x)])[0]
                        if str(x) in known_classes
                        else encoder.transform([encoder.classes_[0]])[0]
                    )

        if self.use_fairlearn:
            predictions = self.predict(X, sensitive_features)
            n_samples = len(predictions)
            n_classes = len(self.classes_)
            probas = np.zeros((n_samples, n_classes))

            for i, pred in enumerate(predictions):
                probas[i, pred] = 0.8
                probas[i, 1 - pred] = 0.2

            return probas
        else:
            return self.base_estimator.predict_proba(X_encoded)

    def get_fairness_info(self) -> Dict[str, Any]:
        """Retrieves details about the fairness configuration of the classifier."""
        info = super().get_fairness_info()
        info.update(
            {
                "constraint": self.constraint_name,
                "base_estimator": str(self.base_estimator),
                "uses_fairlearn": self.use_fairlearn,
            }
        )
        return info