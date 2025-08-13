"""Fairness metrics calculation utilities."""

from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class FairnessMetrics:
    """Calculate various fairness metrics for machine learning models."""

    @staticmethod
    def demographic_parity_difference(
        y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: np.ndarray
    ) -> float:
        """Calculate demographic parity difference."""
        df = pd.DataFrame(
            {"y_true": y_true, "y_pred": y_pred, "sensitive": sensitive_features}
        )

        pos_rates = df.groupby("sensitive")["y_pred"].mean()
        return float(pos_rates.max() - pos_rates.min())

    @staticmethod
    def equalized_odds_difference(
        y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: np.ndarray
    ) -> float:
        """Calculate equalized odds difference."""
        df = pd.DataFrame(
            {"y_true": y_true, "y_pred": y_pred, "sensitive": sensitive_features}
        )

        tpr_by_group = df[df["y_true"] == 1].groupby("sensitive")["y_pred"].mean()
        tpr_diff = tpr_by_group.max() - tpr_by_group.min()

        fpr_by_group = df[df["y_true"] == 0].groupby("sensitive")["y_pred"].mean()
        fpr_diff = fpr_by_group.max() - fpr_by_group.min()

        return float(max(tpr_diff, fpr_diff))

    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive fairness and performance metrics."""
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "demographic_parity_difference": FairnessMetrics.demographic_parity_difference(
                y_true, y_pred, sensitive_features
            ),
            "equalized_odds_difference": FairnessMetrics.equalized_odds_difference(
                y_true, y_pred, sensitive_features
            ),
        }

        return metrics
