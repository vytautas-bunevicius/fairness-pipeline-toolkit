"""Provides a collection of functions for calculating common fairness metrics.

This module centralizes the mathematical implementations of fairness metrics to
ensure they are calculated consistently throughout the toolkit. These metrics
are essential for quantifying bias in model predictions.
"""

from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class FairnessMetrics:
    """A container for static methods that calculate fairness and performance metrics.

    This class acts as a namespace to group related metric calculations. Using a
    class structure allows for clear organization and easy importation of all
    related metrics, without needing to instantiate the class.
    """

    @staticmethod
    def demographic_parity_difference(
        y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: np.ndarray
    ) -> float:
        """Calculates the demographic parity difference.

        This metric measures the difference in the rate of positive outcomes between
        the best-fared and worst-fared demographic groups. A low value indicates
        that the model is making positive predictions at a similar rate across groups,
        regardless of their true outcome.
        """
        df = pd.DataFrame(
            {"y_true": y_true, "y_pred": y_pred, "sensitive": sensitive_features}
        )

        pos_rates = df.groupby("sensitive")["y_pred"].mean()
        return float(pos_rates.max() - pos_rates.min())

    @staticmethod
    def equalized_odds_difference(
        y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: np.ndarray
    ) -> float:
        """Calculates the equalized odds difference.

        This metric is stricter than demographic parity and measures the maximum
        difference in either the true positive rate or the false positive rate
        between demographic groups. A low value indicates that the model performs
        equally well for all groups, for both positive and negative instances.
        """
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
        """Calculates a standard set of fairness and performance metrics.

        This function serves as a convenient wrapper to compute both model performance
        (e.g., accuracy) and key fairness metrics in a single call. This is useful for
        generating a complete overview of a model's behavior. The `zero_division=0`
        parameter is set in precision/recall to prevent crashes on subsets of data
        with no positive instances.
        """
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
