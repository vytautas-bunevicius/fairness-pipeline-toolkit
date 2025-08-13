"""Training module for fairness-constrained model training."""

from .fairness_constrained_classifier import FairnessConstrainedClassifier
from .fair_classifier import FairClassifier

__all__ = ["FairnessConstrainedClassifier", "FairClassifier"]
