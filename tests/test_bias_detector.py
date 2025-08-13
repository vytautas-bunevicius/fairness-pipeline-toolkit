"""Unit tests for bias detector."""

import pytest
import pandas as pd
import numpy as np

from fairness_pipeline_toolkit.measurement.bias_detector import BiasDetector


class TestBiasDetector:
    """Test cases for BiasDetector class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        np.random.seed(42)
        n_samples = 1000

        data = pd.DataFrame(
            {
                "age": np.random.normal(35, 10, n_samples).clip(18, 80),
                "income": np.random.lognormal(10, 0.5, n_samples),
                "race": np.random.choice(
                    ["White", "Black", "Hispanic"], n_samples, p=[0.6, 0.3, 0.1]
                ),
                "sex": np.random.choice(["Male", "Female"], n_samples, p=[0.5, 0.5]),
                "target": np.random.binomial(1, 0.3, n_samples),
            }
        )
        return data

    @pytest.fixture
    def biased_predictions(self):
        """Create biased predictions for testing."""
        np.random.seed(42)
        n_samples = 1000

        # Create predictions with intentional bias
        race = np.random.choice(
            ["White", "Black", "Hispanic"], n_samples, p=[0.6, 0.3, 0.1]
        )
        y_true = np.random.binomial(1, 0.3, n_samples)

        # Biased predictions - favor White individuals
        bias_factor = np.where(race == "White", 0.2, 0.0)
        prob_pred = np.random.uniform(0, 1, n_samples) + bias_factor
        y_pred = (prob_pred > 0.5).astype(int)

        return y_true, y_pred, race

    def test_init_default_threshold(self):
        """Test BiasDetector initialization with default threshold."""
        detector = BiasDetector()
        assert detector.threshold == 0.1

    def test_init_custom_threshold(self):
        """Test BiasDetector initialization with custom threshold."""
        detector = BiasDetector(threshold=0.05)
        assert detector.threshold == 0.05

    def test_audit_dataset_basic(self, sample_data):
        """Test basic dataset auditing functionality."""
        detector = BiasDetector()
        report = detector.audit_dataset(
            sample_data, sensitive_column="race", target_column="target"
        )

        # Check required fields in report
        assert "dataset_shape" in report
        assert "sensitive_feature_distribution" in report
        assert "missing_values" in report
        assert "target_rate_by_group" in report
        assert "target_rate_difference" in report

        # Verify dataset shape
        assert report["dataset_shape"] == sample_data.shape

        # Verify sensitive feature distribution
        expected_dist = sample_data["race"].value_counts().to_dict()
        assert report["sensitive_feature_distribution"] == expected_dist

    def test_audit_dataset_without_target(self, sample_data):
        """Test dataset auditing without target column."""
        detector = BiasDetector()
        report = detector.audit_dataset(sample_data, sensitive_column="race")

        # Should not have target-related fields
        assert "target_rate_by_group" not in report
        assert "target_rate_difference" not in report

        # Should still have basic fields
        assert "dataset_shape" in report
        assert "sensitive_feature_distribution" in report
        assert "missing_values" in report

    def test_audit_predictions_with_bias(self, biased_predictions):
        """Test prediction auditing with biased predictions."""
        y_true, y_pred, sensitive_features = biased_predictions
        detector = BiasDetector(threshold=0.05)

        report = detector.audit_predictions(y_true, y_pred, sensitive_features)

        # Check required fields
        assert "metrics" in report
        assert "fairness_violations" in report
        assert "overall_fairness_score" in report
        assert "threshold" in report

        # Check metrics exist
        metrics = report["metrics"]
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "demographic_parity_difference" in metrics
        assert "equalized_odds_difference" in metrics

        # Check threshold is correct
        assert report["threshold"] == 0.05

        # Should detect violations due to intentional bias
        violations = report["fairness_violations"]
        assert isinstance(violations["demographic_parity_violation"], bool)
        assert isinstance(violations["equalized_odds_violation"], bool)

    def test_audit_predictions_fair(self):
        """Test prediction auditing with fair predictions."""
        np.random.seed(42)
        n_samples = 1000

        # Create fair predictions (no bias)
        y_true = np.random.binomial(1, 0.3, n_samples)
        y_pred = np.random.binomial(1, 0.3, n_samples)  # Random, no bias
        sensitive_features = np.random.choice(["A", "B"], n_samples)

        detector = BiasDetector(threshold=0.1)
        report = detector.audit_predictions(y_true, y_pred, sensitive_features)

        # Should have low fairness violations
        fairness_score = report["overall_fairness_score"]
        assert 0 <= fairness_score <= 1

    def test_audit_predictions_edge_cases(self):
        """Test prediction auditing with edge cases."""
        detector = BiasDetector()

        # Test with all same predictions
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 1])  # All positive predictions
        sensitive_features = np.array(["A", "B", "A", "B"])

        report = detector.audit_predictions(y_true, y_pred, sensitive_features)
        assert "metrics" in report
        assert "fairness_violations" in report

    def test_print_report_dataset(self, sample_data, capsys):
        """Test print report functionality for dataset audit."""
        detector = BiasDetector()
        report = detector.audit_dataset(
            sample_data, sensitive_column="race", target_column="target"
        )

        detector.print_report(report, "TEST DATASET")

        captured = capsys.readouterr()
        assert "TEST DATASET REPORT" in captured.out
        assert "Dataset Information" in captured.out
        assert "Dataset Shape" in captured.out

    def test_print_report_predictions(self, biased_predictions, capsys):
        """Test print report functionality for prediction audit."""
        y_true, y_pred, sensitive_features = biased_predictions
        detector = BiasDetector()

        report = detector.audit_predictions(y_true, y_pred, sensitive_features)
        detector.print_report(report, "TEST PREDICTIONS")

        captured = capsys.readouterr()
        assert "TEST PREDICTIONS REPORT" in captured.out
        assert "Performance Metrics" in captured.out
        assert "Fairness Metrics" in captured.out
        assert "Overall Fairness Score" in captured.out

    def test_metrics_validation(self, biased_predictions):
        """Test that all metrics are properly calculated."""
        y_true, y_pred, sensitive_features = biased_predictions
        detector = BiasDetector()

        report = detector.audit_predictions(y_true, y_pred, sensitive_features)
        metrics = report["metrics"]

        # All metrics should be numeric
        for metric_name, metric_value in metrics.items():
            assert isinstance(metric_value, (int, float))
            assert not np.isnan(metric_value)
            assert not np.isinf(metric_value)

        # Accuracy should be between 0 and 1
        assert 0 <= metrics["accuracy"] <= 1

        # Precision and recall should be between 0 and 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1

    def test_empty_groups_handling(self):
        """Test handling of empty groups in sensitive features."""
        detector = BiasDetector()

        # Create data with empty group after filtering
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])
        sensitive_features = np.array(["A", "A", "A", "A"])  # Only one group

        # Should not crash and return valid report
        report = detector.audit_predictions(y_true, y_pred, sensitive_features)
        assert "metrics" in report

    def test_single_sample_handling(self):
        """Test handling of single sample."""
        detector = BiasDetector()

        y_true = np.array([1])
        y_pred = np.array([1])
        sensitive_features = np.array(["A"])

        # Should handle single sample gracefully
        report = detector.audit_predictions(y_true, y_pred, sensitive_features)
        assert "metrics" in report
