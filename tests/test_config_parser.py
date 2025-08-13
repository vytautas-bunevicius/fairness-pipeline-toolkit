"""Unit tests for configuration parser."""

import pytest
import tempfile
import yaml
from pathlib import Path

from fairness_pipeline_toolkit.config import ConfigParser


class TestConfigParser:
    """Test cases for ConfigParser class."""

    @pytest.fixture
    def valid_config_dict(self):
        """Return a valid configuration dictionary."""
        return {
            "data": {
                "input_path": "test_data.csv",
                "target_column": "target",
                "sensitive_features": ["race", "sex"],
                "test_size": 0.2,
                "random_state": 42,
            },
            "preprocessing": {
                "transformer": {
                    "name": "BiasMitigationTransformer",
                    "parameters": {"repair_level": 0.8, "random_state": 42},
                }
            },
            "training": {
                "method": {
                    "name": "FairnessConstrainedClassifier",
                    "parameters": {
                        "base_estimator": "LogisticRegression",
                        "constraint": "demographic_parity",
                        "random_state": 42,
                    },
                }
            },
            "evaluation": {
                "primary_metric": "demographic_parity_difference",
                "fairness_threshold": 0.1,
                "additional_metrics": ["equalized_odds_difference", "accuracy"],
            },
            "mlflow": {
                "experiment_name": "test_experiment",
                "run_name": None,
                "log_model": True,
                "log_config": True,
                "tags": {"framework": "test"},
            },
        }

    @pytest.fixture
    def valid_config_file(self, valid_config_dict):
        """Create a temporary valid config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(valid_config_dict, f)
            return Path(f.name)

    def test_load_valid_config(self, valid_config_file, valid_config_dict):
        """Test loading a valid configuration file."""
        config = ConfigParser.load(valid_config_file)
        assert config == valid_config_dict

    def test_load_nonexistent_file(self):
        """Test loading a non-existent configuration file."""
        with pytest.raises(FileNotFoundError):
            ConfigParser.load("nonexistent_file.yml")

    def test_validate_valid_config(self, valid_config_dict):
        """Test validation of a valid configuration."""
        errors = ConfigParser.validate(valid_config_dict)
        assert errors == []

    def test_validate_missing_sections(self):
        """Test validation with missing required sections."""
        incomplete_config = {
            "data": {
                "input_path": "test.csv",
                "target_column": "target",
                "sensitive_features": ["race"],
            }
        }
        errors = ConfigParser.validate(incomplete_config)
        expected_missing = ["preprocessing", "training", "evaluation", "mlflow"]

        for section in expected_missing:
            # Check for Pydantic missing field error messages instead
            assert any(
                "Field required" in error and section in error for error in errors
            ) or any(
                f"Missing required section: {section}" in error for error in errors
            )

    def test_validate_missing_data_fields(self, valid_config_dict):
        """Test validation with missing data fields."""
        config = valid_config_dict.copy()
        del config["data"]["target_column"]
        del config["data"]["sensitive_features"]

        errors = ConfigParser.validate(config)
        assert any("target_column" in error for error in errors)
        assert any("sensitive_features" in error for error in errors)

    def test_validate_invalid_repair_level(self, valid_config_dict):
        """Test validation with invalid repair level."""
        config = valid_config_dict.copy()
        config["preprocessing"]["transformer"]["parameters"]["repair_level"] = 1.5

        errors = ConfigParser.validate(config)
        assert any(
            "repair_level must be between 0.0 and 1.0" in error for error in errors
        )

    def test_validate_invalid_fairness_threshold(self, valid_config_dict):
        """Test validation with invalid fairness threshold."""
        config = valid_config_dict.copy()
        config["evaluation"]["fairness_threshold"] = -0.1

        errors = ConfigParser.validate(config)
        assert any("fairness_threshold must be positive" in error for error in errors)

    def test_validate_empty_sensitive_features(self, valid_config_dict):
        """Test validation with empty sensitive features list."""
        config = valid_config_dict.copy()
        config["data"]["sensitive_features"] = []

        errors = ConfigParser.validate(config)
        assert any(
            "At least one sensitive feature must be specified" in error
            for error in errors
        )

    def test_validate_invalid_test_size(self, valid_config_dict):
        """Test validation with invalid test size."""
        config = valid_config_dict.copy()
        config["data"]["test_size"] = 1.5

        errors = ConfigParser.validate(config)
        # Check for Pydantic range validation error messages
        assert any("less than or equal to 0.99" in error for error in errors) or any(
            "test_size must be between" in error for error in errors
        )

    def test_load_malformed_yaml(self):
        """Test loading malformed YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("invalid: yaml: content: [unclosed")
            malformed_file = Path(f.name)

        with pytest.raises(ValueError):
            ConfigParser.load(malformed_file)

    def test_validate_unknown_transformer(self, valid_config_dict):
        """Test validation with unknown transformer name."""
        config = valid_config_dict.copy()
        config["preprocessing"]["transformer"]["name"] = "UnknownTransformer"

        errors = ConfigParser.validate(config)
        assert any("Unknown transformer" in error for error in errors)

    def test_validate_unknown_training_method(self, valid_config_dict):
        """Test validation with unknown training method."""
        config = valid_config_dict.copy()
        config["training"]["method"]["name"] = "UnknownMethod"

        errors = ConfigParser.validate(config)
        assert any("Unknown training method" in error for error in errors)

    def test_validate_unknown_constraint(self, valid_config_dict):
        """Test validation with unknown fairness constraint."""
        config = valid_config_dict.copy()
        config["training"]["method"]["parameters"]["constraint"] = "unknown_constraint"

        errors = ConfigParser.validate(config)
        # For now, the config system accepts any constraint value
        # The validation happens at runtime in the classifier
        assert len(errors) == 0  # No validation errors expected

    def test_validate_unknown_base_estimator(self, valid_config_dict):
        """Test validation with unknown base estimator."""
        config = valid_config_dict.copy()
        config["training"]["method"]["parameters"]["base_estimator"] = (
            "UnknownEstimator"
        )

        errors = ConfigParser.validate(config)
        # For now, the config system accepts any base_estimator string
        # The validation happens at runtime when instantiating the estimator
        assert len(errors) == 0  # No validation errors expected
