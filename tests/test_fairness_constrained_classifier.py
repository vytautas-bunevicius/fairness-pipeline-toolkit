"""Unit tests for fairness constrained classifier."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from fairness_pipeline_toolkit.training.fairness_constrained_classifier import FairnessConstrainedClassifier


class TestFairnessConstrainedClassifier:
    """Test cases for FairnessConstrainedClassifier class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 1000
        
        X = pd.DataFrame({
            'age': np.random.normal(35, 10, n_samples),
            'income': np.random.lognormal(10, 0.5, n_samples),
            'race': np.random.choice(['White', 'Black'], n_samples),
            'sex': np.random.choice(['Male', 'Female'], n_samples)
        })
        
        # Create target with some correlation to features
        logit = -2 + 0.1 * X['age'] + 0.00002 * X['income']
        prob = 1 / (1 + np.exp(-logit))
        y = np.random.binomial(1, prob, n_samples)
        
        return X, pd.Series(y)
    
    @pytest.fixture
    def sample_features_only(self):
        """Create sample feature data without sensitive features."""
        np.random.seed(42)
        n_samples = 100
        
        return pd.DataFrame({
            'age': np.random.normal(35, 10, n_samples),
            'income': np.random.lognormal(10, 0.5, n_samples),
        })
    
    def test_init_default_params(self):
        """Test classifier initialization with default parameters."""
        classifier = FairnessConstrainedClassifier(
            sensitive_features=['race']
        )
        assert classifier.sensitive_features == ['race']
        assert classifier.constraint_name == 'demographic_parity'
        assert classifier.random_state == 42
    
    def test_init_custom_params(self):
        """Test classifier initialization with custom parameters."""
        from sklearn.ensemble import RandomForestClassifier
        base_estimator = RandomForestClassifier()
        
        classifier = FairnessConstrainedClassifier(
            base_estimator=base_estimator,
            constraint='equalized_odds',
            sensitive_features=['race', 'sex'],
            random_state=123
        )
        
        assert classifier.base_estimator is base_estimator
        assert classifier.constraint_name == 'equalized_odds'
        assert classifier.sensitive_features == ['race', 'sex']
        assert classifier.random_state == 123
    
    def test_init_invalid_constraint(self):
        """Test initialization with invalid constraint."""
        with pytest.raises(ValueError, match="Unknown constraint"):
            FairnessConstrainedClassifier(
                constraint='invalid_constraint',
                sensitive_features=['race']
            )
    
    @patch('fairness_pipeline_toolkit.training.fairness_constrained_classifier.FAIRLEARN_AVAILABLE', False)
    def test_init_without_fairlearn(self):
        """Test initialization when fairlearn is not available."""
        classifier = FairnessConstrainedClassifier(
            sensitive_features=['race']
        )
        assert classifier.use_fairlearn is False
        assert hasattr(classifier, 'base_estimator')
    
    def test_fit_basic(self, sample_data):
        """Test basic fit functionality."""
        X, y = sample_data
        classifier = FairnessConstrainedClassifier(
            sensitive_features=['race', 'sex']
        )
        
        fitted_classifier = classifier.fit(X, y)
        assert fitted_classifier is classifier  # Returns self
        assert classifier.is_fitted_
        assert hasattr(classifier, 'classes_')
        assert len(classifier.classes_) == 2  # Binary classification
    
    def test_fit_with_separate_sensitive_features(self, sample_data):
        """Test fit with sensitive features provided separately."""
        X, y = sample_data
        sensitive_features = X[['race', 'sex']]
        X_features = X[['age', 'income']]  # Without sensitive features
        
        classifier = FairnessConstrainedClassifier(
            sensitive_features=['race', 'sex']
        )
        
        classifier.fit(X_features, y, sensitive_features)
        assert classifier.is_fitted_
    
    def test_fit_no_sensitive_features_provided(self, sample_features_only):
        """Test fit when no sensitive features are available."""
        X = sample_features_only
        y = pd.Series([0, 1] * 50)  # Binary target
        
        classifier = FairnessConstrainedClassifier()
        
        with pytest.raises(ValueError, match="sensitive_features must be provided"):
            classifier.fit(X, y)
    
    @patch('fairness_pipeline_toolkit.training.fairness_constrained_classifier.FAIRLEARN_AVAILABLE', True)
    def test_fit_with_fairlearn(self, sample_data):
        """Test fit with fairlearn available."""
        X, y = sample_data
        
        # Mock fairlearn components
        with patch('fairness_pipeline_toolkit.training.fairness_constrained_classifier.ExponentiatedGradient') as mock_eg:
            mock_mitigator = MagicMock()
            mock_eg.return_value = mock_mitigator
            
            classifier = FairnessConstrainedClassifier(
                sensitive_features=['race']
            )
            classifier.fit(X, y)
            
            # Verify fairlearn mitigator was used
            mock_mitigator.fit.assert_called_once()
    
    @patch('fairness_pipeline_toolkit.training.fairness_constrained_classifier.FAIRLEARN_AVAILABLE', False)
    def test_fit_fallback_method(self, sample_data):
        """Test fit with fallback method when fairlearn unavailable."""
        X, y = sample_data
        
        classifier = FairnessConstrainedClassifier(
            sensitive_features=['race']
        )
        classifier.fit(X, y)
        
        assert classifier.is_fitted_
        assert hasattr(classifier, 'group_thresholds_')
        
        # Check that group thresholds were computed
        assert 'race' in classifier.group_thresholds_
        thresholds = classifier.group_thresholds_['race']
        assert len(thresholds) > 0
    
    def test_predict_basic(self, sample_data):
        """Test basic predict functionality."""
        X, y = sample_data
        classifier = FairnessConstrainedClassifier(
            sensitive_features=['race']
        )
        classifier.fit(X, y)
        
        predictions = classifier.predict(X)
        
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)  # Binary predictions
    
    def test_predict_with_separate_sensitive_features(self, sample_data):
        """Test predict with sensitive features provided separately."""
        X, y = sample_data
        classifier = FairnessConstrainedClassifier(
            sensitive_features=['race']
        )
        classifier.fit(X, y)
        
        X_features = X[['age', 'income', 'sex']]  # Include non-sensitive features
        sensitive_features = X[['race']]  # Only the declared sensitive features
        
        predictions = classifier.predict(X_features, sensitive_features)
        assert len(predictions) == len(X_features)
    
    def test_predict_not_fitted(self, sample_features_only):
        """Test predict without fitting first."""
        classifier = FairnessConstrainedClassifier(
            sensitive_features=['race']
        )
        
        with pytest.raises(ValueError, match="Classifier must be fitted"):
            classifier.predict(sample_features_only)
    
    def test_predict_proba_basic(self, sample_data):
        """Test predict_proba functionality."""
        X, y = sample_data
        classifier = FairnessConstrainedClassifier(
            sensitive_features=['race']
        )
        classifier.fit(X, y)
        
        probabilities = classifier.predict_proba(X)
        
        assert probabilities.shape == (len(X), 2)  # Binary classification
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)  # Valid probabilities
    
    @patch('fairness_pipeline_toolkit.training.fairness_constrained_classifier.FAIRLEARN_AVAILABLE', False)
    def test_predict_proba_fallback(self, sample_data):
        """Test predict_proba with fallback method."""
        X, y = sample_data
        classifier = FairnessConstrainedClassifier(
            sensitive_features=['race']
        )
        classifier.fit(X, y)
        
        probabilities = classifier.predict_proba(X)
        
        assert probabilities.shape == (len(X), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_get_fairness_info(self, sample_data):
        """Test get_fairness_info method."""
        X, y = sample_data
        classifier = FairnessConstrainedClassifier(
            constraint='equalized_odds',
            sensitive_features=['race']
        )
        classifier.fit(X, y)
        
        info = classifier.get_fairness_info()
        
        # Check required fields
        assert 'constraint' in info
        assert 'base_estimator' in info
        assert 'uses_fairlearn' in info
        assert 'sensitive_features' in info
        
        # Check values
        assert info['constraint'] == 'equalized_odds'
        assert info['sensitive_features'] == ['race']
        assert isinstance(info['uses_fairlearn'], bool)
    
    def test_get_fairness_info_not_fitted(self):
        """Test get_fairness_info when not fitted."""
        classifier = FairnessConstrainedClassifier(
            sensitive_features=['race']
        )
        
        # Should still work (returns basic info)
        info = classifier.get_fairness_info()
        assert 'constraint' in info
        assert 'base_estimator' in info
    
    def test_threshold_optimization_edge_cases(self):
        """Test threshold optimization with edge cases."""
        # Create classifier for testing threshold optimization
        classifier = FairnessConstrainedClassifier(
            sensitive_features=['group']
        )
        classifier.use_fairlearn = False  # Force fallback method
        
        # Mock base estimator
        classifier.base_estimator = MagicMock()
        classifier.base_estimator.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2], [0.4, 0.6]])
        
        X = pd.DataFrame({'feature': [1, 2, 3]})
        y = pd.Series([1, 0, 1])
        sensitive_features = pd.DataFrame({'group': ['A', 'B', 'A']})
        
        # Should handle threshold optimization without error
        classifier._fit_threshold_optimization(X, y, sensitive_features)
        
        assert hasattr(classifier, 'group_thresholds_')
        assert 'group' in classifier.group_thresholds_
    
    def test_empty_groups_in_prediction(self, sample_data):
        """Test prediction with groups not seen during training."""
        X, y = sample_data
        classifier = FairnessConstrainedClassifier(
            sensitive_features=['race']
        )
        classifier.fit(X, y)
        
        # Create test data with new group
        X_test = X.head(10).copy()
        X_test['race'] = 'Unknown'  # New group not seen in training
        
        # Should handle gracefully (may use default behavior)
        predictions = classifier.predict(X_test)
        assert len(predictions) == len(X_test)
    
    def test_single_class_target(self):
        """Test handling of single-class target variable."""
        X = pd.DataFrame({
            'feature': [1, 2, 3, 4],
            'race': ['A', 'B', 'A', 'B']
        })
        y = pd.Series([1, 1, 1, 1])  # All same class
        
        classifier = FairnessConstrainedClassifier(
            sensitive_features=['race']
        )
        
        # Should handle single class without error
        classifier.fit(X, y)
        predictions = classifier.predict(X)
        
        # All predictions should be the same class
        assert all(pred == 1 for pred in predictions)