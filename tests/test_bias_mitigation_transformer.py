"""Unit tests for bias mitigation transformer."""

import pytest
import pandas as pd
import numpy as np

from fairness_pipeline_toolkit.pipeline.bias_mitigation_transformer import BiasMitigationTransformer


class TestBiasMitigationTransformer:
    """Test cases for BiasMitigationTransformer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create biased dataset
        race = np.random.choice(['White', 'Black'], n_samples, p=[0.7, 0.3])
        sex = np.random.choice(['Male', 'Female'], n_samples, p=[0.5, 0.5])
        
        # Create features with group-based bias
        age_bias = np.where(race == 'White', 5, 0)  # White individuals tend to be older
        income_bias = np.where((race == 'White') & (sex == 'Male'), 10000, 0)
        
        data = pd.DataFrame({
            'age': np.random.normal(35, 10, n_samples) + age_bias,
            'income': np.random.lognormal(10, 0.5, n_samples) + income_bias,
            'education_years': np.random.normal(12, 3, n_samples).clip(8, 20),
            'race': race,
            'sex': sex
        })
        return data
    
    @pytest.fixture
    def simple_data(self):
        """Create simple dataset for testing edge cases."""
        return pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [10.0, 20.0, 30.0, 40.0],
            'sensitive': ['A', 'B', 'A', 'B']
        })
    
    def test_init_default_params(self):
        """Test transformer initialization with default parameters."""
        transformer = BiasMitigationTransformer(sensitive_features=['race'])
        assert transformer.sensitive_features == ['race']
        assert transformer.repair_level == 1.0
        assert transformer.method == 'mean_matching'
        assert transformer.random_state == 42
    
    def test_init_custom_params(self):
        """Test transformer initialization with custom parameters."""
        transformer = BiasMitigationTransformer(
            sensitive_features=['race', 'sex'],
            repair_level=0.5,
            random_state=123,
            method='multivariate_repair'
        )
        assert transformer.sensitive_features == ['race', 'sex']
        assert transformer.repair_level == 0.5
        assert transformer.method == 'multivariate_repair'
        assert transformer.random_state == 123
    
    def test_fit_basic(self, sample_data):
        """Test basic fit functionality."""
        transformer = BiasMitigationTransformer(sensitive_features=['race'])
        
        # Should not raise exception
        fitted_transformer = transformer.fit(sample_data)
        assert fitted_transformer is transformer  # Returns self
        assert transformer.is_fitted_
        
        # Check fitted attributes
        assert hasattr(transformer, 'non_sensitive_features_')
        assert hasattr(transformer, 'group_stats_')
        assert hasattr(transformer, 'feature_means_')
        
        # Non-sensitive features should exclude 'race'
        expected_non_sensitive = ['age', 'income', 'education_years', 'sex']
        assert set(transformer.non_sensitive_features_) == set(expected_non_sensitive)
    
    def test_fit_multiple_sensitive_features(self, sample_data):
        """Test fit with multiple sensitive features."""
        transformer = BiasMitigationTransformer(sensitive_features=['race', 'sex'])
        transformer.fit(sample_data)
        
        assert transformer.is_fitted_
        assert 'race' in transformer.group_stats_
        assert 'sex' in transformer.group_stats_
        
        # Non-sensitive features should exclude both 'race' and 'sex'
        expected_non_sensitive = ['age', 'income', 'education_years']
        assert set(transformer.non_sensitive_features_) == set(expected_non_sensitive)
    
    def test_fit_no_non_sensitive_features(self, simple_data):
        """Test fit when no non-sensitive features remain."""
        # All features are sensitive
        all_features = simple_data.columns.tolist()
        transformer = BiasMitigationTransformer(sensitive_features=all_features)
        
        with pytest.raises(ValueError, match="At least one non-sensitive feature is required"):
            transformer.fit(simple_data)
    
    def test_transform_mean_matching(self, sample_data):
        """Test transform with mean matching method."""
        transformer = BiasMitigationTransformer(
            sensitive_features=['race'],
            repair_level=0.8,
            method='mean_matching'
        )
        transformer.fit(sample_data)
        
        transformed_data = transformer.transform(sample_data)
        
        # Check that data shape is preserved
        assert transformed_data.shape == sample_data.shape
        assert list(transformed_data.columns) == list(sample_data.columns)
        
        # Sensitive features should remain unchanged
        pd.testing.assert_series_equal(
            transformed_data['race'], 
            sample_data['race'],
            check_names=False
        )
        
        # Numerical non-sensitive features should be modified
        for feature in transformer.non_sensitive_features_:
            if feature != 'race' and feature not in transformer.categorical_features_:
                # Check that numerical values have changed (bias mitigation applied)
                assert not transformed_data[feature].equals(sample_data[feature])
        
        # Categorical non-sensitive features should remain unchanged
        for feature in transformer.categorical_features_:
            if feature in transformer.non_sensitive_features_:
                pd.testing.assert_series_equal(
                    transformed_data[feature],
                    sample_data[feature],
                    check_names=False
                )
    
    def test_transform_multivariate_repair(self, sample_data):
        """Test transform with multivariate repair method."""
        transformer = BiasMitigationTransformer(
            sensitive_features=['race'],
            repair_level=0.6,
            method='multivariate_repair'
        )
        transformer.fit(sample_data)
        
        transformed_data = transformer.transform(sample_data)
        
        # Basic checks
        assert transformed_data.shape == sample_data.shape
        assert list(transformed_data.columns) == list(sample_data.columns)
        
        # Sensitive features should remain unchanged
        pd.testing.assert_series_equal(
            transformed_data['race'], 
            sample_data['race'],
            check_names=False
        )
    
    def test_transform_covariance_matching(self, sample_data):
        """Test transform with covariance matching method."""
        transformer = BiasMitigationTransformer(
            sensitive_features=['race'],
            repair_level=0.7,
            method='covariance_matching'
        )
        transformer.fit(sample_data)
        
        transformed_data = transformer.transform(sample_data)
        
        # Basic checks
        assert transformed_data.shape == sample_data.shape
        assert list(transformed_data.columns) == list(sample_data.columns)
    
    def test_transform_not_fitted(self, sample_data):
        """Test transform without fitting first."""
        transformer = BiasMitigationTransformer(sensitive_features=['race'])
        
        with pytest.raises(ValueError, match="Transformer must be fitted"):
            transformer.transform(sample_data)
    
    def test_transform_different_data(self, sample_data):
        """Test transform on different data after fitting."""
        transformer = BiasMitigationTransformer(sensitive_features=['race'])
        transformer.fit(sample_data)
        
        # Create new data with same structure
        new_data = sample_data.sample(100, random_state=123).reset_index(drop=True)
        transformed_data = transformer.transform(new_data)
        
        assert transformed_data.shape == new_data.shape
        assert list(transformed_data.columns) == list(new_data.columns)
    
    def test_repair_level_zero(self, sample_data):
        """Test transform with repair level 0 (no repair)."""
        transformer = BiasMitigationTransformer(
            sensitive_features=['race'],
            repair_level=0.0
        )
        transformer.fit(sample_data)
        
        transformed_data = transformer.transform(sample_data)
        
        # With repair level 0, data should be essentially unchanged
        # (except for potential small numerical differences)
        non_sensitive_cols = transformer.non_sensitive_features_
        for col in non_sensitive_cols:
            if col in transformer.categorical_features_:
                # Categorical features should be exactly the same
                pd.testing.assert_series_equal(
                    transformed_data[col],
                    sample_data[col],
                    check_names=False
                )
            else:
                # Numerical features should be very close (allowing for small numerical differences)
                np.testing.assert_allclose(
                    transformed_data[col].values,
                    sample_data[col].values,
                    rtol=1e-10
                )
    
    def test_repair_level_full(self, sample_data):
        """Test transform with repair level 1.0 (full repair)."""
        transformer = BiasMitigationTransformer(
            sensitive_features=['race'],
            repair_level=1.0
        )
        transformer.fit(sample_data)
        
        transformed_data = transformer.transform(sample_data)
        
        # Check that group means are closer to overall means
        non_sensitive_features = transformer.non_sensitive_features_
        
        for feature in non_sensitive_features:
            # Skip categorical features - mean doesn't make sense for them
            if feature in transformer.categorical_features_:
                continue
                
            overall_mean = sample_data[feature].mean()
            
            # Calculate group means in transformed data
            white_mean = transformed_data[transformed_data['race'] == 'White'][feature].mean()
            black_mean = transformed_data[transformed_data['race'] == 'Black'][feature].mean()
            
            # With full repair, group means should be closer to overall mean
            # than in original data
            original_white_mean = sample_data[sample_data['race'] == 'White'][feature].mean()
            original_black_mean = sample_data[sample_data['race'] == 'Black'][feature].mean()
            
            # Differences should be reduced
            original_diff = abs(original_white_mean - overall_mean) + abs(original_black_mean - overall_mean)
            transformed_diff = abs(white_mean - overall_mean) + abs(black_mean - overall_mean)
            
            assert transformed_diff <= original_diff + 1e-6  # Allow small numerical error
    
    def test_get_mitigation_details(self, sample_data):
        """Test get_mitigation_details method."""
        transformer = BiasMitigationTransformer(
            sensitive_features=['race'],
            repair_level=0.8,
            method='mean_matching'
        )
        transformer.fit(sample_data)
        
        details = transformer.get_mitigation_details()
        
        # Check required fields
        assert 'method' in details
        assert 'repair_level' in details
        assert 'sensitive_features' in details
        assert 'non_sensitive_features' in details
        assert 'group_statistics' in details
        assert 'overall_means' in details
        
        # Check values
        assert details['method'] == 'mean_matching'
        assert details['repair_level'] == 0.8
        assert details['sensitive_features'] == ['race']
    
    def test_get_mitigation_details_not_fitted(self):
        """Test get_mitigation_details when not fitted."""
        transformer = BiasMitigationTransformer(sensitive_features=['race'])
        
        with pytest.raises(ValueError, match="Transformer must be fitted"):
            transformer.get_mitigation_details()
    
    def test_invalid_method(self):
        """Test initialization with invalid method."""
        transformer = BiasMitigationTransformer(
            sensitive_features=['race'],
            method='invalid_method'
        )
        transformer.fit(pd.DataFrame({'feature': [1, 2], 'race': ['A', 'B']}))
        
        with pytest.raises(ValueError, match="Unknown method"):
            transformer.transform(pd.DataFrame({'feature': [1, 2], 'race': ['A', 'B']}))
    
    def test_small_group_handling(self):
        """Test handling of small groups with insufficient data."""
        # Create data with very small group
        data = pd.DataFrame({
            'feature': [1.0, 2.0, 3.0],
            'sensitive': ['A', 'A', 'B']  # Group B has only 1 sample
        })
        
        transformer = BiasMitigationTransformer(sensitive_features=['sensitive'])
        
        # Should handle small groups without crashing
        transformer.fit(data)
        transformed = transformer.transform(data)
        
        assert transformed.shape == data.shape
    
    def test_single_group_handling(self):
        """Test handling when only one group is present."""
        data = pd.DataFrame({
            'feature': [1.0, 2.0, 3.0],
            'sensitive': ['A', 'A', 'A']  # Only one group
        })
        
        transformer = BiasMitigationTransformer(sensitive_features=['sensitive'])
        
        # Should handle single group without crashing
        transformer.fit(data)
        transformed = transformer.transform(data)
        
        assert transformed.shape == data.shape
        
        # With single group, no bias mitigation should occur
        pd.testing.assert_frame_equal(transformed, data)