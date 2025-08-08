"""Pipeline executor for fairness-aware ML workflows."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import tempfile

from .config import ConfigParser
from .measurement.bias_detector import BiasDetector
from .pipeline.bias_mitigation_transformer import BiasMitigationTransformer
from .training.fairness_constrained_classifier import FairnessConstrainedClassifier


class PipelineExecutor:
    """Executes fairness-aware ML pipelines with bias detection, mitigation, and fair training."""
    
    def __init__(self, config: Dict[str, Any], verbose: bool = False):
        """Initialize pipeline executor."""
        self.config = config
        self.verbose = verbose
        self.bias_detector = BiasDetector(threshold=config['evaluation']['fairness_threshold'])
        
        # Initialize components
        self.transformer = None
        self.model = None
        self.baseline_report = None
        self.final_report = None
        
    def execute_pipeline(self) -> Dict[str, Any]:
        """Execute the complete fairness pipeline."""
        if self.verbose:
            print("\nStarting fairness pipeline execution...")
        
        # Setup MLflow
        self._setup_mlflow()
        
        # End any existing active run
        if mlflow.active_run():
            mlflow.end_run()
        
        with mlflow.start_run(run_name=self.config['mlflow'].get('run_name')):
            # Log configuration
            if self.config['mlflow'].get('log_config', True):
                self._log_config()
            
            # Step 1: Load and audit baseline data
            X_train, X_test, y_train, y_test = self._load_and_split_data()
            self._measure_baseline_bias(X_train, y_train)
            
            # Step 2: Transform data and train model
            X_train_transformed, X_test_transformed = self._mitigate_bias_and_train_model(
                X_train, X_test, y_train
            )
            
            # Step 3: Final validation
            results = self._evaluate_final_fairness(
                X_test_transformed, y_test, X_train_transformed, y_train
            )
            
            # Log results to MLflow
            self._log_results(results)
            
            if self.verbose:
                print("\nPipeline execution completed successfully!")
            
            return results
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow experiment and run."""
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
        # Set tags
        if 'tags' in self.config['mlflow']:
            for key, value in self.config['mlflow']['tags'].items():
                mlflow.set_tag(key, value)
    
    def _log_config(self) -> None:
        """Log configuration to MLflow."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            import yaml
            yaml.dump(self.config, f)
            mlflow.log_artifact(f.name, "config")
    
    def _load_and_split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load data and create train/test splits."""
        if self.verbose:
            print(f"Loading data from: {self.config['data']['input_path']}")
        
        # For demo purposes, create synthetic data if file doesn't exist
        data_path = Path(self.config['data']['input_path'])
        if not data_path.exists():
            if self.verbose:
                print("Data file not found. Generating synthetic data for demo...")
            data = self._generate_synthetic_data()
        else:
            data = pd.read_csv(data_path)
        
        # Separate features and target
        X = data.drop(columns=[self.config['data']['target_column']])
        y = data[self.config['data']['target_column']]
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['data'].get('test_size', 0.2),
            random_state=self.config['data'].get('random_state', 42),
            stratify=y
        )
        
        if self.verbose:
            print(f"Data loaded: {len(X_train)} training, {len(X_test)} test samples")
        
        return X_train, X_test, y_train, y_test
    
    def _generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic data for demonstration."""
        np.random.seed(self.config['data'].get('random_state', 42))
        
        # Generate synthetic features
        data = pd.DataFrame({
            'age': np.random.normal(35, 10, n_samples).clip(18, 80),
            'income': np.random.lognormal(10, 0.5, n_samples),
            'education_years': np.random.normal(12, 3, n_samples).clip(8, 20),
            'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
            'sex': np.random.choice(['Male', 'Female'], n_samples, p=[0.5, 0.5])
        })
        
        # Generate biased target variable
        bias_factor = (data['race'] == 'White').astype(int) * 0.3 + (data['sex'] == 'Male').astype(int) * 0.2
        logit = -2 + 0.1 * data['age'] + 0.00002 * data['income'] + 0.1 * data['education_years'] + bias_factor
        prob = 1 / (1 + np.exp(-logit))
        data['target'] = np.random.binomial(1, prob, n_samples)
        
        return data
    
    def _measure_baseline_bias(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Measure baseline bias in raw data before any intervention."""
        if self.verbose:
            print("\nSTEP 1: Baseline Measurement")
        
        # Audit dataset
        dataset_report = self.bias_detector.audit_dataset(
            pd.concat([X, y], axis=1), 
            sensitive_column=self.config['data']['sensitive_features'][0],
            target_column=self.config['data']['target_column']
        )
        
        # For predictions audit, we need a simple model prediction
        simple_model = LogisticRegression(random_state=self.config['data'].get('random_state', 42))
        X_numeric = self._prepare_features_for_modeling(X)
        simple_model.fit(X_numeric, y)
        y_pred_baseline = simple_model.predict(X_numeric)
        
        # Audit baseline predictions
        sensitive_features = X[self.config['data']['sensitive_features'][0]]
        prediction_report = self.bias_detector.audit_predictions(
            y.values, y_pred_baseline, sensitive_features.values
        )
        
        self.baseline_report = {
            'dataset_audit': dataset_report,
            'prediction_audit': prediction_report
        }
        
        # Print baseline report
        self.bias_detector.print_report(dataset_report, "BASELINE DATASET")
        self.bias_detector.print_report(prediction_report, "BASELINE PREDICTION")
    
    def _mitigate_bias_and_train_model(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                  y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply bias mitigation transformation and train fairness-constrained model."""
        if self.verbose:
            print("\nSTEP 2: Data Transformation and Model Training")
        
        # Initialize and fit transformer
        transformer_config = self.config['preprocessing']['transformer']
        if transformer_config['name'] == 'BiasMitigationTransformer':
            params = transformer_config.get('parameters', {})
            self.transformer = BiasMitigationTransformer(
                sensitive_features=self.config['data']['sensitive_features'],
                **params
            )
            
            if self.verbose:
                print(f"Applying {transformer_config['name']} transformation...")
            
            self.transformer.fit(X_train)
            X_train_transformed = self.transformer.transform(X_train)
            X_test_transformed = self.transformer.transform(X_test)
        else:
            # No transformation
            X_train_transformed = X_train.copy()
            X_test_transformed = X_test.copy()
        
        # Initialize and train fair model
        training_config = self.config['training']['method']
        if training_config['name'] == 'FairnessConstrainedClassifier':
            params = training_config.get('parameters', {})
            
            # Prepare base estimator
            base_estimator = None
            if params.get('base_estimator') == 'LogisticRegression':
                base_estimator = LogisticRegression(random_state=self.config['data'].get('random_state', 42))
            
            self.model = FairnessConstrainedClassifier(
                base_estimator=base_estimator,
                sensitive_features=self.config['data']['sensitive_features'],
                constraint=params.get('constraint', 'demographic_parity'),
                random_state=self.config['data'].get('random_state', 42)
            )
            
            if self.verbose:
                print(f"Training {training_config['name']} model...")
            
            # Prepare features for modeling
            X_model = self._prepare_features_for_modeling(X_train_transformed)
            sensitive_features = X_train_transformed[self.config['data']['sensitive_features']]
            
            self.model.fit(X_model, y_train, sensitive_features)
        
        if self.verbose:
            print("Transformation and training completed")
        
        return X_train_transformed, X_test_transformed
    
    def _evaluate_final_fairness(self, X_test: pd.DataFrame, y_test: pd.Series,
                               X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Evaluate final model fairness and compare with baseline metrics."""
        if self.verbose:
            print("\nSTEP 3: Final Validation")
        
        # Make predictions with trained model
        X_test_numeric = self._prepare_features_for_modeling(X_test)
        sensitive_features_test = X_test[self.config['data']['sensitive_features']]
        
        y_pred_final = self.model.predict(X_test_numeric, sensitive_features_test)
        
        # Audit final predictions
        sensitive_values = X_test[self.config['data']['sensitive_features'][0]].values
        final_prediction_report = self.bias_detector.audit_predictions(
            y_test.values, y_pred_final, sensitive_values
        )
        
        self.final_report = final_prediction_report
        
        # Print final report
        self.bias_detector.print_report(final_prediction_report, "FINAL PREDICTION")
        
        # Compare with baseline
        if self.baseline_report:
            baseline_metrics = self.baseline_report['prediction_audit']['metrics']
            final_metrics = final_prediction_report['metrics']
            
            print("\nIMPROVEMENT COMPARISON:")
            print(f"Accuracy: {baseline_metrics['accuracy']:.4f} → {final_metrics['accuracy']:.4f}")
            print(f"Primary Fairness Metric ({self.config['evaluation']['primary_metric']}):")
            print(f"  Baseline: {baseline_metrics[self.config['evaluation']['primary_metric']]:.4f}")
            print(f"  Final: {final_metrics[self.config['evaluation']['primary_metric']]:.4f}")
            
            improvement = baseline_metrics[self.config['evaluation']['primary_metric']] - final_metrics[self.config['evaluation']['primary_metric']]
            if improvement > 0:
                print(f"  Improvement: {improvement:.4f}")
            else:
                print(f"  Change: {improvement:.4f}")
        
        return {
            'baseline_report': self.baseline_report,
            'final_report': self.final_report,
            'model': self.model,
            'transformer': self.transformer
        }
    
    def _prepare_features_for_modeling(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model training (handle categorical variables)."""
        X_numeric = X.copy()
        
        # Simple encoding for categorical variables
        for col in X_numeric.columns:
            if X_numeric[col].dtype == 'object':
                # Label encoding for simplicity
                unique_values = X_numeric[col].unique()
                value_map = {val: idx for idx, val in enumerate(unique_values)}
                X_numeric[col] = X_numeric[col].map(value_map)
        
        return X_numeric
    
    def _log_results(self, results: Dict[str, Any]) -> None:
        """Log results to MLflow."""
        if self.verbose:
            print("\nLogging results to MLflow...")
        
        # Log baseline metrics
        if results['baseline_report']:
            baseline_metrics = results['baseline_report']['prediction_audit']['metrics']
            for metric, value in baseline_metrics.items():
                mlflow.log_metric(f"baseline_{metric}", value)
        
        # Log final metrics
        final_metrics = results['final_report']['metrics']
        for metric, value in final_metrics.items():
            mlflow.log_metric(f"final_{metric}", value)
        
        # Log primary fairness metric
        primary_metric = self.config['evaluation']['primary_metric']
        mlflow.log_metric("primary_fairness_metric", final_metrics[primary_metric])
        
        # Log model if configured
        if self.config['mlflow'].get('log_model', True) and results['model']:
            mlflow.sklearn.log_model(
                results['model'], 
                "fair_model",
                registered_model_name=f"{self.config['mlflow']['experiment_name']}_model"
            )