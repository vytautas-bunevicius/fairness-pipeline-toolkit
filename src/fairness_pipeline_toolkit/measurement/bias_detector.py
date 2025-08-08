"""Bias detection and reporting functionality."""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from .fairness_metrics import FairnessMetrics


class BiasDetector:
    """Detect and report bias in datasets and model predictions."""
    
    def __init__(self, threshold: float = 0.1):
        """Initialize bias detector with fairness threshold."""
        self.threshold = threshold
        self.metrics_calculator = FairnessMetrics()
    
    def audit_dataset(self, data: pd.DataFrame, 
                     sensitive_column: str, 
                     target_column: Optional[str] = None) -> Dict[str, Any]:
        """Audit dataset for potential bias in data distribution."""
        report = {
            'dataset_shape': data.shape,
            'sensitive_feature_distribution': data[sensitive_column].value_counts().to_dict(),
            'missing_values': data.isnull().sum().to_dict()
        }
        
        if target_column:
            target_by_sensitive = data.groupby(sensitive_column)[target_column].mean()
            report['target_rate_by_group'] = target_by_sensitive.to_dict()
            report['target_rate_difference'] = float(target_by_sensitive.max() - target_by_sensitive.min())
        
        return report
    
    def audit_predictions(self, y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         sensitive_features: np.ndarray) -> Dict[str, Any]:
        """Audit model predictions for fairness violations."""
        metrics = self.metrics_calculator.calculate_all_metrics(
            y_true, y_pred, sensitive_features
        )
        
        fairness_violations = {
            'demographic_parity_violation': metrics['demographic_parity_difference'] > self.threshold,
            'equalized_odds_violation': metrics['equalized_odds_difference'] > self.threshold
        }
        
        report = {
            'metrics': metrics,
            'fairness_violations': fairness_violations,
            'overall_fairness_score': 1.0 - max(
                metrics['demographic_parity_difference'], 
                metrics['equalized_odds_difference']
            ),
            'threshold': self.threshold
        }
        
        return report
    
    def print_report(self, report: Dict[str, Any], report_type: str = "audit"):
        """Print formatted bias audit report."""
        print(f"\n=== {report_type.upper()} REPORT ===")
        
        if 'dataset_shape' in report:
            print(f"Dataset Shape: {report['dataset_shape']}")
            print(f"Sensitive Feature Distribution: {report['sensitive_feature_distribution']}")
            if 'target_rate_difference' in report:
                print(f"Target Rate Difference: {report['target_rate_difference']:.4f}")
        
        if 'metrics' in report:
            print("Performance Metrics:")
            for metric, value in report['metrics'].items():
                if 'difference' not in metric:
                    print(f"  {metric.title()}: {value:.4f}")
            
            print("Fairness Metrics:")
            for metric, value in report['metrics'].items():
                if 'difference' in metric:
                    print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
            
            print(f"Overall Fairness Score: {report['overall_fairness_score']:.4f}")
            
            if any(report['fairness_violations'].values()):
                print("FAIRNESS VIOLATIONS DETECTED:")
                for violation, detected in report['fairness_violations'].items():
                    if detected:
                        print(f"  - {violation.replace('_', ' ').title()}")
            else:
                print("No significant fairness violations detected")
        
        print("=" * 50)