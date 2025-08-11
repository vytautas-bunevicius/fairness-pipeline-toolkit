"""Bias detection and reporting functionality."""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
from .fairness_metrics import FairnessMetrics


class BiasDetector:
    """Detect and report bias in datasets and model predictions."""
    
    def __init__(self, threshold: float = 0.1):
        """Initialize bias detector with fairness threshold."""
        self.threshold = threshold
        self.metrics_calculator = FairnessMetrics()
        self.logger = logging.getLogger('fairness_pipeline.bias_detector')
    
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
        """Log formatted bias audit report using structured logging."""
        self.logger.info(f"{report_type.title()} Report", extra={'component': 'bias_detector', 'report_type': report_type})
        
        if 'dataset_shape' in report:
            self.logger.info(f"Dataset Shape: {report['dataset_shape']}", extra={
                'component': 'bias_detector',
                'dataset_shape': report['dataset_shape'],
                'sensitive_feature_distribution': report.get('sensitive_feature_distribution', {})
            })
            if 'target_rate_difference' in report:
                self.logger.info(f"Target Rate Difference: {report['target_rate_difference']:.4f}", extra={
                    'component': 'bias_detector',
                    'target_rate_difference': report['target_rate_difference']
                })
        
        if 'metrics' in report:
            # Log performance metrics
            perf_metrics = {k: v for k, v in report['metrics'].items() if 'difference' not in k}
            if perf_metrics:
                for metric, value in perf_metrics.items():
                    self.logger.info(f"{metric.title()}: {value:.4f}", extra={
                        'component': 'bias_detector',
                        'metric_type': 'performance',
                        'metric_name': metric,
                        'metric_value': value
                    })
            
            # Log fairness metrics
            fairness_metrics = {k: v for k, v in report['metrics'].items() if 'difference' in k}
            if fairness_metrics:
                for metric, value in fairness_metrics.items():
                    status = "OK" if abs(value) <= self.threshold else "VIOLATION"
                    self.logger.info(f"{metric.replace('_', ' ').title()}: {value:.4f} ({status})", extra={
                        'component': 'bias_detector',
                        'metric_type': 'fairness',
                        'metric_name': metric,
                        'metric_value': value,
                        'fairness_status': status,
                        'threshold': self.threshold
                    })
            
            if 'overall_fairness_score' in report:
                score = report['overall_fairness_score']
                self.logger.info(f"Overall Fairness Score: {score:.4f}", extra={
                    'component': 'bias_detector',
                    'overall_fairness_score': score
                })
            
            if 'fairness_violations' in report:
                if any(report['fairness_violations'].values()):
                    violated_metrics = [violation for violation, detected in report['fairness_violations'].items() if detected]
                    self.logger.warning(f"Fairness violations detected: {', '.join(violated_metrics)}", extra={
                        'component': 'bias_detector',
                        'violations': violated_metrics,
                        'violation_count': len(violated_metrics)
                    })
                else:
                    self.logger.info("No significant fairness violations detected", extra={
                        'component': 'bias_detector',
                        'violations': [],
                        'violation_count': 0
                    })