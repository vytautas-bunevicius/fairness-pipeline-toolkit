"""Bias detection and reporting functionality."""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
from rich.console import Console
from rich.table import Table
from rich import box
from .fairness_metrics import FairnessMetrics


class BiasDetector:
    """Detect and report bias in datasets and model predictions."""

    def __init__(self, threshold: float = 0.1):
        """Initialize bias detector with fairness threshold."""
        self.threshold = threshold
        self.metrics_calculator = FairnessMetrics()
        self.logger = logging.getLogger("fairness_pipeline.bias_detector")
        self.console = Console(force_terminal=True, width=100)

    @staticmethod
    def audit_dataset(
        data: pd.DataFrame, sensitive_column: str, target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """Audit dataset for potential bias in data distribution."""
        report = {
            "dataset_shape": data.shape,
            "sensitive_feature_distribution": data[sensitive_column]
            .value_counts()
            .to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
        }

        if target_column:
            target_by_sensitive = data.groupby(sensitive_column)[target_column].mean()
            report["target_rate_by_group"] = target_by_sensitive.to_dict()
            report["target_rate_difference"] = float(
                target_by_sensitive.max() - target_by_sensitive.min()
            )

        return report

    def audit_predictions(
        self, y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: np.ndarray
    ) -> Dict[str, Any]:
        """Audit model predictions for fairness violations."""
        metrics = self.metrics_calculator.calculate_all_metrics(
            y_true, y_pred, sensitive_features
        )

        fairness_violations = {
            "demographic_parity_violation": metrics["demographic_parity_difference"]
            > self.threshold,
            "equalized_odds_violation": metrics["equalized_odds_difference"]
            > self.threshold,
        }

        report = {
            "metrics": metrics,
            "fairness_violations": fairness_violations,
            "overall_fairness_score": 1.0
            - max(
                metrics["demographic_parity_difference"],
                metrics["equalized_odds_difference"],
            ),
            "threshold": self.threshold,
        }

        return report

    def print_report(self, report: Dict[str, Any], report_type: str = "audit"):
        """Print and log formatted bias audit report using Rich tables."""
        self.console.print(
            f"\n[bold blue]{report_type.upper()} REPORT[/bold blue]", style="bold blue"
        )

        self.logger.info(
            f"{report_type.title()} Report",
            extra={"component": "bias_detector", "report_type": report_type},
        )

        if "dataset_shape" in report:
            dataset_table = Table(
                title="Dataset Information",
                box=box.SIMPLE,
                show_header=True,
                header_style="bold blue",
            )
            dataset_table.add_column(
                "Attribute", style="cyan", no_wrap=True, min_width=15, max_width=25
            )
            dataset_table.add_column(
                "Value", style="magenta", min_width=15, max_width=30
            )

            dataset_table.add_row("Dataset Shape", str(report["dataset_shape"]))

            if "sensitive_feature_distribution" in report:
                dist = report["sensitive_feature_distribution"]
                if isinstance(dist, dict):
                    for group, count in dist.items():
                        dataset_table.add_row(f"Group: {group}", str(count))
                else:
                    dataset_table.add_row("Sensitive Feature Distribution", str(dist))

            if "target_rate_difference" in report:
                dataset_table.add_row(
                    "Target Rate Difference", f"{report['target_rate_difference']:.4f}"
                )

            self.console.print(dataset_table)

            self.logger.info(
                f"Dataset Shape: {report['dataset_shape']}",
                extra={
                    "component": "bias_detector",
                    "dataset_shape": report["dataset_shape"],
                    "sensitive_feature_distribution": report.get(
                        "sensitive_feature_distribution", {}
                    ),
                },
            )

        if "metrics" in report:
            performance_metrics = {
                metric_name: metric_value
                for metric_name, metric_value in report["metrics"].items()
                if "difference" not in metric_name
            }
            if performance_metrics:
                performance_table = Table(
                    title="Performance Metrics",
                    box=box.SIMPLE,
                    show_header=True,
                    header_style="bold blue",
                )
                performance_table.add_column(
                    "Metric", style="cyan", no_wrap=True, min_width=15, max_width=20
                )
                performance_table.add_column(
                    "Value", style="green", justify="right", min_width=8, max_width=15
                )

                for metric_name, metric_value in performance_metrics.items():
                    performance_table.add_row(
                        metric_name.title(), f"{metric_value:.4f}"
                    )
                    self.logger.info(
                        f"{metric_name.title()}: {metric_value:.4f}",
                        extra={
                            "component": "bias_detector",
                            "metric_type": "performance",
                            "metric_name": metric_name,
                            "metric_value": metric_value,
                        },
                    )

                self.console.print(performance_table)

            fairness_metrics = {
                metric_name: metric_value
                for metric_name, metric_value in report["metrics"].items()
                if "difference" in metric_name
            }
            if fairness_metrics:
                fairness_table = Table(
                    title="Fairness Metrics",
                    box=box.SIMPLE,
                    show_header=True,
                    header_style="bold blue",
                )
                fairness_table.add_column(
                    "Metric", style="cyan", no_wrap=False, min_width=20, max_width=30
                )
                fairness_table.add_column(
                    "Value", style="magenta", justify="right", min_width=8, max_width=12
                )
                fairness_table.add_column(
                    "Status", style="bold", justify="center", min_width=12, max_width=18
                )
                fairness_table.add_column(
                    "Threshold", style="dim", justify="right", min_width=8, max_width=15
                )

                for metric, value in fairness_metrics.items():
                    status = "✅ OK" if abs(value) <= self.threshold else "❌ VIOLATION"
                    status_style = "green" if abs(value) <= self.threshold else "red"

                    fairness_table.add_row(
                        metric.replace("_", " ").title(),
                        f"{value:.4f}",
                        f"[{status_style}]{status}[/{status_style}]",
                        f"≤ {self.threshold}",
                    )

                    self.logger.info(
                        f"{metric.replace('_', ' ').title()}: {value:.4f} ({status})",
                        extra={
                            "component": "bias_detector",
                            "metric_type": "fairness",
                            "metric_name": metric,
                            "metric_value": value,
                            "fairness_status": "OK"
                            if abs(value) <= self.threshold
                            else "VIOLATION",
                            "threshold": self.threshold,
                        },
                    )

                self.console.print(fairness_table)

            if "overall_fairness_score" in report:
                score = report["overall_fairness_score"]
                score_table = Table(
                    title="Overall Assessment",
                    box=box.SIMPLE,
                    show_header=True,
                    header_style="bold blue",
                )
                score_table.add_column(
                    "Metric", style="cyan", no_wrap=True, min_width=18, max_width=25
                )
                score_table.add_column(
                    "Score",
                    style="bold green",
                    justify="right",
                    min_width=10,
                    max_width=20,
                )
                score_table.add_row("Overall Fairness Score", f"{score:.4f}")

                self.console.print(score_table)

                self.logger.info(
                    f"Overall Fairness Score: {score:.4f}",
                    extra={
                        "component": "bias_detector",
                        "overall_fairness_score": score,
                    },
                )

            if "fairness_violations" in report:
                violations = [
                    violation
                    for violation, detected in report["fairness_violations"].items()
                    if detected
                ]
                if violations:
                    self.console.print(
                        f"[red]⚠️  {len(violations)} fairness violations detected[/red]"
                    )
                    self.logger.warning(
                        f"Fairness violations detected: {', '.join(violations)}",
                        extra={
                            "component": "bias_detector",
                            "violations": violations,
                            "violation_count": len(violations),
                        },
                    )
                else:
                    self.console.print(
                        "[green]✅ No fairness violations detected[/green]"
                    )
                    self.logger.info(
                        "No significant fairness violations detected",
                        extra={
                            "component": "bias_detector",
                            "violations": [],
                            "violation_count": 0,
                        },
                    )

        self.console.print("")
