"""
Central orchestrator for end-to-end fairness-aware machine learning workflows.

This module coordinates the three-phase fairness pipeline: baseline measurement to identify
bias, bias mitigation to reduce disparities, and fair training for more equitable outcomes.
The executor manages the interactions between components and provides detailed logging
through structured logging and MLflow integration.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.exceptions import MlflowException
import tempfile
import warnings
from datetime import datetime

os.environ.setdefault("MLFLOW_SUPPRESS_ENVIRONMENT_WARNINGS", "1")
warnings.filterwarnings("ignore", message=".*pip.*", module="mlflow.*")
warnings.filterwarnings("ignore", message=".*artifact_path.*", module="mlflow.*")
from rich.console import Console
from rich.table import Table
from rich import box

from .measurement.bias_detector import BiasDetector
from .pipeline.bias_mitigation_transformer import BiasMitigationTransformer
from .training.fairness_constrained_classifier import FairnessConstrainedClassifier
from .config import get_pipeline_logger, setup_logging


class PipelineExecutor:
    """
    Orchestrates the complete fairness-aware machine learning workflow.

    The executor implements a systematic approach to fairness: first measuring baseline
    bias, then applying data-level corrections to reduce group disparities, and finally
    training models with fairness constraints for more equitable outcomes. This approach
    addresses bias at multiple levels while maintaining model performance.
    """

    def __init__(
        self, config: Dict[str, Any], verbose: bool = False, enable_logging: bool = True
    ):
        """
        Initialize with configuration and observability settings.

        The executor requires comprehensive configuration to coordinate between bias
        detection, mitigation, and fair training components. Logging is enabled by
        default to provide full audit trails of the fairness pipeline execution.
        """
        self.config = config
        self.verbose = verbose
        self.bias_detector = BiasDetector(
            threshold=config["evaluation"]["fairness_threshold"]
        )
        self.experiment_name = None

        self.transformer = None
        self.model = None
        self.baseline_report = None
        self.final_report = None
        self.feature_scaler = StandardScaler()

        if enable_logging:
            log_level = "DEBUG" if verbose else "INFO"
            setup_logging(level=log_level, structured=True, console_output=verbose)

        self.logger = get_pipeline_logger("executor")
        self.console = Console(force_terminal=True, width=120)

    def execute_pipeline(self) -> Dict[str, Any]:
        """
        Execute the fairness pipeline with detailed logging.

        The pipeline follows a systematic approach: baseline measurement identifies
        bias, data processing reduces disparities through transformation, and fair
        training promotes equitable model behavior. Each phase builds on the
        previous one while maintaining detailed logging for audit and debugging.
        """
        self.logger.log_stage_start(
            "pipeline_execution",
            {
                "experiment_name": self.experiment_name
                or self.config["mlflow"]["experiment_name"],
                "primary_metric": self.config["evaluation"]["primary_metric"],
            },
        )

        self.logger.start_timer("full_pipeline")

        try:
            self._setup_mlflow()

            if mlflow.active_run():
                mlflow.end_run()

            with mlflow.start_run(run_name=self.config["mlflow"].get("run_name")):
                if self.config["mlflow"].get("log_config", True):
                    self._log_config()

                self.logger.start_timer("data_loading")
                X_train, X_test, y_train, y_test = self._load_and_split_data()
                self.logger.end_timer("data_loading")

                self.logger.start_timer("baseline_measurement")
                self._measure_baseline_bias(X_train, y_train)
                self.logger.end_timer("baseline_measurement")

                self.logger.start_timer("bias_mitigation_training")
                X_train_transformed, X_test_transformed = (
                    self._mitigate_bias_and_train_model(X_train, X_test, y_train)
                )
                self.logger.end_timer("bias_mitigation_training")

                self.logger.start_timer("final_evaluation")
                results = self._evaluate_final_fairness(
                    X_test_transformed, y_test, X_train_transformed, y_train
                )
                self.logger.end_timer("final_evaluation")

                self.logger.start_timer("mlflow_logging")
                self._log_results(results)
                self.logger.end_timer("mlflow_logging")

                pipeline_duration = self.logger.end_timer("full_pipeline")

                self.logger.log_stage_complete(
                    "pipeline_execution",
                    {
                        "total_duration_ms": pipeline_duration,
                        "train_samples": len(X_train),
                        "test_samples": len(X_test),
                    },
                )

                results["experiment_name"] = self.experiment_name
                active_run = mlflow.active_run()
                if active_run:
                    results["run_id"] = active_run.info.run_id

                return results

        except Exception as e:
            self.logger.log_error(
                "Pipeline execution failed", e, {"config": self.config}
            )
            raise

    def _setup_mlflow(self) -> None:
        """Setup MLflow experiment and run."""
        project_root = Path(__file__).parent.parent.parent
        mlruns_path = project_root / "mlruns"
        mlflow.set_tracking_uri(f"file://{mlruns_path.absolute()}")

        base_name = self.config["mlflow"]["experiment_name"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{base_name}_{timestamp}"

        try:
            mlflow.set_experiment(experiment_name)
            self.experiment_name = experiment_name
        except MlflowException as e:
            if "deleted experiment" in str(e).lower():
                import random

                experiment_name = (
                    f"{base_name}_{timestamp}_{random.randint(1000, 9999)}"
                )
                mlflow.set_experiment(experiment_name)
                self.experiment_name = experiment_name
            else:
                raise e

        if "tags" in self.config["mlflow"]:
            for key, value in self.config["mlflow"]["tags"].items():
                mlflow.set_tag(key, value)

    def _log_config(self) -> None:
        """Log configuration to MLflow."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            import yaml

            yaml.dump(self.config, f)
            mlflow.log_artifact(f.name, "config")

    def _load_and_split_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load data and create train/test splits."""
        data_path = Path(self.config["data"]["input_path"])

        if not data_path.exists():
            self.logger.log_warning(
                "Data file not found, generating synthetic data",
                {"requested_path": str(data_path)},
            )
            data = self._generate_synthetic_data()
        else:
            self.logger.log_stage_start("data_loading", {"data_path": str(data_path)})
            data = pd.read_csv(data_path)

        self.logger.log_data_info(data.shape, self.config["data"]["sensitive_features"])

        X = data.drop(columns=[self.config["data"]["target_column"]])
        y = data[self.config["data"]["target_column"]]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config["data"].get("test_size", 0.2),
            random_state=self.config["data"].get("random_state", 42),
            stratify=y,
        )

        self.logger.log_stage_complete(
            "data_split",
            {
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "test_size": self.config["data"].get("test_size", 0.2),
            },
        )

        return X_train, X_test, y_train, y_test

    def _generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Create synthetic data with intentional bias patterns.

        The generated dataset includes missing values, outliers, and correlated features
        to simulate real-world data, with measurable bias patterns for testing the
        pipeline's effectiveness.
        """
        np.random.seed(self.config["data"].get("random_state", 42))

        data = pd.DataFrame(
            {
                "age": np.random.normal(35, 12, n_samples).clip(18, 80),
                "income": np.random.lognormal(10, 0.8, n_samples),
                "education_years": np.random.normal(12, 4, n_samples).clip(6, 20),
                "race": np.random.choice(
                    ["White", "Black", "Hispanic", "Asian"],
                    n_samples,
                    p=[0.6, 0.2, 0.15, 0.05],
                ),
                "sex": np.random.choice(["Male", "Female"], n_samples, p=[0.52, 0.48]),
            }
        )

        race_income_bias = {"White": 1.2, "Asian": 1.15, "Hispanic": 0.85, "Black": 0.8}
        sex_income_bias = {"Male": 1.1, "Female": 0.9}

        for i in range(len(data)):
            race_multiplier = race_income_bias[data.loc[i, "race"]]
            sex_multiplier = sex_income_bias[data.loc[i, "sex"]]
            data.loc[i, "income"] *= race_multiplier * sex_multiplier

        for col in ["age", "income", "education_years"]:
            missing_mask = np.random.random(n_samples) < 0.07
            data.loc[missing_mask, col] = np.nan

        outlier_mask = np.random.random(n_samples) < 0.02
        data.loc[outlier_mask, "income"] *= np.random.uniform(3, 8, sum(outlier_mask))

        data = data.ffill().bfill()

        bias_factor = (
            (data["race"] == "White").astype(int) * 0.4
            + (data["race"] == "Asian").astype(int) * 0.3
            + (data["sex"] == "Male").astype(int) * 0.25
            + ((data["race"] == "White") & (data["sex"] == "Male")).astype(int) * 0.2
        )

        logit = (
            -3.2
            + 0.08 * (data["age"] - 35)
            + 0.00003 * (data["income"] - data["income"].mean())
            + 0.12 * (data["education_years"] - 12)
            + bias_factor
            + np.random.normal(0, 0.1, n_samples)
        )

        prob = 1 / (1 + np.exp(-logit))
        data["target"] = np.random.binomial(1, prob, n_samples)

        return data

    def _measure_baseline_bias(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Measure baseline bias in raw data before any intervention."""
        self.logger.log_stage_start("baseline_measurement")

        dataset_report = self.bias_detector.audit_dataset(
            pd.concat([X, y], axis=1),
            sensitive_column=self.config["data"]["sensitive_features"][0],
            target_column=self.config["data"]["target_column"],
        )

        simple_model = LogisticRegression(
            random_state=self.config["data"].get("random_state", 42),
            max_iter=1000,
            solver="lbfgs",
        )

        baseline_scaler = StandardScaler()
        X_baseline = X.drop(columns=self.config["data"]["sensitive_features"])
        X_numeric = self._prepare_features_for_baseline(X_baseline, baseline_scaler)
        simple_model.fit(X_numeric, y)
        y_pred_baseline = simple_model.predict(X_numeric)

        sensitive_features = X[self.config["data"]["sensitive_features"][0]]
        prediction_report = self.bias_detector.audit_predictions(
            y.values, y_pred_baseline, sensitive_features.values
        )

        self.baseline_report = {
            "dataset_audit": dataset_report,
            "prediction_audit": prediction_report,
        }

        baseline_metrics = prediction_report["metrics"]
        self.logger.log_fairness_metrics(baseline_metrics, "baseline")

        violations = prediction_report.get("fairness_violations", {})
        violation_count = sum(
            1 for violation_detected in violations.values() if violation_detected
        )

        self.logger.log_stage_complete(
            "baseline_measurement",
            {
                "fairness_violations": violation_count,
                "overall_fairness_score": prediction_report.get(
                    "overall_fairness_score", 0
                ),
            },
        )

        if self.verbose:
            self.bias_detector.print_report(dataset_report, "BASELINE DATASET")
            self.bias_detector.print_report(prediction_report, "BASELINE PREDICTION")

    def _mitigate_bias_and_train_model(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply bias mitigation transformation and train fairness-constrained model."""
        self.logger.log_stage_start("bias_mitigation")

        transformer_config = self.config["preprocessing"]["transformer"]
        if transformer_config["name"] == "BiasMitigationTransformer":
            params = transformer_config.get("parameters", {})
            self.transformer = BiasMitigationTransformer(
                sensitive_features=self.config["data"]["sensitive_features"], **params
            )

            self.logger.log_stage_start(
                "transformer_fitting",
                {
                    "transformer_type": transformer_config["name"],
                    "repair_level": params.get("repair_level", 1.0),
                    "method": params.get("method", "mean_matching"),
                },
            )

            self.transformer.fit(X_train)
            X_train_transformed = self.transformer.transform(X_train)
            X_test_transformed = self.transformer.transform(X_test)

            self.logger.log_stage_complete("transformer_fitting")
        else:
            self.logger.log_warning("No bias mitigation transformer specified")
            X_train_transformed = X_train.copy()
            X_test_transformed = X_test.copy()

        training_config = self.config["training"]["method"]
        if training_config["name"] == "FairnessConstrainedClassifier":
            params = training_config.get("parameters", {})

            base_estimator = None
            if params.get("base_estimator") == "LogisticRegression":
                base_estimator = LogisticRegression(
                    random_state=self.config["data"].get("random_state", 42),
                    max_iter=1000,
                    solver="lbfgs",
                    C=1.0,
                )

            self.model = FairnessConstrainedClassifier(
                base_estimator=base_estimator,
                sensitive_features=self.config["data"]["sensitive_features"],
                constraint=params.get("constraint", "demographic_parity"),
                random_state=self.config["data"].get("random_state", 42),
            )

            self.logger.log_model_info(
                training_config["name"],
                params.get("constraint", "demographic_parity"),
                params,
            )

            self.logger.log_stage_start("model_training")

            X_model = self._prepare_features_for_modeling(
                X_train_transformed.drop(
                    columns=self.config["data"]["sensitive_features"]
                ),
                fit_scaler=True,
            )
            sensitive_features = X_train_transformed[
                self.config["data"]["sensitive_features"]
            ]

            self.model.fit(X_model, y_train, sensitive_features)

            self.logger.log_stage_complete(
                "model_training",
                {
                    "model_fitted": True,
                    "uses_fairlearn": getattr(self.model, "use_fairlearn", False),
                },
            )

        self.logger.log_stage_complete("bias_mitigation")

        return X_train_transformed, X_test_transformed

    def _evaluate_final_fairness(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Dict[str, Any]:
        """Evaluate final model fairness and compare with baseline metrics."""
        if self.verbose:
            self.logger.log_stage_start("final_validation")

        X_test_numeric = self._prepare_features_for_modeling(
            X_test.drop(columns=self.config["data"]["sensitive_features"]),
            fit_scaler=False,
        )
        sensitive_features_test = X_test[self.config["data"]["sensitive_features"]]

        y_pred_final = self.model.predict(X_test_numeric, sensitive_features_test)

        sensitive_values = X_test[self.config["data"]["sensitive_features"][0]].values
        final_prediction_report = self.bias_detector.audit_predictions(
            y_test.values, y_pred_final, sensitive_values
        )

        self.final_report = final_prediction_report

        self.bias_detector.print_report(final_prediction_report, "FINAL PREDICTION")

        if self.baseline_report:
            baseline_metrics = self.baseline_report["prediction_audit"]["metrics"]
            final_metrics = final_prediction_report["metrics"]

            self.logger.log_stage_start(
                "improvement_comparison",
                {
                    "accuracy_baseline": baseline_metrics["accuracy"],
                    "accuracy_final": final_metrics["accuracy"],
                    "primary_metric": self.config["evaluation"]["primary_metric"],
                    "primary_baseline": baseline_metrics[
                        self.config["evaluation"]["primary_metric"]
                    ],
                    "primary_final": final_metrics[
                        self.config["evaluation"]["primary_metric"]
                    ],
                },
            )

            improvement = (
                baseline_metrics[self.config["evaluation"]["primary_metric"]]
                - final_metrics[self.config["evaluation"]["primary_metric"]]
            )
            if improvement > 0:
                self.logger.logger.info(
                    f"Primary metric improved by {improvement:.4f}",
                    extra={
                        "component": "evaluation",
                        "improvement": improvement,
                        "improvement_type": "positive",
                    },
                )
            else:
                self.logger.logger.info(
                    f"Primary metric changed by {improvement:.4f}",
                    extra={
                        "component": "evaluation",
                        "improvement": improvement,
                        "improvement_type": "negative"
                        if improvement < 0
                        else "neutral",
                    },
                )

            if self.verbose:
                self._display_improvement_comparison(baseline_metrics, final_metrics)

        return {
            "baseline_report": self.baseline_report,
            "final_report": self.final_report,
            "model": self.model,
            "transformer": self.transformer,
        }

    def _prepare_features_for_modeling(
        self, X: pd.DataFrame, fit_scaler: bool = False
    ) -> pd.DataFrame:
        """Prepare features for model training (handle categorical variables and scaling)."""
        X_processed = X.copy()

        X_num = X_processed.select_dtypes(include=["number"]).copy()
        X_cat = X_processed.select_dtypes(exclude=["number"]).copy()

        if len(X_num.columns) > 0:
            if fit_scaler:
                X_scaled = self.feature_scaler.fit_transform(X_num)
            else:
                X_scaled = self.feature_scaler.transform(X_num)
            X_num = pd.DataFrame(X_scaled, columns=X_num.columns, index=X_num.index)

        if len(X_cat.columns) > 0:
            result = pd.concat([X_num, X_cat], axis=1)
        else:
            result = X_num

        return result[X.columns]

    def _prepare_features_for_baseline(self, X: pd.DataFrame, scaler) -> pd.DataFrame:
        """Prepare features for baseline model using a separate scaler to avoid data leakage."""
        X_processed = X.copy()

        X_num = X_processed.select_dtypes(include=["number"]).copy()
        X_cat = X_processed.select_dtypes(exclude=["number"]).copy()

        if not X_num.empty:
            X_scaled = scaler.fit_transform(X_num)
            X_num = pd.DataFrame(X_scaled, columns=X_num.columns, index=X_num.index)

        if not X_cat.empty:
            X_cat = X_cat.apply(lambda s: s.astype("category").cat.codes)

        if not X_cat.empty:
            result = pd.concat([X_num, X_cat], axis=1)
        else:
            result = X_num

        return result[X.columns]

    def _display_improvement_comparison(
        self, baseline_metrics: Dict[str, float], final_metrics: Dict[str, float]
    ) -> None:
        """Display improvement comparison using Rich tables."""
        self.console.print("\n[bold blue]IMPROVEMENT COMPARISON[/bold blue]")

        performance_metrics = ["accuracy", "precision", "recall", "f1_score"]
        perf_table = Table(
            title="Performance Metrics Comparison",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold blue",
        )
        perf_table.add_column(
            "Metric", style="cyan", no_wrap=True, min_width=10, max_width=15
        )
        perf_table.add_column(
            "Baseline", style="yellow", justify="right", min_width=8, max_width=12
        )
        perf_table.add_column(
            "Final", style="green", justify="right", min_width=8, max_width=12
        )
        perf_table.add_column(
            "Change", style="bold", justify="right", min_width=15, max_width=25
        )
        perf_table.add_column(
            "Status", style="bold", justify="center", min_width=10, max_width=15
        )

        for metric in performance_metrics:
            if metric in baseline_metrics and metric in final_metrics:
                baseline_val = baseline_metrics[metric]
                final_val = final_metrics[metric]
                change = final_val - baseline_val
                change_pct = (change / baseline_val) * 100 if baseline_val != 0 else 0

                if change > 0:
                    status = "📈 Improved"
                    change_style = "green"
                elif change < 0:
                    status = "📉 Decreased"
                    change_style = "red"
                else:
                    status = "➡️ Unchanged"
                    change_style = "dim"

                perf_table.add_row(
                    metric.title(),
                    f"{baseline_val:.4f}",
                    f"{final_val:.4f}",
                    f"[{change_style}]{change:+.4f} ({change_pct:+.1f}%)[/{change_style}]",
                    status,
                )

        self.console.print(perf_table)

        fairness_metric_names = [
            metric_name
            for metric_name in baseline_metrics.keys()
            if "difference" in metric_name
        ]
        if fairness_metric_names:
            fairness_table = Table(
                title="Fairness Metrics Comparison",
                box=box.SIMPLE,
                show_header=True,
                header_style="bold blue",
            )
            fairness_table.add_column(
                "Metric", style="cyan", no_wrap=False, min_width=20, max_width=30
            )
            fairness_table.add_column(
                "Baseline", style="yellow", justify="right", min_width=8, max_width=12
            )
            fairness_table.add_column(
                "Final", style="green", justify="right", min_width=8, max_width=12
            )
            fairness_table.add_column(
                "Improvement", style="bold", justify="right", min_width=15, max_width=25
            )
            fairness_table.add_column(
                "Status", style="bold", justify="center", min_width=10, max_width=15
            )

            primary_metric = self.config["evaluation"]["primary_metric"]

            for metric_name in fairness_metric_names:
                if metric_name in final_metrics:
                    baseline_value = baseline_metrics[metric_name]
                    final_value = final_metrics[metric_name]
                    improvement = baseline_value - final_value
                    improvement_percentage = (
                        (improvement / baseline_value) * 100
                        if baseline_value != 0
                        else 0
                    )

                    if improvement > 0:
                        status = "✅ Better"
                        improvement_style = "green"
                    elif improvement < 0:
                        status = "❌ Worse"
                        improvement_style = "red"
                    else:
                        status = "➡️ Same"
                        improvement_style = "dim"

                    display_metric_name = metric_name.replace("_", " ").title()
                    if metric_name == primary_metric:
                        display_metric_name = f"🎯 {display_metric_name} (Primary)"

                    fairness_table.add_row(
                        display_metric_name,
                        f"{baseline_value:.4f}",
                        f"{final_value:.4f}",
                        f"[{improvement_style}]{improvement:+.4f} ({improvement_percentage:+.1f}%)[/{improvement_style}]",
                        status,
                    )

            self.console.print(fairness_table)

        primary_metric = self.config["evaluation"]["primary_metric"]
        if primary_metric in baseline_metrics and primary_metric in final_metrics:
            baseline_primary = baseline_metrics[primary_metric]
            final_primary = final_metrics[primary_metric]
            improvement = baseline_primary - final_primary

            summary_table = Table(
                title="Summary",
                box=box.SIMPLE,
                show_header=True,
                header_style="bold blue",
            )
            summary_table.add_column(
                "Assessment", style="cyan", no_wrap=False, min_width=20, max_width=30
            )
            summary_table.add_column(
                "Result", style="bold", justify="left", min_width=30, max_width=50
            )

            if improvement > 0:
                summary_table.add_row(
                    "Primary Fairness Goal",
                    f"[green]✅ Achieved ({improvement:.4f} improvement)[/green]",
                )
            else:
                summary_table.add_row(
                    "Primary Fairness Goal",
                    f"[red]❌ Not achieved ({improvement:.4f} change)[/red]",
                )

            threshold = self.config["evaluation"]["fairness_threshold"]
            if final_primary <= threshold:
                summary_table.add_row(
                    "Fairness Threshold",
                    f"[green]✅ Meets threshold (≤{threshold})[/green]",
                )
            else:
                summary_table.add_row(
                    "Fairness Threshold",
                    f"[red]❌ Exceeds threshold (>{threshold})[/red]",
                )

            self.console.print(summary_table)
            self.console.print("")

    def _log_results(self, results: Dict[str, Any]) -> None:
        """Log results to MLflow with enhanced validation and signatures."""
        if self.verbose:
            self.logger.log_stage_start("mlflow_logging")

        try:
            if results["baseline_report"]:
                baseline_metrics = results["baseline_report"]["prediction_audit"][
                    "metrics"
                ]
                for metric, value in baseline_metrics.items():
                    if isinstance(value, (int, float)) and not (
                        np.isnan(value) or np.isinf(value)
                    ):
                        mlflow.log_metric(f"baseline_{metric}", value)
                    else:
                        if self.verbose:
                            self.logger.log_warning(
                                f"Skipping invalid baseline metric {metric}: {value}"
                            )

            final_metrics = results["final_report"]["metrics"]
            for metric, value in final_metrics.items():
                if isinstance(value, (int, float)) and not (
                    np.isnan(value) or np.isinf(value)
                ):
                    mlflow.log_metric(f"final_{metric}", value)
                else:
                    if self.verbose:
                        self.logger.log_warning(
                            f"Skipping invalid final metric {metric}: {value}"
                        )

            primary_metric = self.config["evaluation"]["primary_metric"]
            if primary_metric in final_metrics:
                primary_value = final_metrics[primary_metric]
                if isinstance(primary_value, (int, float)) and not (
                    np.isnan(primary_value) or np.isinf(primary_value)
                ):
                    mlflow.log_metric("primary_fairness_metric", primary_value)

            if results["baseline_report"]:
                baseline_primary = results["baseline_report"]["prediction_audit"][
                    "metrics"
                ].get(primary_metric)
                final_primary = final_metrics.get(primary_metric)
                if baseline_primary is not None and final_primary is not None:
                    improvement = baseline_primary - final_primary
                    improvement_pct = (
                        (improvement / baseline_primary) * 100
                        if baseline_primary != 0
                        else 0
                    )
                    mlflow.log_metric("fairness_improvement", improvement)
                    mlflow.log_metric("fairness_improvement_pct", improvement_pct)

            if self.config["mlflow"].get("log_model", True) and results["model"]:
                self._log_model_with_signature(results)

            if results["transformer"]:
                self._log_transformer_details(results["transformer"])

        except Exception as e:
            if self.verbose:
                self.logger.log_error("MLflow logging failed", e)
            import traceback

            traceback.print_exc()

    def _log_model_with_signature(self, results: Dict[str, Any]) -> None:
        """Log model with proper signature and validation."""
        try:
            model = results["model"]

            sample_data = self._generate_sample_data_for_signature()

            if sample_data is not None and len(sample_data) > 0:
                sample_features = self._prepare_features_for_modeling(
                    sample_data.drop(columns=self.config["data"]["sensitive_features"]),
                    fit_scaler=False,
                )
                sensitive_features = sample_data[
                    self.config["data"]["sensitive_features"]
                ]

                sample_predictions = model.predict(sample_features, sensitive_features)

                signature = infer_signature(sample_features, sample_predictions)

                model_name = (
                    f"{self.experiment_name}_model"
                    if self.experiment_name
                    else f"{self.config['mlflow']['experiment_name']}_model"
                )

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*already exists.*")
                    mlflow.sklearn.log_model(
                        model,
                        artifact_path="fair_model",
                        signature=signature,
                        input_example=sample_features.head(3),
                        registered_model_name=model_name,
                        metadata={
                            "fairness_constraint": getattr(
                                model, "constraint_name", "unknown"
                            ),
                            "sensitive_features": self.config["data"][
                                "sensitive_features"
                            ],
                            "uses_fairlearn": getattr(model, "use_fairlearn", False),
                            "repair_level": getattr(
                                results.get("transformer"), "repair_level", None
                            ),
                        },
                    )

                if self.verbose:
                    self.logger.logger.info(
                        "✅ Model logged with signature and metadata",
                        extra={
                            "component": "mlflow",
                            "model_logging": "success_with_signature",
                        },
                    )

            else:
                model_name = (
                    f"{self.experiment_name}_model"
                    if self.experiment_name
                    else f"{self.config['mlflow']['experiment_name']}_model"
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*already exists.*")
                    mlflow.sklearn.log_model(
                        model,
                        artifact_path="fair_model",
                        registered_model_name=model_name,
                    )
                if self.verbose:
                    self.logger.log_warning(
                        "Model logged without signature (no sample data available)",
                        {"model_logging": "success_no_signature"},
                    )

        except Exception as e:
            if self.verbose:
                self.logger.log_warning(
                    f"Enhanced model logging failed, using basic logging: {e}"
                )
            try:
                model_name = (
                    f"{self.experiment_name}_model"
                    if self.experiment_name
                    else f"{self.config['mlflow']['experiment_name']}_model"
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*already exists.*")
                    mlflow.sklearn.log_model(
                        results["model"],
                        artifact_path="fair_model",
                        registered_model_name=model_name,
                    )
            except Exception as e2:
                if self.verbose:
                    self.logger.log_error("Basic model logging also failed", e2)

    def _generate_sample_data_for_signature(self) -> pd.DataFrame:
        """Generate small sample dataset for signature inference."""
        try:
            sample_data = self._generate_synthetic_data(n_samples=10)
            if self.config["data"]["target_column"] in sample_data.columns:
                sample_features = sample_data.drop(
                    columns=[self.config["data"]["target_column"]]
                )
                return sample_features
            return sample_data
        except Exception:
            return None

    def _log_transformer_details(self, transformer) -> None:
        """Log transformer configuration and statistics."""
        try:
            if hasattr(transformer, "get_mitigation_details"):
                details = transformer.get_mitigation_details()

                mlflow.log_param(
                    "bias_mitigation_method", details.get("method", "unknown")
                )
                mlflow.log_param("repair_level", details.get("repair_level", "unknown"))
                mlflow.log_param(
                    "sensitive_features", str(details.get("sensitive_features", []))
                )

                group_stats = details.get("group_statistics", {})
                for attr, groups in group_stats.items():
                    group_sizes = [stats["size"] for stats in groups.values()]
                    if group_sizes:
                        mlflow.log_metric(f"group_size_min_{attr}", min(group_sizes))
                        mlflow.log_metric(f"group_size_max_{attr}", max(group_sizes))
                        mlflow.log_metric(
                            f"group_size_mean_{attr}", np.mean(group_sizes)
                        )

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as f:
                    import json

                    json_safe_details = self._make_json_serializable(details)
                    json.dump(json_safe_details, f, indent=2)
                    mlflow.log_artifact(f.name, "transformer_details")

        except Exception as e:
            if self.verbose:
                self.logger.log_warning(f"Failed to log transformer details: {e}")

    def _make_json_serializable(self, obj) -> Any:
        """Convert numpy arrays and other non-serializable objects to JSON-safe format."""
        if isinstance(obj, dict):
            return {
                key: self._make_json_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)
