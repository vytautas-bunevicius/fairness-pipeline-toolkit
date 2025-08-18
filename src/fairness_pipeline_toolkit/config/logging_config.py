"""Configures structured JSON logging to provide consistent, machine-readable output.

This module sets up formatters and loggers to ensure that all pipeline events,
metrics, and errors can be easily parsed, filtered, and ingested by monitoring
systems. It separates performance metrics from general logs for clarity.
"""

import logging
import logging.config
import json
import sys
from typing import Dict, Any
from pathlib import Path
import time


class StructuredFormatter(logging.Formatter):
    """A custom logging formatter that outputs log records as a single JSON object.

    This approach ensures that log entries are self-contained and machine-readable,
    which is essential for reliable parsing and querying in log management systems.
    It includes core metadata by default and allows for adding extra context fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Formats a log record into a JSON string.

        The base log data includes standard fields like timestamp and level. It also
        dynamically includes any extra fields passed to the logger, allowing for
        flexible and context-rich structured logs.
        """
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if hasattr(record, "component"):
            log_data["component"] = record.component
        if hasattr(record, "stage"):
            log_data["stage"] = record.stage
        if hasattr(record, "metric_name"):
            log_data["metric_name"] = record.metric_name
        if hasattr(record, "metric_value"):
            log_data["metric_value"] = record.metric_value
        if hasattr(record, "duration"):
            log_data["duration_ms"] = record.duration
        if hasattr(record, "error_type"):
            log_data["error_type"] = record.error_type

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


class PerformanceLogger:
    """A dedicated logger for capturing performance-related metrics and timings.

    This class provides a simple interface for timing operations and logging key
    metrics. Separating performance logging from general application logging allows
    for focused performance analysis without the noise of other log messages.
    """

    def __init__(self, logger_name: str = "fairness_pipeline.performance"):
        self.logger = logging.getLogger(logger_name)
        self._start_times = {}

    def start_timer(self, operation_name: str) -> None:
        """Records the start time of a named operation for later duration calculation."""
        self._start_times[operation_name] = time.time()
        self.logger.info(
            f"Started {operation_name}",
            extra={
                "component": "performance",
                "stage": "start",
                "operation": operation_name,
            },
        )

    def end_timer(self, operation_name: str) -> float:
        """Calculates and logs the duration of a named operation since it started."""
        if operation_name not in self._start_times:
            self.logger.warning(f"Timer for {operation_name} was not started")
            return 0.0

        duration = (time.time() - self._start_times[operation_name]) * 1000
        del self._start_times[operation_name]

        self.logger.info(
            f"Completed {operation_name}",
            extra={
                "component": "performance",
                "stage": "complete",
                "operation": operation_name,
                "duration": duration,
            },
        )
        return duration

    def log_metric(
        self, metric_name: str, metric_value: float, stage: str = "evaluation"
    ) -> None:
        """Log a metric value."""
        self.logger.info(
            f"Metric {metric_name}: {metric_value:.4f}",
            extra={
                "component": "metrics",
                "stage": stage,
                "metric_name": metric_name,
                "metric_value": metric_value,
            },
        )

    def log_data_stats(self, data_stats: Dict[str, Any]) -> None:
        """Log data statistics."""
        self.logger.info(
            "Data statistics",
            extra={"component": "data", "stage": "preprocessing", **data_stats},
        )


def setup_logging(
    level: str = "INFO",
    log_file: str = None,
    structured: bool = True,
    console_output: bool = True,
) -> logging.Logger:
    """Initializes and configures the root logger for the entire application.

    This function provides a central entry point for setting up logging handlers
    (for console or file output) and formatters (structured JSON or plain text).
    It also sets the logging levels for noisy third-party libraries to reduce
    log spam and keep the application's output clean.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        structured: Use structured JSON logging
        console_output: Enable console output

    Returns:
        Configured logger instance
    """
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    handlers = []

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        if structured:
            console_handler.setFormatter(
                StructuredFormatter(datefmt="%Y-%m-%d %H:%M:%S")
            )
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
        handlers.append(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        if structured:
            file_handler.setFormatter(StructuredFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
        handlers.append(file_handler)

    for handler in handlers:
        handler.setLevel(numeric_level)
        root_logger.addHandler(handler)

    fairness_logger = logging.getLogger("fairness_pipeline")
    fairness_logger.setLevel(numeric_level)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("sklearn").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)

    return fairness_logger


class PipelineLogger:
    """A wrapper class providing a context-aware logger for pipeline stages.

    This logger simplifies logging for different pipeline components by automatically
    injecting component and stage information into log records. It provides
    helper methods for common logging patterns, such as starting/ending stages
    and logging errors, to ensure consistency in log structure.
    """

    def __init__(self, component_name: str):
        self.logger = logging.getLogger(f"fairness_pipeline.{component_name}")
        self.component = component_name
        self.perf_logger = PerformanceLogger()

    def log_stage_start(self, stage: str, details: Dict[str, Any] = None) -> None:
        """Log the start of a pipeline stage."""
        extra = {"component": self.component, "stage": stage, "status": "start"}
        if details:
            extra.update(details)

        self.logger.info(f"Starting {stage}", extra=extra)

    def log_stage_complete(self, stage: str, details: Dict[str, Any] = None) -> None:
        """Log the completion of a pipeline stage."""
        extra = {"component": self.component, "stage": stage, "status": "complete"}
        if details:
            extra.update(details)

        self.logger.info(f"Completed {stage}", extra=extra)

    def log_warning(self, message: str, details: Dict[str, Any] = None) -> None:
        """Log a warning with structured data."""
        extra = {"component": self.component}
        if details:
            extra.update(details)

        self.logger.warning(message, extra=extra)

    def log_error(
        self, message: str, error: Exception = None, details: Dict[str, Any] = None
    ) -> None:
        """Log an error with structured data."""
        extra = {"component": self.component}
        if error:
            extra["error_type"] = type(error).__name__
        if details:
            extra.update(details)

        if error:
            self.logger.error(message, exc_info=True, extra=extra)
        else:
            self.logger.error(message, extra=extra)

    def log_config_validation(self, errors: list) -> None:
        """Log configuration validation results."""
        if errors:
            self.logger.error(
                f"Configuration validation failed with {len(errors)} errors",
                extra={
                    "component": self.component,
                    "stage": "validation",
                    "error_count": len(errors),
                    "errors": errors,
                },
            )
        else:
            self.logger.info(
                "Configuration validation passed",
                extra={"component": self.component, "stage": "validation"},
            )

    def log_data_info(self, data_shape: tuple, sensitive_features: list) -> None:
        """Log data information."""
        self.logger.info(
            f"Loaded data with shape {data_shape}",
            extra={
                "component": self.component,
                "stage": "data_loading",
                "rows": data_shape[0],
                "columns": data_shape[1],
                "sensitive_features": sensitive_features,
            },
        )

    def log_model_info(
        self, model_type: str, constraint: str, parameters: Dict[str, Any]
    ) -> None:
        """Log model training information."""
        self.logger.info(
            f"Training {model_type} with {constraint} constraint",
            extra={
                "component": self.component,
                "stage": "model_training",
                "model_type": model_type,
                "constraint": constraint,
                "parameters": parameters,
            },
        )

    def log_fairness_metrics(
        self, metrics: Dict[str, float], stage: str = "evaluation"
    ) -> None:
        """Log fairness metrics."""
        for metric_name, metric_value in metrics.items():
            self.perf_logger.log_metric(metric_name, metric_value, stage)

    def start_timer(self, operation: str) -> None:
        """Start performance timing."""
        self.perf_logger.start_timer(operation)

    def end_timer(self, operation: str) -> float:
        """End performance timing."""
        return self.perf_logger.end_timer(operation)


def get_pipeline_logger(component_name: str) -> PipelineLogger:
    """A factory function to get a configured PipelineLogger instance.

    This ensures that all components get a logger with a consistent naming
    scheme (fairness_pipeline.<component_name>) without needing to instantiate
    the PipelineLogger directly.
    """
    return PipelineLogger(component_name)
