"""Structured logging configuration for the fairness pipeline toolkit."""

import logging
import logging.config
import json
import sys
from typing import Dict, Any
from pathlib import Path
import time


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'component'):
            log_data['component'] = record.component
        if hasattr(record, 'stage'):
            log_data['stage'] = record.stage
        if hasattr(record, 'metric_name'):
            log_data['metric_name'] = record.metric_name
        if hasattr(record, 'metric_value'):
            log_data['metric_value'] = record.metric_value
        if hasattr(record, 'duration'):
            log_data['duration_ms'] = record.duration
        if hasattr(record, 'error_type'):
            log_data['error_type'] = record.error_type
            
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_data, default=str)


class PerformanceLogger:
    """Logger for performance monitoring."""
    
    def __init__(self, logger_name: str = 'fairness_pipeline.performance'):
        self.logger = logging.getLogger(logger_name)
        self._start_times = {}
    
    def start_timer(self, operation_name: str) -> None:
        """Start timing an operation."""
        self._start_times[operation_name] = time.time()
        self.logger.info(
            f"Started {operation_name}",
            extra={'component': 'performance', 'stage': 'start', 'operation': operation_name}
        )
    
    def end_timer(self, operation_name: str) -> float:
        """End timing an operation and return duration."""
        if operation_name not in self._start_times:
            self.logger.warning(f"Timer for {operation_name} was not started")
            return 0.0
            
        duration = (time.time() - self._start_times[operation_name]) * 1000
        del self._start_times[operation_name]
        
        self.logger.info(
            f"Completed {operation_name}",
            extra={
                'component': 'performance', 
                'stage': 'complete', 
                'operation': operation_name,
                'duration': duration
            }
        )
        return duration
    
    def log_metric(self, metric_name: str, metric_value: float, stage: str = 'evaluation') -> None:
        """Log a metric value."""
        self.logger.info(
            f"Metric {metric_name}: {metric_value:.4f}",
            extra={
                'component': 'metrics',
                'stage': stage,
                'metric_name': metric_name,
                'metric_value': metric_value
            }
        )
    
    def log_data_stats(self, data_stats: Dict[str, Any]) -> None:
        """Log data statistics."""
        self.logger.info(
            "Data statistics",
            extra={
                'component': 'data',
                'stage': 'preprocessing',
                **data_stats
            }
        )


def setup_logging(
    level: str = 'INFO',
    log_file: str = None,
    structured: bool = True,
    console_output: bool = True
) -> logging.Logger:
    """Setup structured logging configuration.
    
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
            console_handler.setFormatter(StructuredFormatter(
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
        else:
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        handlers.append(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        if structured:
            file_handler.setFormatter(StructuredFormatter(
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
        else:
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        handlers.append(file_handler)
    
    for handler in handlers:
        handler.setLevel(numeric_level)
        root_logger.addHandler(handler)
    
    fairness_logger = logging.getLogger('fairness_pipeline')
    fairness_logger.setLevel(numeric_level)
    
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING)
    logging.getLogger('mlflow').setLevel(logging.WARNING)
    
    return fairness_logger


class PipelineLogger:
    """Enhanced logger for pipeline operations."""
    
    def __init__(self, component_name: str):
        self.logger = logging.getLogger(f'fairness_pipeline.{component_name}')
        self.component = component_name
        self.perf_logger = PerformanceLogger()
    
    def log_stage_start(self, stage: str, details: Dict[str, Any] = None) -> None:
        """Log the start of a pipeline stage."""
        extra = {'component': self.component, 'stage': stage, 'status': 'start'}
        if details:
            extra.update(details)
            
        self.logger.info(f"Starting {stage}", extra=extra)
    
    def log_stage_complete(self, stage: str, details: Dict[str, Any] = None) -> None:
        """Log the completion of a pipeline stage."""
        extra = {'component': self.component, 'stage': stage, 'status': 'complete'}
        if details:
            extra.update(details)
            
        self.logger.info(f"Completed {stage}", extra=extra)
    
    def log_warning(self, message: str, details: Dict[str, Any] = None) -> None:
        """Log a warning with structured data."""
        extra = {'component': self.component}
        if details:
            extra.update(details)
            
        self.logger.warning(message, extra=extra)
    
    def log_error(self, message: str, error: Exception = None, details: Dict[str, Any] = None) -> None:
        """Log an error with structured data."""
        extra = {'component': self.component}
        if error:
            extra['error_type'] = type(error).__name__
        if details:
            extra.update(details)
            
        if error:
            self.logger.error(message, exc_info=True, extra=extra)
        else:
            self.logger.error(message, extra=extra)
    
    def log_config_validation(self, config: Dict[str, Any], errors: list) -> None:
        """Log configuration validation results."""
        if errors:
            self.logger.error(
                f"Configuration validation failed with {len(errors)} errors",
                extra={
                    'component': self.component,
                    'stage': 'validation',
                    'error_count': len(errors),
                    'errors': errors
                }
            )
        else:
            self.logger.info(
                "Configuration validation passed",
                extra={'component': self.component, 'stage': 'validation'}
            )
    
    def log_data_info(self, data_shape: tuple, sensitive_features: list) -> None:
        """Log data information."""
        self.logger.info(
            f"Loaded data with shape {data_shape}",
            extra={
                'component': self.component,
                'stage': 'data_loading',
                'rows': data_shape[0],
                'columns': data_shape[1],
                'sensitive_features': sensitive_features
            }
        )
    
    def log_model_info(self, model_type: str, constraint: str, parameters: Dict[str, Any]) -> None:
        """Log model training information."""
        self.logger.info(
            f"Training {model_type} with {constraint} constraint",
            extra={
                'component': self.component,
                'stage': 'model_training',
                'model_type': model_type,
                'constraint': constraint,
                'parameters': parameters
            }
        )
    
    def log_fairness_metrics(self, metrics: Dict[str, float], stage: str = 'evaluation') -> None:
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
    """Get a configured pipeline logger for a component."""
    return PipelineLogger(component_name)