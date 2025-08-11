"""Configuration parsing and validation."""

import yaml
import re
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
from pydantic import BaseModel, Field, validator, ValidationError


class DataConfig(BaseModel):
    input_path: str = Field(..., description="Path to input data file")
    target_column: str = Field(..., description="Name of target column")
    sensitive_features: List[str] = Field(..., min_items=1, max_items=10, description="List of sensitive feature columns")
    test_size: float = Field(0.2, ge=0.01, le=0.99, description="Test set proportion")
    random_state: int = Field(42, description="Random seed")


class TransformerConfig(BaseModel):
    name: str = Field(..., description="Transformer class name")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Transformer parameters")
    
    @validator('name')
    def validate_transformer_name(cls, v):
        known_transformers = ['BiasMitigationTransformer']
        if v not in known_transformers:
            raise ValueError(f"Unknown transformer: {v}")
        return v


class PreprocessingConfig(BaseModel):
    transformer: TransformerConfig


class TrainingMethodConfig(BaseModel):
    name: str = Field(..., description="Training method name")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Training parameters")
    
    @validator('name')
    def validate_method_name(cls, v):
        known_methods = ['FairnessConstrainedClassifier']
        if v not in known_methods:
            raise ValueError(f"Unknown training method: {v}")
        return v


class TrainingConfig(BaseModel):
    method: TrainingMethodConfig


class EvaluationConfig(BaseModel):
    primary_metric: str = Field(..., description="Primary fairness metric")
    fairness_threshold: float = Field(..., gt=0, le=1.0, description="Fairness threshold")
    additional_metrics: List[str] = Field(default_factory=list, max_items=20, description="Additional metrics")


class MLflowConfig(BaseModel):
    experiment_name: str = Field(..., max_length=200, description="MLflow experiment name")
    run_name: Optional[str] = Field(None, description="MLflow run name")
    log_config: bool = Field(True, description="Whether to log config")
    log_model: bool = Field(True, description="Whether to log model")
    tags: Dict[str, str] = Field(default_factory=dict, description="MLflow tags")
    
    @validator('experiment_name')
    def validate_experiment_name(cls, v):
        if re.search(r'[<>:"/\\|?*]', v):
            raise ValueError("experiment_name contains invalid characters")
        return v


class PipelineConfig(BaseModel):
    data: DataConfig
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    mlflow: MLflowConfig


class ConfigParser:
    """Parse and validate pipeline configuration."""
    
    @staticmethod
    def load(config_path: Union[str, Path]) -> Dict[str, Any]:
        config_path = Path(config_path)
        
        try:
            config_path = config_path.resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid configuration path: {e}")
        
        if '..' in str(config_path) or str(config_path).startswith('/'):
            if not ConfigParser._is_safe_path(config_path):
                raise ValueError(f"Potentially unsafe configuration path: {config_path}")
        
        if config_path.exists():
            file_size = config_path.stat().st_size
            if file_size > 10 * 1024 * 1024:
                raise ValueError(f"Configuration file too large: {file_size} bytes (max 10MB)")
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                
                if config is None:
                    raise ValueError("Empty or invalid YAML configuration file")
                
                if not isinstance(config, dict):
                    raise ValueError("Configuration must be a YAML dictionary")
                
                ConfigParser._validate_security(config)
                
                return config
                
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Configuration file encoding error: {e}")
    
    @staticmethod
    def _is_safe_path(path: Path) -> bool:
        try:
            abs_path = path.resolve()
            
            dangerous_paths = [
                '/etc', '/proc', '/sys', '/dev', '/root',
                '/usr/bin', '/bin', '/sbin', '/boot'
            ]
            
            path_str = str(abs_path).lower()
            for dangerous in dangerous_paths:
                if path_str.startswith(dangerous.lower()):
                    return False
            
            suspicious_patterns = [
                r'\.\./', r'\\\.\\\.\\', r'/etc/passwd', r'/etc/shadow',
                r'\.ssh/', r'\.aws/', r'\.config/'
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, path_str):
                    return False
                    
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def _validate_security(config: Dict[str, Any]) -> None:
        ConfigParser._check_file_paths(config)
        ConfigParser._check_command_injection(config)
        ConfigParser._check_resource_limits(config)
    
    @staticmethod
    def _check_file_paths(config: Dict[str, Any], path: str = "") -> None:
        dangerous_patterns = [
            r'/etc/passwd', r'/etc/shadow', r'\.\./', r'\\\.\\\.\\',
            r'/proc/', r'/sys/', r'/dev/', r'~/.ssh', r'~/.aws',
            r'file://', r'ftp://', r'sftp://'
        ]
        
        for key, value in config.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                ConfigParser._check_file_paths(value, current_path)
            elif isinstance(value, str):
                value_lower = value.lower()
                for pattern in dangerous_patterns:
                    if re.search(pattern, value_lower):
                        raise ValueError(f"Suspicious file path detected in {current_path}: {value}")
                        
                if value.startswith('/') and not ConfigParser._is_safe_absolute_path(value):
                    raise ValueError(f"Potentially dangerous absolute path in {current_path}: {value}")
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        item_lower = item.lower()
                        for pattern in dangerous_patterns:
                            if re.search(pattern, item_lower):
                                raise ValueError(f"Suspicious path in {current_path}[{i}]: {item}")
    
    @staticmethod
    def _is_safe_absolute_path(path: str) -> bool:
        safe_prefixes = [
            '/tmp/', '/var/tmp/', '/home/', '/Users/',
            '/opt/', '/data/', '/mnt/', '/media/'
        ]
        
        path_lower = path.lower()
        return any(path_lower.startswith(prefix.lower()) for prefix in safe_prefixes)
    
    @staticmethod
    def _check_command_injection(config: Dict[str, Any], path: str = "") -> None:
        dangerous_patterns = [
            r';.*rm\s', r';.*del\s', r'`.*`', r'\$\(.*\)',
            r'&&.*rm\s', r'\|\|.*rm\s', r'exec\s*\(',
            r'eval\s*\(', r'system\s*\(', r'subprocess',
            r'os\.system', r'os\.popen', r'__import__',
            r'curl\s+', r'wget\s+', r'nc\s+', r'netcat'
        ]
        
        for key, value in config.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                ConfigParser._check_command_injection(value, current_path)
            elif isinstance(value, str):
                for pattern in dangerous_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        raise ValueError(f"Potential command injection in {current_path}: {value}")
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        for pattern in dangerous_patterns:
                            if re.search(pattern, item, re.IGNORECASE):
                                raise ValueError(f"Potential command injection in {current_path}[{i}]: {item}")
    
    @staticmethod  
    def _check_resource_limits(config: Dict[str, Any]) -> None:
        if 'data' in config:
            data = config['data']
            
            if 'test_size' in data:
                test_size = data['test_size']
                if not isinstance(test_size, (int, float)) or not (0.01 <= test_size <= 0.99):
                    raise ValueError(f"test_size must be between 0.01 and 0.99, got: {test_size}")
            
            if 'sensitive_features' in data:
                sensitive_features = data['sensitive_features']
                if not isinstance(sensitive_features, list):
                    raise ValueError("sensitive_features must be a list")
                if len(sensitive_features) == 0:
                    raise ValueError("At least one sensitive feature must be specified")
                if len(sensitive_features) > 10:
                    raise ValueError(f"Too many sensitive features (max 10): {len(sensitive_features)}")
        
        if 'preprocessing' in config and 'transformer' in config['preprocessing']:
            transformer = config['preprocessing']['transformer']
            if 'parameters' in transformer:
                params = transformer['parameters']
                
                if 'repair_level' in params:
                    repair_level = params['repair_level']
                    if not isinstance(repair_level, (int, float)) or not (0.0 <= repair_level <= 1.0):
                        raise ValueError("repair_level must be between 0.0 and 1.0")
        
        if 'evaluation' in config:
            evaluation = config['evaluation']
            
            if 'fairness_threshold' in evaluation:
                threshold = evaluation['fairness_threshold']
                if not isinstance(threshold, (int, float)) or threshold <= 0 or threshold > 1.0:
                    raise ValueError("fairness_threshold must be positive and <= 1.0")
            
            if 'additional_metrics' in evaluation:
                additional_metrics = evaluation['additional_metrics']
                if not isinstance(additional_metrics, list):
                    raise ValueError("additional_metrics must be a list")
                if len(additional_metrics) > 20:
                    raise ValueError(f"Too many additional metrics (max 20): {len(additional_metrics)}")
        
        if 'mlflow' in config:
            mlflow_config = config['mlflow']
            
            if 'experiment_name' in mlflow_config:
                exp_name = mlflow_config['experiment_name']
                if not isinstance(exp_name, str):
                    raise ValueError("experiment_name must be a string")
                if len(exp_name) > 200:
                    raise ValueError("experiment_name too long (max 200 characters)")
                if re.search(r'[<>:"/\\|?*]', exp_name):
                    raise ValueError("experiment_name contains invalid characters")
    
    @staticmethod
    def validate(config: Dict[str, Any]) -> list[str]:
        try:
            ConfigParser._validate_security(config)
            PipelineConfig(**config)
            return []
        except ValidationError as e:
            return [str(error) for error in e.errors()]
        except Exception as e:
            return [f"Validation error: {str(e)}"]