"""Configuration parsing and validation."""

import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigParser:
    """Parse and validate pipeline configuration."""
    
    @staticmethod
    def load(config_path: str | Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    @staticmethod
    def validate(config: Dict[str, Any]) -> list[str]:
        """Validate configuration parameters."""
        errors = []
        
        # Required sections
        required_sections = ['data', 'preprocessing', 'training', 'evaluation', 'mlflow']
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")
                continue
        
        # Data validation
        if 'data' in config:
            data = config['data']
            required_data = ['target_column', 'sensitive_features']
            for field in required_data:
                if field not in data:
                    errors.append(f"Missing data.{field}")
        
        # Preprocessing validation
        if 'preprocessing' in config and 'transformer' in config['preprocessing']:
            transformer = config['preprocessing']['transformer']
            if 'name' not in transformer:
                errors.append("Missing preprocessing.transformer.name")
        
        # Training validation
        if 'training' in config and 'method' in config['training']:
            method = config['training']['method']
            if 'name' not in method:
                errors.append("Missing training.method.name")
        
        # Evaluation validation
        if 'evaluation' in config:
            evaluation = config['evaluation']
            if 'primary_metric' not in evaluation:
                errors.append("Missing evaluation.primary_metric")
            if 'fairness_threshold' not in evaluation:
                errors.append("Missing evaluation.fairness_threshold")
        
        # MLflow validation
        if 'mlflow' in config:
            mlflow_config = config['mlflow']
            if 'experiment_name' not in mlflow_config:
                errors.append("Missing mlflow.experiment_name")
        
        return errors