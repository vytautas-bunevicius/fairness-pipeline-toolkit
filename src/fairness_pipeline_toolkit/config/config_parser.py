"""Configuration parsing and validation utilities for the fairness pipeline.

This module contains Pydantic models for configuration and a parser that
performs several validation steps.
These steps include schema validation
using Pydantic models (shape, types, ranges), security controls for paths
and command-like strings, and resource and size limits to prevent
excessive resource usage.

The resulting configuration is intended to be clear,
predictable, and extensible.
"""

import yaml
import re
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
from pydantic import BaseModel, Field, field_validator, ValidationError
import tomllib


class DataConfig(BaseModel):
    """Dataset and split configuration.

    Defines the configuration for loading data and creating a reproducible
    train/test split.
    Includes constraints on test_size and sensitive_features
    to prevent misconfigurations that would make the analysis unreliable.

    Attributes:
      input_path (str): Path to the input data file.
      target_column (str): Name of the target column in the dataset.
      sensitive_features (List[str]):
      Names of sensitive attribute columns.
      test_size (float): Proportion of the dataset to use for testing.
      random_state (int): Seed for the random number generator.
    """

    input_path: str = Field(..., description="Path to input data file")
    target_column: str = Field(..., description="Name of target column")
    sensitive_features: List[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of sensitive feature columns",
    )
    test_size: float = Field(0.2, ge=0.01, le=0.99, description="Test set proportion")
    random_state: int = Field(42, description="Random seed")


class TransformerConfig(BaseModel):
    """Preprocessing transformer selection.

    Restricting the transformer name to a known set reduces configuration drift
    and avoids runtime errors from non-existent or unsafe components.

    Attributes:
      name (str): Transformer class identifier.
      Must be one of the supported
        options.
      parameters (Dict[str, Any]): Keyword arguments forwarded to the
        transformer.
        Use this to control behavior (e.g., repair levels).
    """

    name: str = Field(..., description="Transformer class name")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Transformer parameters"
    )

    @field_validator("name")
    def validate_transformer_name(cls, v):
        """Validate the transformer name against the allowlist.

        Args:
          v (str): Provided transformer name.

        Raises:
          ValueError: If the name is not in the known allowlist.

        Returns:
          str: The validated transformer name.
        """
        known_transformers = ["BiasMitigationTransformer"]
        if v not in known_transformers:
            raise ValueError(f"Unknown transformer: {v}")
        return v


class PreprocessingConfig(BaseModel):
    """Container for preprocessing configuration.

    Attributes:
      transformer (TransformerConfig):
      The preprocessing transformer settings
        applied prior to training.
    """

    transformer: TransformerConfig


class TrainingMethodConfig(BaseModel):
    """Training algorithm configuration.

    Keeping the method name constrained and parameters explicit ensures model
    training is reproducible and comparable across experiments.

    Attributes:
      name (str): Training method identifier.
      Must be a supported method.
      parameters (Dict[str, Any]):
      Keyword arguments passed to the training
        routine to tune behavior and constraints.
    """

    name: str = Field(..., description="Training method name")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Training parameters"
    )

    @field_validator("name")
    def validate_method_name(cls, v):
        """Validate the training method name against the allowlist.

        Args:
          v (str): Provided training method name.

        Raises:
          ValueError: If the method is not supported.

        Returns:
          str: The validated method name.
        """
        known_methods = ["FairnessConstrainedClassifier"]
        if v not in known_methods:
            raise ValueError(f"Unknown training method: {v}")
        return v


class TrainingConfig(BaseModel):
    """Container for model training configuration.

    Attributes:
      method (TrainingMethodConfig):
      The selected algorithm and its parameters.
    """

    method: TrainingMethodConfig


class EvaluationConfig(BaseModel):
    """Evaluation configuration with fairness objectives.

    The fairness_threshold provides a clear bar for acceptable disparity.
    Limiting additional_metrics helps maintain focused reports.

    Attributes:
      primary_metric (str): Metric optimized or prioritized in reporting.
      fairness_threshold (float): Target bound in [0, 1] for fairness
        constraints/alerts.
        Must be > 0 up to 1.0.
      additional_metrics (List[str]): Extra metrics to report.
      Kept small to
        avoid overly wide dashboards.
    """

    primary_metric: str = Field(..., description="Primary fairness metric")
    fairness_threshold: float = Field(
        ..., gt=0, le=1.0, description="Fairness threshold"
    )
    additional_metrics: List[str] = Field(
        default_factory=list, max_length=20, description="Additional metrics"
    )


class MLflowConfig(BaseModel):
    """MLflow tracking configuration.

    Naming rules guard against path-like or shell-unfriendly names in tracking
    systems and ensure a consistent experiment taxonomy.

    Attributes:
      experiment_name (str): MLflow experiment name
      (max 200 chars, restricted
        characters).
      run_name (Optional[str]): Optional run name to distinguish runs.
      log_config (bool): Whether to log configuration into MLflow.
      log_model (bool): Whether to log the trained model artifact.
      tags (Dict[str, str]): Tags to attach to the MLflow run.
    """

    experiment_name: str = Field(
        ..., max_length=200, description="MLflow experiment name"
    )
    run_name: Optional[str] = Field(None, description="MLflow run name")
    log_config: bool = Field(True, description="Whether to log config")
    log_model: bool = Field(True, description="Whether to log model")
    tags: Dict[str, str] = Field(default_factory=dict, description="MLflow tags")

    @field_validator("experiment_name")
    def validate_experiment_name(cls, v):
        """Validate experiment name against disallowed characters.

        Args:
          v (str): Provided experiment name.

        Raises:
          ValueError:
          If the name contains characters that are unsafe in
            common filesystems or UIs.

        Returns:
          str: The validated experiment name.
        """
        if re.search(r'[<>:"/\\|?*]', v):
            raise ValueError("experiment_name contains invalid characters")
        return v


class VisualizationColorsConfig(BaseModel):
    """Color palette settings.

    Hex color constraints ensure valid rendering and consistent visuals across
    plots.

    Attributes:
      primary (str): Primary color as a hex string.
      secondary (str): Secondary color as a hex string.
      accent (str): Accent color as a hex string.
      success (str): Success color as a hex string.
      warning (str): Warning color as a hex string.
      danger (str): Danger/error color as a hex string.
    """

    primary: str = Field(
        "#3A5CED", pattern=r"^#[0-9A-Fa-f]{6}$", description="Primary color"
    )
    secondary: str = Field(
        "#7E7AE6", pattern=r"^#[0-9A-Fa-f]{6}$", description="Secondary color"
    )
    accent: str = Field(
        "#7BC0FF", pattern=r"^#[0-9A-Fa-f]{6}$", description="Accent color"
    )
    success: str = Field(
        "#82E5E8", pattern=r"^#[0-9A-Fa-f]{6}$", description="Success color"
    )
    warning: str = Field(
        "#C2A9FF", pattern=r"^#[0-9A-Fa-f]{6}$", description="Warning color"
    )
    danger: str = Field(
        "#D30B3B", pattern=r"^#[0-9A-Fa-f]{6}$", description="Danger/error color"
    )


class VisualizationFontsConfig(BaseModel):
    """Font configuration for plot text.

    Attributes:
      family (str): CSS font-family stack used in plots.
      title_size (int): Title font size in points.
      subtitle_size (int): Subtitle font size in points.
      axis_size (int): Axis label font size in points.
    """

    family: str = Field("Gordita, Figtree, sans-serif", description="Font family")
    title_size: int = Field(24, ge=12, le=48, description="Title font size")
    subtitle_size: int = Field(20, ge=10, le=36, description="Subtitle font size")
    axis_size: int = Field(16, ge=8, le=24, description="Axis font size")


class VisualizationLayoutConfig(BaseModel):
    """Layout configuration for plots.

    Attributes:
      height (int): Default plot height in pixels.
      margins (Dict[str, int]): Plot margins in Plotly format
      (l, r, t, b, pad).
    """

    height: int = Field(600, ge=200, le=1200, description="Default plot height")
    margins: Dict[str, int] = Field(
        default_factory=lambda: {"l": 60, "r": 150, "t": 100, "b": 80, "pad": 10},
        description="Plot margins (Plotly format: l, r, t, b, pad)",
    )


class VisualizationConfig(BaseModel):
    """High-level visualization configuration.

    This aggregates color, font, and layout defaults. The from_pyproject helper
    allows centralizing visualization preferences in pyproject.toml, keeping
    repository-wide styling consistent and discoverable.

    Attributes:
      theme (str): Visualization theme preset name.
      colors (VisualizationColorsConfig): Color palette settings.
      fonts (VisualizationFontsConfig): Font settings.
      layout (VisualizationLayoutConfig): Layout and margin settings.
    """

    theme: str = Field("default", description="Visualization theme")
    colors: VisualizationColorsConfig = Field(default_factory=VisualizationColorsConfig)
    fonts: VisualizationFontsConfig = Field(default_factory=VisualizationFontsConfig)
    layout: VisualizationLayoutConfig = Field(default_factory=VisualizationLayoutConfig)

    @classmethod
    def from_pyproject(
        cls, pyproject_path: Optional[Path] = None
    ) -> "VisualizationConfig":
        """Load visualization settings from a pyproject.toml.

        The method searches upward from the current working directory for the
        closest pyproject.toml.
        If found, it reads settings under:
        [tool.fairness_pipeline_toolkit.visualization].

        Args:
          pyproject_path (Optional[Path]): Optional explicit path to
            pyproject.toml.
            If not provided, a parent search is performed.

        Returns:
          VisualizationConfig:
          A configuration instance populated from the file,
          or defaults if none found or parsing fails.
        """
        if pyproject_path is None:
            current_dir = Path.cwd()
            for parent in [current_dir] + list(current_dir.parents):
                candidate = parent / "pyproject.toml"
                if candidate.exists():
                    pyproject_path = candidate
                    break

        if pyproject_path and pyproject_path.exists():
            try:
                with open(pyproject_path, "rb") as f:
                    config_data = tomllib.load(f)

                viz_config = (
                    config_data.get("tool", {})
                    .get("fairness_pipeline_toolkit", {})
                    .get("visualization", {})
                )

                if viz_config:
                    return cls._from_pyproject_dict(viz_config)
            except (IOError, tomllib.TOMLDecodeError):
                # Fall back to defaults on any error to remain resilient.
                pass

        return cls()

    @classmethod
    def _from_pyproject_dict(cls, data: Dict[str, Any]) -> "VisualizationConfig":
        """Create a VisualizationConfig from a flattened TOML dict.

        The pyproject configuration may use flattened dot keys (e.g.,
        colors.primary = "#123456").
        This helper expands the structure and
        builds validated config objects.

        Args:
          data (Dict[str, Any]): Flattened dict from TOML.

        Returns:
          VisualizationConfig:
          An instance with validated nested settings.
        """

        def unflatten_dict(flat_dict: Dict[str, Any]) -> Dict[str, Any]:
            result = {}
            for key, value in flat_dict.items():
                parts = key.split(".")
                current = result
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            return result

        nested_data = unflatten_dict(data)

        colors_data = nested_data.get("colors", {})
        fonts_data = nested_data.get("fonts", {})
        layout_data = nested_data.get("layout", {})

        colors = (
            VisualizationColorsConfig(**colors_data)
            if colors_data
            else VisualizationColorsConfig()
        )
        fonts = (
            VisualizationFontsConfig(**fonts_data)
            if fonts_data
            else VisualizationFontsConfig()
        )
        layout = (
            VisualizationLayoutConfig(**layout_data)
            if layout_data
            else VisualizationLayoutConfig()
        )

        return cls(
            theme=nested_data.get("theme", "default"),
            colors=colors,
            fonts=fonts,
            layout=layout,
        )


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration.

    This model stitches together all pipeline sections into a single validated
    object, enabling end-to-end validation in one pass before execution.

    Attributes:
      data (DataConfig): Data loading and split settings.
      preprocessing (PreprocessingConfig): Preprocessing stage settings.
      training (TrainingConfig): Training algorithm settings.
      evaluation (EvaluationConfig):
      Evaluation and fairness goal settings.
      mlflow (MLflowConfig): MLflow experiment tracking settings.
    """

    data: DataConfig
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    mlflow: MLflowConfig


class ConfigParser:
    """Parse and validate pipeline configuration safely.

    This class provides a YAML loader with additional security and resource
    checks.
    It prevents dangerous paths, command-like strings, or oversize
    configurations from slipping into downstream components.
    Validations are
    conservative by design to prefer safety and debuggability.
    """

    @staticmethod
    def load(config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load and validate a YAML configuration file.

        This method performs several layers of checks:
        - Path resolution and allowlist checks for safety.
        - File size limits to avoid loading unexpectedly large files.
        - YAML parsing and type checks.
        - Security validations on values (paths, commands, limits).

        Args:
          config_path (Union[str, Path]):
          Path to the YAML configuration file.

        Raises:
          ValueError: If the path is invalid,
          unsafe, file too large, YAML is
            invalid, or the contents fail security validations.
          FileNotFoundError: If the path does not exist.
          UnicodeDecodeError: If the file cannot be decoded as UTF-8.

        Returns:
          Dict[str, Any]: The parsed configuration dictionary.
        """
        config_path = Path(config_path)

        try:
            config_path = config_path.resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid configuration path: {e}")

        if not ConfigParser._is_safe_path(config_path):
            raise ValueError(f"Potentially unsafe configuration path: {config_path}")

        if config_path.exists():
            file_size = config_path.stat().st_size
            if file_size > 10 * 1024 * 1024:
                raise ValueError(
                    f"Configuration file too large: {file_size} bytes (max 10MB)"
                )

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as file:
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
        """Return whether a path is considered safe to read.

        The safety policy rejects references to system directories and commonly
        sensitive locations.
        Assumes the path is already resolved and absolute.

        Args:
          path (Path): Absolute path to check.

        Returns:
          bool: True if the path is considered safe, False otherwise.
        """
        dangerous_paths = [
            "/etc",
            "/proc",
            "/sys",
            "/dev",
            "/root",
            "/usr/bin",
            "/bin",
            "/sbin",
            "/boot",
        ]

        path_str = str(path).lower()
        for dangerous in dangerous_paths:
            if path_str.startswith(dangerous.lower()):
                return False

        suspicious_patterns = [
            r"/etc/passwd",
            r"/etc/shadow",
            r"\.ssh/",
            r"\.aws/",
            r"\.config/",
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, path_str):
                return False

        return True

    @staticmethod
    def _validate_security(config: Dict[str, Any]) -> None:
        """Run security-focused validations on the configuration.

        This includes:
        - File path checks to avoid sensitive directories.
        - Command injection checks against common patterns.
        - Resource limit checks to keep settings within safe bounds.

        Args:
          config (Dict[str, Any]): Parsed configuration dictionary.

        Raises:
          ValueError: If any validation fails.
        """
        ConfigParser._check_file_paths(config)
        ConfigParser._check_command_injection(config)
        ConfigParser._check_resource_limits(config)

    @staticmethod
    def _check_file_paths(config: Dict[str, Any], path: str = "") -> None:
        """Recursively validate file path values.

        The function traverses nested dictionaries and lists to flag unsafe
        or suspicious absolute paths and known sensitive locations.

        Args:
          config (Dict[str, Any]): Configuration subtree to validate.
          path (str): Dot-delimited path used for error context.

        Raises:
          ValueError: If an unsafe path is detected.
        """
        dangerous_patterns = [
            r"/etc/passwd",
            r"/etc/shadow",
            r"\.\./",
            r"\\\.\\\.\\",
            r"/proc/",
            r"/sys/",
            r"/dev/",
            r"~/.ssh",
            r"~/.aws",
            r"file://",
            r"ftp://",
            r"sftp://",
        ]

        for key, value in config.items():
            current_path = f"{path}.{key}" if path else key

            if isinstance(value, dict):
                ConfigParser._check_file_paths(value, current_path)
            elif isinstance(value, str):
                value_lower = value.lower()
                for pattern in dangerous_patterns:
                    if re.search(pattern, value_lower):
                        raise ValueError(
                            f"Suspicious file path detected in {current_path}: {value}"
                        )

                if value.startswith("/") and not ConfigParser._is_safe_absolute_path(
                    value
                ):
                    raise ValueError(
                        f"Potentially dangerous absolute path in {current_path}: {value}"
                    )
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        item_lower = item.lower()
                        for pattern in dangerous_patterns:
                            if re.search(pattern, item_lower):
                                raise ValueError(
                                    f"Suspicious path in {current_path}[{i}]: {item}"
                                )

    @staticmethod
    def _is_safe_absolute_path(path: str) -> bool:
        """Return whether an absolute path starts with a safe prefix.

        Args:
          path (str): Absolute path string.

        Returns:
          bool: True if the path starts with an allowlisted prefix.
        """
        safe_prefixes = [
            "/tmp/",
            "/var/tmp/",
            "/home/",
            "/Users/",
            "/opt/",
            "/data/",
            "/mnt/",
            "/media/",
        ]

        path_lower = path.lower()
        return any(path_lower.startswith(prefix.lower()) for prefix in safe_prefixes)

    @staticmethod
    def _check_command_injection(config: Dict[str, Any], path: str = "") -> None:
        """Recursively scan for command injection patterns.

        Args:
          config (Dict[str, Any]): Configuration subtree to scan.
          path (str): Dot-delimited path used for error context.

        Raises:
          ValueError:
          If potentially dangerous command-like content is found.
        """
        dangerous_patterns = [
            r";.*rm\s",
            r";.*del\s",
            r"`.*`",
            r"\$\(.*\)",
            r"&&.*rm\s",
            r"\|\|.*rm\s",
            r"exec\s*\(",
            r"eval\s*\(",
            r"system\s*\(",
            r"subprocess",
            r"os\.system",
            r"os\.popen",
            r"__import__",
            r"curl\s+",
            r"wget\s+",
            r"nc\s+",
            r"netcat",
        ]

        for key, value in config.items():
            current_path = f"{path}.{key}" if path else key

            if isinstance(value, dict):
                ConfigParser._check_command_injection(value, current_path)
            elif isinstance(value, str):
                for pattern in dangerous_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        raise ValueError(
                            f"Potential command injection in {current_path}: {value}"
                        )
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        for pattern in dangerous_patterns:
                            if re.search(pattern, item, re.IGNORECASE):
                                raise ValueError(
                                    f"Potential command injection in {current_path}[{i}]: {item}"
                                )

    @staticmethod
    def _check_resource_limits(config: Dict[str, Any]) -> None:
        """Validate resource-related and count limits.

        This ensures thresholds and list sizes are within reasonable bounds to
        keep evaluations meaningful and prevent excessive resource use.

        Args:
          config (Dict[str, Any]): Parsed configuration dictionary.

        Raises:
          ValueError: If a limit is violated.
        """
        if "data" in config:
            data = config["data"]

            if "test_size" in data:
                test_size = data["test_size"]
                if not isinstance(test_size, (int, float)) or not (
                    0.01 <= test_size <= 0.99
                ):
                    raise ValueError(
                        f"test_size must be between 0.01 and 0.99, got: {test_size}"
                    )

            if "sensitive_features" in data:
                sensitive_features = data["sensitive_features"]
                if not isinstance(sensitive_features, list):
                    raise ValueError("sensitive_features must be a list")
                if len(sensitive_features) == 0:
                    raise ValueError("At least one sensitive feature must be specified")
                if len(sensitive_features) > 10:
                    raise ValueError(
                        f"Too many sensitive features (max 10): {len(sensitive_features)}"
                    )

        if "preprocessing" in config and "transformer" in config["preprocessing"]:
            transformer = config["preprocessing"]["transformer"]
            if "parameters" in transformer:
                params = transformer["parameters"]

                if "repair_level" in params:
                    repair_level = params["repair_level"]
                    if not isinstance(repair_level, (int, float)) or not (
                        0.0 <= repair_level <= 1.0
                    ):
                        raise ValueError("repair_level must be between 0.0 and 1.0")

        if "evaluation" in config:
            evaluation = config["evaluation"]

            if "fairness_threshold" in evaluation:
                threshold = evaluation["fairness_threshold"]
                if (
                    not isinstance(threshold, (int, float))
                    or threshold <= 0
                    or threshold > 1.0
                ):
                    raise ValueError("fairness_threshold must be positive and <= 1.0")

            if "additional_metrics" in evaluation:
                additional_metrics = evaluation["additional_metrics"]
                if not isinstance(additional_metrics, list):
                    raise ValueError("additional_metrics must be a list")
                if len(additional_metrics) > 20:
                    raise ValueError(
                        f"Too many additional metrics (max 20): {len(additional_metrics)}"
                    )

        if "mlflow" in config:
            mlflow_config = config["mlflow"]

            if "experiment_name" in mlflow_config:
                exp_name = mlflow_config["experiment_name"]
                if not isinstance(exp_name, str):
                    raise ValueError("experiment_name must be a string")
                if len(exp_name) > 200:
                    raise ValueError("experiment_name too long (max 200 characters)")
                if re.search(r'[<>:"/\\|?*]', exp_name):
                    raise ValueError("experiment_name contains invalid characters")

    @staticmethod
    def validate(config: Dict[str, Any]) -> list[str]:
        """Validate a configuration dict against the schema and security rules.

        Args:
          config (Dict[str, Any]): Parsed configuration dictionary.

        Returns:
          list[str]: An empty list if valid,
          or a list of human-readable error
          messages describing validation failures.
        """
        try:
            ConfigParser._validate_security(config)
            PipelineConfig(**config)
            return []
        except ValidationError as e:
            return [str(error) for error in e.errors()]
        except Exception as e:
            return [f"Validation error: {str(e)}"]
