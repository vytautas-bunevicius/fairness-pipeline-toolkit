# Fairness Pipeline Toolkit

A comprehensive Python toolkit for implementing fairness-aware machine learning pipelines with automated bias detection, data debiasing, and fairness-constrained model training.

## Installation

### Recommended: Using uv

```bash
git clone https://github.com/TuringCollegeSubmissions/vytbunev-AIET.DS.1.5.git
cd fairness-pipeline-toolkit
uv sync
```

### Alternative: Using pip

```bash
git clone https://github.com/TuringCollegeSubmissions/vytbunev-AIET.DS.1.5.git
cd fairness-pipeline-toolkit
pip install -e .
```

## Quick Start

### Option 1: Direct Python Execution
```bash
python run_pipeline.py config.yml
```

### Option 2: Using uv (Recommended)
```bash
uv run python run_pipeline.py config.yml
```

### Option 3: Interactive Jupyter Notebook
```bash
uv run jupyter lab notebooks/demo.ipynb
# or
jupyter lab notebooks/demo.ipynb
```

### View Results
```bash
mlflow ui  # View experiment tracking
```

### Run Tests
```bash
python -m pytest tests/
# or
uv run python -m pytest tests/
```

## Architecture

```mermaid
flowchart TD
    %% Input Layer
    DATA["Input Data<br/>• CSV files<br/>• Target & sensitive features<br/>• Train/test split"]
    CONFIG["config.yml<br/>• Pipeline configuration<br/>• Model parameters<br/>• Fairness thresholds"]
    
    %% Orchestration Layer
    EXECUTOR["Pipeline Executor<br/>run_pipeline.py<br/>• Configuration parsing<br/>• Component coordination<br/>• Error handling"]
    
    %% Three-Step Process
    subgraph PIPELINE ["Three-Step Fairness Pipeline"]
        STEP1["Step 1: Baseline<br/>• Bias detection<br/>• Initial metrics<br/>• Baseline model"]
        STEP2["Step 2: Mitigation<br/>• Data preprocessing<br/>• Bias reduction<br/>• Feature scaling"]
        STEP3["Step 3: Fair Training<br/>• Constrained learning<br/>• Model optimization<br/>• Final evaluation"]
        
        STEP1 --> STEP2
        STEP2 --> STEP3
    end
    
    %% Core Modules
    subgraph MODULES ["Core Modules"]
        MEASUREMENT["Measurement<br/>• BiasDetector<br/>• FairnessMetrics<br/>• Statistical tests"]
        TRANSFORM["Pipeline<br/>• BiasMitigation<br/>• Data transforms<br/>• Preprocessing"]
        TRAINING["Training<br/>• Fair classifiers<br/>• Constraint methods<br/>• Model validation"]
    end
    
    %% Integration Layer
    subgraph INTEGRATION ["Integration Layer"]
        MLFLOW["MLflow<br/>• Experiment tracking<br/>• Model registry<br/>• Artifact storage"]
        LOGGING["Logging<br/>• Structured logs<br/>• Performance monitoring<br/>• Audit trails"]
    end
    
    %% Outputs
    RESULTS["Results<br/>• Fair models<br/>• Metrics reports<br/>• Improvement analysis"]
    
    %% Flow connections
    DATA --> EXECUTOR
    CONFIG --> EXECUTOR
    
    EXECUTOR --> STEP1
    EXECUTOR --> STEP2
    EXECUTOR --> STEP3
    
    STEP1 -.-> MEASUREMENT
    STEP2 -.-> TRANSFORM
    STEP3 -.-> TRAINING
    
    STEP1 --> MLFLOW
    STEP2 --> MLFLOW
    STEP3 --> MLFLOW
    
    EXECUTOR --> LOGGING
    STEP1 --> LOGGING
    STEP2 --> LOGGING
    STEP3 --> LOGGING
    
    STEP3 --> RESULTS
    MLFLOW --> RESULTS
    
    %% Styling
    classDef inputStyle fill:#E3F2FD,stroke:#1976D2,stroke-width:2px
    classDef executorStyle fill:#E8F5E8,stroke:#388E3C,stroke-width:3px
    classDef stepStyle fill:#FFF3E0,stroke:#F57C00,stroke-width:2px
    classDef moduleStyle fill:#FCE4EC,stroke:#C2185B,stroke-width:2px
    classDef integrationStyle fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px
    classDef outputStyle fill:#E0F2F1,stroke:#00796B,stroke-width:2px
    
    class DATA,CONFIG inputStyle
    class EXECUTOR executorStyle
    class STEP1,STEP2,STEP3 stepStyle
    class MEASUREMENT,TRANSFORM,TRAINING moduleStyle
    class MLFLOW,LOGGING integrationStyle
    class RESULTS outputStyle
```

## What It Does

The toolkit executes a three-step fairness pipeline:

1. **Baseline Measurement**: Analyzes raw data for bias and fairness violations
2. **Data Processing & Training**: Applies bias mitigation and trains fair models
3. **Final Validation**: Compares results and generates improvement reports

## Configuration

Edit `config.yml` to define your pipeline:

```yaml
data:
  input_path: "your_data.csv"
  target_column: "target"
  sensitive_features: ["race", "sex"]
  test_size: 0.2
  random_state: 42

preprocessing:
  transformer:
    name: "BiasMitigationTransformer"
    parameters:
      repair_level: 0.8

training:
  method:
    name: "FairnessConstrainedClassifier"
    parameters:
      base_estimator: "LogisticRegression"
      constraint: "demographic_parity"

evaluation:
  primary_metric: "demographic_parity_difference"
  fairness_threshold: 0.1

mlflow:
  experiment_name: "fairness_pipeline"
  log_model: true
```

## Core Components

### Configuration Module
- **ConfigParser**: Validates and manages pipeline configuration from YAML files
- **Logging System**: Structured JSON logging with performance monitoring and fairness metrics tracking

### Measurement Module
- **BiasDetector**: Identifies bias in datasets and model predictions
- **FairnessMetrics**: Calculates demographic parity, equalized odds, and other fairness metrics

### Pipeline Module
- **BiasMitigationTransformer**: Reduces bias by adjusting feature distributions

### Training Module
- **FairnessConstrainedClassifier**: Enforces fairness constraints during model training
- **FairClassifier**: Base class for implementing fair classification algorithms

## MLflow Integration

Automatically tracks and logs:
- **Performance metrics**: accuracy, precision, recall, F1-score
- **Fairness metrics**: demographic parity difference, equalized odds difference
- **Model artifacts**: trained models with metadata
- **Configuration files**: complete pipeline settings for reproducibility

## Requirements

- Python 3.13+
- Core dependencies: scikit-learn, pandas, numpy, fairlearn, mlflow
- Optional: jupyter (for demo notebook)

## Development

```bash
# Install with development dependencies
uv sync --extra dev

# Run code quality checks
uv run ruff check
uv run mypy src/

# Install with notebook support
uv sync --extra notebook
```

## Example Output

```
Loading configuration from: config.yml
✓ Configuration validated

Baseline Dataset Report:
Dataset Shape: (1000, 5)
Target Rate Difference: 0.234

Baseline Prediction Report:
Demographic Parity Difference: 0.234 (VIOLATION: > 0.1)
Equalized Odds Difference: 0.187 (VIOLATION: > 0.1)
Accuracy: 0.823

Final Prediction Report:
Demographic Parity Difference: 0.089 (OK: < 0.1)
Equalized Odds Difference: 0.095 (OK: < 0.1)
Accuracy: 0.801

Improvement summary:
Accuracy: 0.823 → 0.801
Primary Fairness Metric (demographic_parity_difference):
  Baseline: 0.234
  Final: 0.089
  Improvement: 0.145

✓ Pipeline execution completed successfully
```

## Architecture Design

This toolkit follows modern Python 2025 best practices:

- **pyproject.toml**: Handles project metadata, dependencies, and build configuration
- **config.yml**: Manages runtime application settings for pipeline behavior
- **Modular design**: Separate modules for measurement, pipeline processing, and training
- **MLflow integration**: Automatic experiment tracking and model versioning
- **Type safety**: Full type hints and mypy compatibility

The separation between project configuration (pyproject.toml) and application configuration (config.yml) ensures clear boundaries between development tooling and runtime behavior, following current Python ecosystem standards.

## License

**Unlicense** - This software is released into the public domain. You are free to use, modify, and distribute this code without any restrictions. See the LICENSE file for full details.

The Unlicense promotes maximum freedom and removes all copyright restrictions, making this toolkit freely available for any use case, including commercial applications.