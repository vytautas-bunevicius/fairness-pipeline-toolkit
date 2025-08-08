# Fairness Pipeline Toolkit

A Python toolkit for bias detection, data debiasing, and fairness-aware model training with MLflow integration.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Pipeline Components                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ Measurement │  │  Pipeline   │  │  Training   │      │
│  │   Module    │  │   Module    │  │   Module    │      │
│  │             │  │             │  │             │      │
│  │ Bias        │  │ Data        │  │ Fair        │      │
│  │ Detection   │  │ Debiasing   │  │ Training    │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
│         │                │                │             │
│         └────────────────┼────────────────┘             │
│                          │                              │
│  ┌──────────────────────▼──────────────────────┐        │
│  │            Pipeline Orchestrator            │        │
│  │                                             │        │
│  │  1. Baseline measurement                    │        │
│  │  2. Data transformation & model training    │        │
│  │  3. Final validation & comparison           │        │
│  └─────────────────────────────────────────────┘        │
│                          │                              │
│  ┌──────────────────────▼──────────────────────┐        │
│  │         Configuration & MLflow              │        │
│  │                                             │        │
│  │  YAML config parsing                       │        │
│  │  MLflow metrics & model logging            │        │
│  └─────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

## Installation

```bash
git clone <repository-url>
cd fairness-pipeline-toolkit
uv sync
```

## Usage

1. Create configuration file:

```yaml
data:
  input_path: "data.csv"
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

2. Run pipeline:

```bash
python run_pipeline.py config.yml
```

## Components

### MeasurementModule
- `BiasDetector`: Detects bias in datasets and model predictions
- `FairnessMetrics`: Calculates fairness metrics (demographic parity, equalized odds)

### PipelineModule  
- `BiasMitigationTransformer`: Mitigates bias by adjusting feature distributions
- `BaseTransformer`: Abstract base for custom transformers

### TrainingModule
- `FairnessConstrainedClassifier`: Enforces fairness constraints during training
- `FairClassifier`: Abstract base for fair classifiers

## Configuration

### Required Parameters

- `data.input_path`: Path to CSV dataset
- `data.target_column`: Target column name
- `data.sensitive_features`: List of sensitive attribute columns
- `preprocessing.transformer.name`: Transformer class name
- `training.method.name`: Training method name
- `evaluation.primary_metric`: Primary fairness metric
- `mlflow.experiment_name`: MLflow experiment name

### Optional Parameters

- `data.test_size`: Train/test split ratio (default: 0.2)
- `data.random_state`: Random seed (default: 42)
- `evaluation.fairness_threshold`: Violation threshold (default: 0.1)
- `mlflow.log_model`: Whether to log model (default: true)

## Output

The pipeline generates:
1. Baseline bias audit report
2. Final model evaluation report
3. Improvement comparison metrics
4. MLflow experiment with logged metrics and models

## MLflow Integration

Automatically logs:
- Performance metrics (accuracy, precision, recall)
- Fairness metrics (demographic parity difference, equalized odds difference)
- Model artifacts
- Configuration files

View results: `mlflow ui`

## Development

```bash
uv sync --extra dev
uv run ruff check
uv run mypy src/
```

## License

MIT