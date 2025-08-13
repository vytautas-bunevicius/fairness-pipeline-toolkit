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
    DATA["Input Data<br/>CSV files with target and<br/>sensitive features"]
    CONFIG["config.yml<br/>Pipeline configuration<br/>Fairness parameters"]
    
    %% Orchestration
    EXECUTOR["Pipeline Executor<br/>run_pipeline.py"]
    
    %% Three-Step Process
    STEP1["Step 1: Baseline Measurement<br/>BiasDetector + FairnessMetrics<br/>Audit data and train baseline model"]
    STEP2["Step 2: Bias Mitigation<br/>BiasMitigationTransformer<br/>Preprocess data to reduce bias"]
    STEP3["Step 3: Fair Training<br/>FairnessConstrainedClassifier<br/>Train model with fairness constraints"]
    
    %% Integration
    MLFLOW["MLflow Integration<br/>Experiment tracking<br/>Model registry"]
    
    %% Output
    RESULTS["Results<br/>Fair model + improvement report<br/>Fairness metrics comparison"]
    
    %% Flow
    DATA --> EXECUTOR
    CONFIG --> EXECUTOR
    EXECUTOR --> STEP1
    STEP1 --> STEP2
    STEP2 --> STEP3
    STEP1 --> MLFLOW
    STEP2 --> MLFLOW
    STEP3 --> MLFLOW
    STEP3 --> RESULTS
    
    %% Styling
    classDef input fill:#E3F2FD,stroke:#1976D2,stroke-width:2px
    classDef process fill:#FFF3E0,stroke:#F57C00,stroke-width:2px
    classDef integration fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px
    classDef output fill:#E8F5E8,stroke:#388E3C,stroke-width:2px
    
    class DATA,CONFIG input
    class EXECUTOR,STEP1,STEP2,STEP3 process
    class MLFLOW integration
    class RESULTS output
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

- Python 3.13+ (or check .python-version)
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

BASELINE PREDICTION REPORT

     Performance Metrics      
                              
 Metric          Value
 ──────────────────────────── 
 Accuracy        0.9075
 Precision       0.8956
 Recall          0.9075


                           Fairness Metrics                            
                                                                       
 Metric                        Value     Status       Threshold
 ───────────────────────────────────────────────────────────────────── 
 Demographic Parity Difference  0.0487   ✅ OK           ≤ 0.1
 Equalized Odds Difference      0.4848   ❌ VIOLATION    ≤ 0.1

⚠️  1 fairness violations detected

FINAL PREDICTION REPORT

     Performance Metrics      
                              
 Metric          Value
 ──────────────────────────── 
 Accuracy        0.8950
 Precision       0.8824
 Recall          0.8950


                           Fairness Metrics                            
                                                                       
 Metric                        Value     Status       Threshold
 ───────────────────────────────────────────────────────────────────── 
 Demographic Parity Difference  0.1429   ❌ VIOLATION    ≤ 0.1
 Equalized Odds Difference      0.6667   ❌ VIOLATION    ≤ 0.1

⚠️  2 fairness violations detected

IMPROVEMENT COMPARISON

                   Performance Metrics Comparison                    
                                                                     
 Metric     Baseline    Final          Change           Status
 ─────────────────────────────────────────────────────────────────── 
 Accuracy    0.9075    0.8950     -0.0125 (-1.4%)    📉 Decreased

                      Fairness Metrics Comparison                      
                                                                       
 Metric                        Baseline   Final   Improvement   Status
 ─────────────────────────────────────────────────────────────────────
 🎯 Demographic Parity Difference  0.0487   0.1429   -0.0942     ❌ Worse
    (Primary)

Summary: Primary fairness goal not achieved (-0.0942 change)

Successfully registered model 'fairness_pipeline_model'.
```

## License

This software is released into the public domain. You are free to use, modify, and distribute this code without any restrictions. See the LICENSE file for full details.

The Unlicense promotes maximum freedom and removes all copyright restrictions, making this toolkit freely available for any use case, including commercial applications.