#!/usr/bin/env python3
"""Pipeline orchestrator script for fairness pipeline toolkit.

This script provides the main entry point for executing end-to-end
fairness-aware machine learning pipelines.
"""

import sys
from pathlib import Path

# Add src to Python path for local development
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from fairness_pipeline_toolkit.pipeline_executor import PipelineExecutor
from fairness_pipeline_toolkit.config import ConfigParser


def main():
    """Main orchestrator function."""
    if len(sys.argv) != 2:
        print("Usage: python run_pipeline.py <config_file.yml>")
        sys.exit(1)
    
    config_path = Path(sys.argv[1])
    
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        print("=" * 60)
        print("FAIRNESS PIPELINE DEVELOPMENT TOOLKIT")
        print("=" * 60)
        
        # Load configuration
        print(f"Loading configuration from: {config_path}")
        config = ConfigParser.load(config_path)
        
        # Validate configuration
        print("Validating configuration...")
        errors = ConfigParser.validate(config)
        if errors:
            print("Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
        
        print("Configuration validated successfully")
        
        # Initialize and run pipeline
        executor = PipelineExecutor(config, verbose=True)
        executor.execute_pipeline()
        
        print("=" * 60)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()