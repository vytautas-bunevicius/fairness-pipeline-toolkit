#!/usr/bin/env python3
"""Pipeline orchestrator script for fairness pipeline toolkit.

This script provides the main entry point for executing end-to-end
fairness-aware machine learning pipelines.
"""

import sys
from pathlib import Path

from fairness_pipeline_toolkit.pipeline_executor import PipelineExecutor
from fairness_pipeline_toolkit.config import ConfigParser, setup_logging
import logging


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
        logger = setup_logging(level='INFO', console_output=True, structured=False)
        logger.info(f"Loading configuration from: {config_path}")
        config = ConfigParser.load(config_path)
        
        logger.info("Validating configuration...")
        errors = ConfigParser.validate(config)
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            sys.exit(1)
        
        logger.info("✓ Configuration validated")
        
        executor = PipelineExecutor(config, verbose=True)
        executor.execute_pipeline()
        
        logger.info("✓ Pipeline execution completed successfully")
        
    except Exception as e:
        logger = logging.getLogger('fairness_pipeline')
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()