"""Command line interface for fairness pipeline toolkit."""

import argparse
import sys
from pathlib import Path

from .pipeline_executor import PipelineExecutor
from .config import ConfigParser


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fairness Pipeline Development Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "config_path", type=str, help="Path to pipeline configuration YAML file"
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration without running pipeline",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    try:
        config_path = Path(args.config_path)
        print(f"Loading configuration from: {config_path}")

        config = ConfigParser.load(config_path)

        errors = ConfigParser.validate(config)
        if errors:
            print("Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)

        print("Configuration validated successfully")

        if args.validate_only:
            print("Validation complete. Exiting.")
            return

        executor = PipelineExecutor(config, verbose=args.verbose)
        executor.execute_pipeline()

        print("Pipeline execution completed successfully!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
