#!/usr/bin/env python3
"""
Project structure consolidation script.
Moves and copies files to their proper locations in the project structure.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("consolidation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def setup_directory_structure(base: Path, structure: Dict[Path, List[Union[str, Path]]]) -> None:
    """Create target directories if they don't exist."""
    for path in structure.keys():
        try:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")


def process_files(base: Path, structure: Dict[Path, List[Union[str, Path]]]) -> None:
    """Move or copy files to their target locations."""
    for target_dir, files in structure.items():
        for file in files:
            src_path = base / file if isinstance(file, str) else file
            if not src_path.exists():
                logger.warning(f"Source file not found: {src_path}")
                continue

            try:
                dest_path = target_dir / src_path.name
                if "artifacts" in str(target_dir):
                    shutil.copy2(src_path, dest_path)
                    logger.info(f"Copied: {src_path} → {dest_path}")
                else:
                    shutil.move(str(src_path), str(dest_path))
                    logger.info(f"Moved: {src_path} → {dest_path}")
            except Exception as e:
                logger.error(f"Error processing {src_path}: {e}")


def cleanup_directories(base: Path, dirs_to_remove: List[str]) -> None:
    """Remove specified directories if they exist."""
    for dir_name in dirs_to_remove:
        dir_path = base / dir_name
        try:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                logger.info(f"Deleted directory: {dir_path}")
        except Exception as e:
            logger.error(f"Could not delete {dir_path}: {e}")


def consolidate_keno_package(base: Path) -> None:
    """Consolidate files from keno_analyzer/ into src/keno/ if they exist."""
    keno_analyzer = base / "keno_analyzer"
    if keno_analyzer.exists():
        try:
            # Copy any unique files from keno_analyzer to src/keno
            for item in keno_analyzer.glob("**/*"):
                if item.is_file():
                    rel_path = item.relative_to(keno_analyzer)
                    dest_path = base / "src" / "keno" / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest_path)
                    logger.info(f"Copied from keno_analyzer: {rel_path}")
        except Exception as e:
            logger.error(f"Error consolidating keno_analyzer: {e}")


def main():
    """Main consolidation function."""
    try:
        # Define base paths
        base = Path(__file__).resolve().parent.parent
        src = base / "src" / "keno"
        artifacts = base / "artifacts"

        # Define paths to create and their contents
        structure = {
            src / "prediction": ["predictor.py", "model_trainer.py"],
            src
            / "data": [f for f in os.listdir(base) if f.startswith("data_") and f.endswith(".py")],
            src / "scripts": ["install.py", "run_tests.py"],
            src / "cli": ["keno-cli.py"],
            artifacts / "data": [f for f in base.glob("*.csv")] + list(base.glob("*.json")),
            artifacts / "visualizations": list(base.glob("*.png")),
            artifacts / "logs": list(base.glob("*.log")),
        }

        # Create target directories
        setup_directory_structure(base, structure)

        # Consolidate keno_analyzer package if it exists
        consolidate_keno_package(base)

        # Move or copy files into structure
        process_files(base, structure)

        # Clean up known unused directories
        cleanup_dirs = [
            "keno_analyzer",
            "keno_data",
            "analysis_output",
            "b2b-solution-finder",
            "backend",
            "strategy_analysis",
            "visualizations",
            "reports",
            "models",
            "logs",
            "data",
            "temp_staging",
        ]
        cleanup_directories(base, cleanup_dirs)

        logger.info("Project structure consolidation completed successfully")

    except Exception as e:
        logger.error(f"Fatal error during consolidation: {e}")
        raise


if __name__ == "__main__":
    main()
