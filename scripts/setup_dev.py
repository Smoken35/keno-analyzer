#!/usr/bin/env python3
"""Setup script for development environment."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command, cwd=None):
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(
            command, shell=True, check=True, cwd=cwd, capture_output=True, text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error: {e.stderr}")
        sys.exit(1)


def setup_dev_environment():
    """Set up the development environment."""
    # Create virtual environment if it doesn't exist
    if not Path("venv").exists():
        print("Creating virtual environment...")
        run_command("python -m venv venv")

    # Activate virtual environment and install dependencies
    if sys.platform == "win32":
        activate_cmd = ".\\venv\\Scripts\\activate"
    else:
        activate_cmd = "source ./venv/bin/activate"

    print("Installing dependencies...")
    run_command(f"{activate_cmd} && pip install -r requirements.txt")

    # Install pre-commit hooks
    print("Installing pre-commit hooks...")
    run_command(f"{activate_cmd} && pre-commit install")

    # Run initial formatting
    print("Running initial code formatting...")
    run_command(f"{activate_cmd} && black .")
    run_command(f"{activate_cmd} && isort .")

    print("\nDevelopment environment setup complete!")
    print("\nTo activate the virtual environment:")
    if sys.platform == "win32":
        print("    .\\venv\\Scripts\\activate")
    else:
        print("    source ./venv/bin/activate")

    print("\nTo run tests:")
    print("    pytest")

    print("\nTo check code quality:")
    print("    pre-commit run --all-files")


if __name__ == "__main__":
    setup_dev_environment()
