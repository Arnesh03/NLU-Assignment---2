"""
==============================================================================
Run All — Complete Pipeline Entry Point
==============================================================================
Runs the entire pipeline end-to-end in one command:

    python run_all.py

Steps:
    1. Generate dataset (if TrainingNames.txt doesn't exist)
    2. Train all three models (VanillaRNN, BLSTM, AttentionRNN)
    3. Generate 100 names from each trained model
    4. Evaluate novelty and diversity metrics

Author: Arnesh Singh
Course: NLU Assignment 2
==============================================================================
"""

import os
import subprocess
import sys


def run_script(script_name):
    """Run a Python script and exit if it fails."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(base_dir, script_name)

    print(f"\n{'━'*60}")
    print(f"  Running: {script_name}")
    print(f"{'━'*60}\n")

    result = subprocess.run([sys.executable, script_path], cwd=base_dir)
    if result.returncode != 0:
        print(f"\n  ✗ {script_name} failed (exit code {result.returncode})")
        sys.exit(1)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Step 1: Dataset generation (skip if already exists)
    data_path = os.path.join(base_dir, "..", "data", "TrainingNames.txt")
    if not os.path.exists(data_path):
        run_script("generate_dataset.py")
    else:
        print(f"\n  ✓ Dataset already exists: {data_path}")

    # Step 2: Train all models
    run_script("train.py")

    # Step 3: Generate names from trained models
    run_script("generate.py")

    # Step 4: Evaluate generated names
    run_script("evaluate.py")

    print(f"\n{'━'*60}")
    print("  ✓ Pipeline complete!")
    print(f"{'━'*60}\n")


if __name__ == "__main__":
    main()
