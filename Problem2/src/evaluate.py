"""
==============================================================================
Evaluation Script — Novelty Rate & Diversity Metrics
==============================================================================
Quantitatively evaluates generated names by computing:

  1. Novelty Rate  = (# generated names NOT in training set) / (total generated)
                     → Higher is better (model is creative, not memorizing)

  2. Diversity     = (# unique generated names) / (total generated)
                     → Higher is better (model doesn't repeat itself)

Author: Arnesh Singh
Course: NLU Assignment 2
==============================================================================
"""

import os


def load_lines(filepath):
    """Load non-empty lines from a file, lowercased and stripped."""
    with open(filepath, "r") as f:
        return [line.strip().lower() for line in f if line.strip()]


def novelty_rate(generated, training_set):
    """
    Compute the percentage of generated names that are NOT in the training set.

    A high novelty rate means the model is generating new, creative names
    rather than simply memorizing and reproducing training examples.
    """
    if not generated:
        return 0.0
    novel_count = sum(1 for name in generated if name not in training_set)
    return (novel_count / len(generated)) * 100


def diversity(generated):
    """
    Compute the ratio of unique names to total generated names.

    A diversity of 1.0 means every generated name is different (no duplicates).
    Lower values indicate the model is collapsing to repeated patterns.
    """
    if not generated:
        return 0.0
    return len(set(generated)) / len(generated)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_path = os.path.join(base_dir, "..", "data", "TrainingNames.txt")
    gen_dir = os.path.join(base_dir, "..", "generated")

    # Load training names into a set for fast lookup
    training_names = set(load_lines(training_path))
    print(f"  Training set: {len(training_names)} names\n")

    model_names = ["VanillaRNN", "BLSTM", "AttentionRNN"]

    # ── Results Table ───────────────────────────────────────────────────
    print(f"  {'Model':<15} {'Novelty Rate':>13} {'Diversity':>10}")
    print(f"  {'-'*42}")

    for model_name in model_names:
        gen_path = os.path.join(gen_dir, f"gen_{model_name}.txt")

        if not os.path.exists(gen_path):
            print(f"  {model_name:<15} {'(no file)':>13}")
            continue

        generated = load_lines(gen_path)
        nov = novelty_rate(generated, training_names)
        div = diversity(generated)

        print(f"  {model_name:<15} {nov:>12.1f}% {div:>10.4f}")

        # Show a few sample names for quick inspection
        print(f"    Samples: {generated[:5]}")
        novel_names = [n for n in generated if n not in training_names]
        print(f"    Novel: {len(novel_names)}/{len(generated)}\n")


if __name__ == "__main__":
    main()
