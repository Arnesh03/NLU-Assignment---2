"""
==============================================================================
Name Generation Script — Generate Names from Trained Models
==============================================================================
Loads each trained model checkpoint and generates 100 names using
temperature-controlled autoregressive sampling.

Output is saved to generated/gen_<ModelName>.txt (one name per line).

Author: Arnesh Singh
Course: NLU Assignment 2
==============================================================================
"""

import os
import torch
from dataset import CharVocab
from models import get_model


def load_model(model_name, ckpt_dir, device):
    """
    Load a trained model from its checkpoint file.

    Reconstructs the vocabulary and model architecture from saved metadata,
    then loads the trained weights.

    Returns:
        model: The model in eval mode, ready for generation
        vocab: The character vocabulary used during training
        n_params: Number of trainable parameters
    """
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}.pth")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Reconstruct vocabulary from saved mappings
    vocab = CharVocab.__new__(CharVocab)
    vocab.itos = checkpoint["vocab_itos"]
    vocab.stoi = checkpoint["vocab_stoi"]
    vocab.pad_idx = vocab.stoi["<PAD>"]
    vocab.sos_idx = vocab.stoi["<SOS>"]
    vocab.eos_idx = vocab.stoi["<EOS>"]

    # Rebuild model with the same architecture and load weights
    config = checkpoint["config"]
    model = get_model(model_name, len(vocab),
                      embed_dim=config["embed_dim"],
                      hidden_size=config["hidden_size"],
                      num_layers=config["num_layers"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, vocab, checkpoint["n_params"]


def generate_names(model, vocab, num_names=100, temperature=0.8):
    """
    Generate `num_names` names from a trained model.

    Temperature controls randomness:
      - temp < 1.0 → more conservative, common patterns
      - temp = 1.0 → sample directly from learned distribution
      - temp > 1.0 → more random, creative outputs
    """
    names = []
    for _ in range(num_names):
        name = model.generate(vocab, max_len=50, temperature=temperature)
        names.append(name)
    return names


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(base_dir, "..", "checkpoints")
    out_dir = os.path.join(base_dir, "..", "generated")
    os.makedirs(out_dir, exist_ok=True)

    model_names = ["VanillaRNN", "BLSTM", "AttentionRNN"]

    for model_name in model_names:
        print(f"\n  Generating with {model_name}...")
        model, vocab, n_params = load_model(model_name, ckpt_dir, device)
        names = generate_names(model, vocab, num_names=100, temperature=0.8)

        # Save generated names to file
        out_path = os.path.join(out_dir, f"gen_{model_name}.txt")
        with open(out_path, "w") as f:
            for name in names:
                f.write(name + "\n")

        print(f"  ✓ {len(names)} names → {out_path}")
        print(f"  Samples: {names[:5]}")

    print("\n  Done! Generated names saved to generated/")


if __name__ == "__main__":
    main()
