"""
==============================================================================
Training Script — Character-Level Name Generation Models
==============================================================================
Trains all three models (VanillaRNN, BLSTM, AttentionRNN) on the Indian
names dataset with the following safeguards against overfitting:

  - 80/20 train/validation split
  - Dropout regularization (applied in model architectures)
  - L2 weight decay in the optimizer
  - Early stopping when validation loss stops improving

Each model's best checkpoint (by validation loss) is saved to checkpoints/.

Author: Arnesh Singh
Course: NLU Assignment 2
==============================================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import load_names, CharVocab, NamesDataset, collate_fn
from models import get_model, count_parameters


# ── Hyperparameters ─────────────────────────────────────────────────────────
CONFIG = {
    "embed_dim": 64,       # Character embedding dimensionality
    "hidden_size": 128,    # RNN hidden state size
    "num_layers": 1,       # Number of stacked RNN layers
    "lr": 0.003,           # Learning rate for Adam optimizer
    "batch_size": 64,      # Mini-batch size
    "epochs": 50,          # Maximum training epochs
    "dropout": 0.3,        # Dropout probability
    "weight_decay": 1e-4,  # L2 regularization strength
    "patience": 15,        # Early stopping patience (epochs without improvement)
}


def compute_val_loss(model, loader, criterion, device, model_name):
    """
    Evaluate model on the validation set without updating weights.

    For BLSTM, we only compute loss on the bidirectional head (the one
    that's actually trained), matching what we do during training.
    """
    model.eval()
    total_loss, num_batches = 0.0, 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            if model_name == "BLSTM":
                # Compute loss on both heads: bidirectional (primary) + forward-only (auxiliary)
                logits_bi, logits_fwd, _ = model(inputs)
                loss_bi = criterion(logits_bi.reshape(-1, logits_bi.size(-1)),
                                    targets.reshape(-1))
                loss_fwd = criterion(logits_fwd.reshape(-1, logits_fwd.size(-1)),
                                     targets.reshape(-1))
                loss = loss_bi + loss_fwd
            else:
                logits, _ = model(inputs)
                loss = criterion(logits.reshape(-1, logits.size(-1)),
                                 targets.reshape(-1))

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def train_model(model_name, vocab, train_dataset, val_dataset, config, device):
    """
    Train a single model with validation tracking and early stopping.

    Steps:
        1. Initialize model, optimizer, and loss function
        2. For each epoch: train on mini-batches, evaluate on validation set
        3. Save checkpoint whenever validation loss improves
        4. Stop early if val loss hasn't improved for `patience` epochs
        5. Restore and save the best model weights
    """
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")

    # ── Initialize model ────────────────────────────────────────────────
    model = get_model(model_name, len(vocab),
                      embed_dim=config["embed_dim"],
                      hidden_size=config["hidden_size"],
                      num_layers=config["num_layers"],
                      dropout=config["dropout"]).to(device)

    num_params = count_parameters(model)
    print(f"  Trainable parameters: {num_params:,}")
    print(f"  Config: hidden={config['hidden_size']}, layers={config['num_layers']}, "
          f"lr={config['lr']}, dropout={config['dropout']}, embed={config['embed_dim']}")

    # ── Data loaders ────────────────────────────────────────────────────
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"],
                            shuffle=False, collate_fn=collate_fn)

    # ── Optimizer & loss ────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    # ── Training loop with early stopping ───────────────────────────────
    is_blstm = (model_name == "BLSTM")
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(1, config["epochs"] + 1):
        # — Train phase —
        model.train()
        epoch_loss, num_batches = 0.0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            if is_blstm:
                # Train both heads: bidirectional + forward-only
                # The forward-only head (fc_fwd) learns to predict from
                # forward-direction features, enabling generation
                logits_bi, logits_fwd, _ = model(inputs)
                loss_bi = criterion(logits_bi.reshape(-1, logits_bi.size(-1)),
                                    targets.reshape(-1))
                loss_fwd = criterion(logits_fwd.reshape(-1, logits_fwd.size(-1)),
                                     targets.reshape(-1))
                loss = loss_bi + loss_fwd
            else:
                logits, _ = model(inputs)
                loss = criterion(logits.reshape(-1, logits.size(-1)),
                                 targets.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        # — Validation phase —
        train_loss = epoch_loss / num_batches
        val_loss = compute_val_loss(model, val_loader, criterion, device, model_name)

        # Print progress every 10 epochs (and the first epoch)
        if epoch % 10 == 0 or epoch == 1:
            gap = val_loss - train_loss
            print(f"  Epoch {epoch:3d}/{config['epochs']}  "
                  f"Train: {train_loss:.4f}  Val: {val_loss:.4f}  Gap: {gap:+.4f}")

        # — Early stopping check —
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"  ⏹ Early stopping at epoch {epoch} "
                      f"(best val: {best_val_loss:.4f})")
                break

    # ── Restore best model and save checkpoint ──────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}.pth")

    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_itos": vocab.itos,
        "vocab_stoi": vocab.stoi,
        "config": config,
        "model_name": model_name,
        "n_params": num_params,
    }, ckpt_path)

    print(f"  ✓ Best val loss: {best_val_loss:.4f}")
    print(f"  ✓ Saved → {ckpt_path}")
    return model, num_params


def main():
    """Main entry point: load data, split, and train all three models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load dataset ────────────────────────────────────────────────────
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "TrainingNames.txt")
    names = load_names(data_path)
    vocab = CharVocab(names)
    full_dataset = NamesDataset(names, vocab)
    print(f"Loaded {len(names)} names | Vocab: {len(vocab)} chars")

    # ── 80/20 train/validation split ────────────────────────────────────
    indices = list(range(len(full_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    split_point = int(0.8 * len(indices))

    train_ds = Subset(full_dataset, indices[:split_point])
    val_ds = Subset(full_dataset, indices[split_point:])
    print(f"Split: {len(train_ds)} train / {len(val_ds)} val")

    # ── Train each model ────────────────────────────────────────────────
    model_names = ["VanillaRNN", "BLSTM", "AttentionRNN"]
    results = {}

    for name in model_names:
        model, num_params = train_model(name, vocab, train_ds, val_ds, CONFIG, device)
        results[name] = num_params

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Training Complete")
    print(f"{'='*60}")
    print(f"  {'Model':<15} {'Parameters':>12}")
    print(f"  {'-'*28}")
    for name, params in results.items():
        print(f"  {name:<15} {params:>12,}")


if __name__ == "__main__":
    main()
