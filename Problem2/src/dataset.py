"""
==============================================================================
Dataset Utilities for Character-Level Name Generation
==============================================================================
This module handles the character-level preprocessing pipeline:
  - Building a vocabulary from raw names (with special tokens)
  - Encoding/decoding names as integer sequences
  - Creating PyTorch Dataset and collation for batched training

Author: Arnesh Singh
Course: NLU Assignment 2
==============================================================================
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# ── Special Tokens ──────────────────────────────────────────────────────────
# PAD: padding for batching variable-length sequences
# SOS: start-of-sequence marker (signals the model to begin generating)
# EOS: end-of-sequence marker (signals the model to stop generating)
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"


class CharVocab:
    """
    Character-level vocabulary that maps individual characters to integer IDs.

    Given a list of names, it extracts every unique character and assigns
    a unique index. Three special tokens (PAD, SOS, EOS) are prepended so
    that PAD always maps to index 0 (required by nn.Embedding's padding_idx).

    Example:
        vocab = CharVocab(["aarav sharma", "priya patel"])
        encoded = vocab.encode("aarav")   # → [1, 4, 4, 17, 4, 20, 2]
        decoded = vocab.decode(encoded)   # → "aarav"
    """

    def __init__(self, names):
        # Collect and sort all unique characters across all names
        all_chars = sorted(set("".join(names)))

        # Build index-to-string and string-to-index mappings
        # Special tokens come first: PAD=0, SOS=1, EOS=2
        self.itos = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + all_chars
        self.stoi = {char: idx for idx, char in enumerate(self.itos)}

        # Store frequently-used indices as attributes for convenience
        self.pad_idx = self.stoi[PAD_TOKEN]  # 0
        self.sos_idx = self.stoi[SOS_TOKEN]  # 1
        self.eos_idx = self.stoi[EOS_TOKEN]  # 2

    def encode(self, name):
        """Convert a name string into a list of token indices: [SOS, c1, c2, ..., cn, EOS]."""
        return [self.sos_idx] + [self.stoi[ch] for ch in name] + [self.eos_idx]

    def decode(self, indices):
        """Convert a list of token indices back to a clean string (strips special tokens)."""
        chars = []
        for idx in indices:
            token = self.itos[idx]
            if token == EOS_TOKEN:
                break  # Stop at end-of-sequence
            if token not in (PAD_TOKEN, SOS_TOKEN):
                chars.append(token)
        return "".join(chars)

    def __len__(self):
        """Total vocabulary size (including special tokens)."""
        return len(self.itos)


class NamesDataset(Dataset):
    """
    PyTorch Dataset that pre-encodes all names into integer tensors.

    Each item is a 1-D LongTensor: [SOS, char1, char2, ..., charN, EOS].
    The collate_fn handles padding when creating batches.
    """

    def __init__(self, names, vocab):
        self.vocab = vocab
        # Pre-encode every name once (avoids repeated encoding during training)
        self.data = [torch.tensor(vocab.encode(name), dtype=torch.long) for name in names]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """
    Custom collation function for DataLoader.

    Takes a list of variable-length tensors and:
      1. Pads them to the longest sequence in the batch
      2. Splits into input/target pairs for teacher-forcing training

    Input sequence:  [SOS, c1, c2, ..., cn]       (drop last token)
    Target sequence: [c1,  c2, c3, ..., cn, EOS]   (drop first token)

    This way the model learns: given SOS → predict c1, given c1 → predict c2, etc.
    """
    padded = pad_sequence(batch, batch_first=True, padding_value=0)
    inputs = padded[:, :-1]   # Everything except the last token
    targets = padded[:, 1:]   # Everything except the first token
    return inputs, targets


def load_names(filepath):
    """
    Load names from a text file (one name per line).

    All names are lowercased for consistency — the model operates on
    lowercase characters only, which keeps the vocabulary small.
    """
    with open(filepath, "r") as f:
        names = [line.strip().lower() for line in f if line.strip()]
    return names
