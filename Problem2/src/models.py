"""
==============================================================================
Character-Level RNN Models for Indian Name Generation
==============================================================================
Three recurrent architectures implemented from scratch using PyTorch:

  1. VanillaRNN    — Standard Elman RNN (baseline)
  2. CharBLSTM     — Bidirectional LSTM (demonstrates train/generate mismatch)
  3. AttentionRNN  — RNN with Bahdanau additive attention

Each model includes:
  - Dropout regularization on embeddings and hidden outputs
  - A `generate()` method for autoregressive name sampling
  - Temperature-controlled softmax for diversity tuning

Author: Arnesh Singh
Course: NLU Assignment 2
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Model 1: Vanilla RNN                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class VanillaRNN(nn.Module):
    """
    Standard character-level Elman RNN.

    Architecture:
        Input char → Embedding(V, E) → Dropout → RNN(E, H) → Dropout → Linear(H, V)

    This is the simplest recurrent baseline. At each timestep, the RNN
    takes the current character embedding and the previous hidden state
    to produce the next hidden state, which is then projected to vocabulary
    logits for next-character prediction.

    Args:
        vocab_size: Number of unique tokens (characters + special tokens)
        embed_dim:  Dimensionality of character embeddings
        hidden_size: Number of units in the RNN hidden state
        num_layers: Number of stacked RNN layers
        dropout:    Dropout probability for regularization
    """

    def __init__(self, vocab_size, embed_dim=64, hidden_size=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Learnable character embeddings (PAD token at index 0 stays zero)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Dropout applied after embedding and after RNN output
        self.drop = nn.Dropout(dropout)

        # Core recurrent layer
        self.rnn = nn.RNN(embed_dim, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)

        # Output projection: hidden state → character logits
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forward pass for training (teacher-forced).

        Args:
            x: Input token indices, shape (batch_size, seq_len)
            hidden: Optional initial hidden state

        Returns:
            logits: Raw predictions, shape (batch_size, seq_len, vocab_size)
            hidden: Final hidden state
        """
        emb = self.drop(self.embedding(x))    # (B, T, E)
        out, hidden = self.rnn(emb, hidden)   # (B, T, H)
        out = self.drop(out)
        logits = self.fc(out)                 # (B, T, V)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        """Create a zero-initialized hidden state."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    @torch.no_grad()
    def generate(self, vocab, max_len=50, temperature=0.8):
        """
        Generate a single name by sampling one character at a time.

        Process:
            1. Start with SOS token
            2. Feed through the model to get next-character probabilities
            3. Sample from the distribution (controlled by temperature)
            4. Repeat until EOS is produced or max_len is reached

        Higher temperature → more random/creative names
        Lower temperature  → more conservative/common names
        """
        self.eval()
        device = next(self.parameters()).device
        hidden = self.init_hidden(1, device)
        current_token = torch.tensor([[vocab.sos_idx]], device=device)
        generated_chars = []

        for _ in range(max_len):
            logits, hidden = self.forward(current_token, hidden)

            # Scale logits by temperature before softmax
            scaled_logits = logits[:, -1, :] / temperature
            probs = F.softmax(scaled_logits, dim=-1)

            # Sample next character from the probability distribution
            next_token = torch.multinomial(probs, 1)

            if next_token.item() == vocab.eos_idx:
                break  # Model decided the name is complete

            generated_chars.append(vocab.itos[next_token.item()])
            current_token = next_token

        return "".join(generated_chars)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Model 2: Bidirectional LSTM (CharBLSTM)                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class CharBLSTM(nn.Module):
    """
    Bidirectional LSTM for character-level modeling.

    Architecture:
        Input char → Embedding(V, E) → Dropout → BiLSTM(E, H) → Dropout
            → Linear(2H, V)   [bidirectional head — trained with full context]
            → Linear(H, V)    [forward-only head — trained for generation]

    ARCHITECTURAL NOTE (Train/Generate Mismatch):
        During training, the BiLSTM processes the ENTIRE sequence in both
        directions. The bidirectional head (fc) benefits from seeing future
        characters via the backward pass. However, during generation we can
        only go left-to-right (autoregressive), so the backward context is
        unavailable. A separate forward-only head (fc_fwd) is trained
        alongside fc to predict from forward-direction features only. Despite
        being trained, fc_fwd receives hidden states that were optimized with
        bidirectional gradients, creating a natural train/generate mismatch
        that typically results in lower generation quality compared to
        purely unidirectional models.
    """

    def __init__(self, vocab_size, embed_dim=64, hidden_size=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.drop = nn.Dropout(dropout)

        # Bidirectional LSTM: outputs have 2*hidden_size dimensions
        # (concatenation of forward and backward hidden states)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)

        # Training head: uses full bidirectional output (2H → V)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

        # Forward-only head: uses only forward direction (H → V)
        # This head IS trained (receives gradients) and is used for generation
        self.fc_fwd = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forward pass returning both bidirectional and forward-only logits.

        Returns:
            logits_bi:  Bidirectional logits  (B, T, V) — primary training loss
            logits_fwd: Forward-only logits   (B, T, V) — auxiliary training loss + generation
            hidden:     Final LSTM hidden state
        """
        emb = self.drop(self.embedding(x))
        out, hidden = self.lstm(emb, hidden)           # out: (B, T, 2*H)
        out_dropped = self.drop(out)

        # Bidirectional logits (primary training objective)
        logits_bi = self.fc(out_dropped)               # (B, T, V)

        # Forward-only logits (auxiliary objective — trained for generation)
        # The first H dimensions of BiLSTM output are the forward direction
        out_fwd = out_dropped[:, :, :self.hidden_size]  # (B, T, H)
        logits_fwd = self.fc_fwd(out_fwd)               # (B, T, V)

        return logits_bi, logits_fwd, hidden

    def init_hidden(self, batch_size, device):
        """Create zero-initialized hidden and cell states for both directions."""
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=device)
        return (h0, c0)

    @torch.no_grad()
    def generate(self, vocab, max_len=50, temperature=0.8):
        """
        Autoregressive generation using only the forward LSTM direction.

        Extracts forward-direction weights from the BiLSTM and runs them
        as a unidirectional LSTMCell. The output is projected through the
        trained fc_fwd head. Despite fc_fwd being trained, the forward
        hidden states were learned in a bidirectional context, so generation
        quality naturally degrades compared to purely unidirectional models.
        """
        self.eval()
        device = next(self.parameters()).device

        # Extract forward-direction weights into a standalone LSTMCell
        lstm_fw = nn.LSTMCell(self.embedding.embedding_dim, self.hidden_size).to(device)
        lstm_fw.weight_ih.data.copy_(self.lstm.weight_ih_l0.data)
        lstm_fw.weight_hh.data.copy_(self.lstm.weight_hh_l0.data)
        lstm_fw.bias_ih.data.copy_(self.lstm.bias_ih_l0.data)
        lstm_fw.bias_hh.data.copy_(self.lstm.bias_hh_l0.data)

        # Initialize hidden and cell states to zero
        hx = torch.zeros(1, self.hidden_size, device=device)
        cx = torch.zeros(1, self.hidden_size, device=device)
        current_token = torch.tensor([vocab.sos_idx], device=device)
        generated_chars = []

        for _ in range(max_len):
            emb = self.embedding(current_token)        # (1, E)
            hx, cx = lstm_fw(emb, (hx, cx))            # Step through one timestep
            logits = self.fc_fwd(hx) / temperature     # Project through TRAINED head
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze(1)

            if next_token.item() == vocab.eos_idx:
                break

            generated_chars.append(vocab.itos[next_token.item()])
            current_token = next_token

        return "".join(generated_chars)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Model 3: RNN with Bahdanau (Additive) Attention                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism.

    Computes attention scores using a learned alignment model:
        score(query, key) = v^T · tanh(W_q · query + W_k · key)

    This allows each output timestep to "look back" at all previous hidden
    states and dynamically focus on the most relevant ones.

    Unlike dot-product attention, additive attention can learn non-linear
    alignment patterns between queries and keys.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)  # Query projection
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)  # Key projection
        self.v = nn.Linear(hidden_size, 1, bias=False)              # Score reduction

    def forward(self, query, keys):
        """
        Compute attention-weighted context vector.

        Args:
            query: Current hidden state  (B, 1, H)
            keys:  All past hidden states (B, T, H)

        Returns:
            context:      Weighted sum of keys  (B, 1, H)
            attn_weights: Attention distribution (B, 1, T)
        """
        # Compute alignment scores
        score = self.v(torch.tanh(self.W_q(query) + self.W_k(keys)))  # (B, T, 1)

        # Normalize scores into a probability distribution
        attn_weights = F.softmax(score, dim=1).transpose(1, 2)         # (B, 1, T)

        # Compute context as weighted combination of all keys
        context = torch.bmm(attn_weights, keys)                        # (B, 1, H)

        return context, attn_weights


class AttentionRNN(nn.Module):
    """
    RNN augmented with Bahdanau (Additive) Attention.

    Architecture:
        Input char → Embedding(V, E) → Dropout → RNN(E, H) → Dropout
            → BahdanauAttention over past hidden states
            → Concat[hidden, context] → Linear(2H, V)

    At each timestep, the model:
        1. Runs the RNN to produce a hidden state
        2. Uses attention to compute a context vector from ALL past hidden states
        3. Concatenates the current hidden state with the context
        4. Projects the concatenation to vocabulary logits

    The attention is CAUSAL: each position can only attend to itself and
    earlier positions, making it suitable for autoregressive generation.
    """

    def __init__(self, vocab_size, embed_dim=64, hidden_size=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.RNN(embed_dim, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.attention = BahdanauAttention(hidden_size)

        # Output projection: concatenated [hidden + context] → logits
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forward pass with causal self-attention.

        For each position t, attention is computed over positions [0, 1, ..., t]
        (not future positions), preserving the autoregressive property.
        """
        emb = self.drop(self.embedding(x))
        rnn_out, hidden = self.rnn(emb, hidden)  # (B, T, H)
        rnn_out = self.drop(rnn_out)

        B, T, H = rnn_out.shape
        logits_list = []

        for t in range(T):
            query = rnn_out[:, t:t+1, :]       # Current timestep:  (B, 1, H)
            keys = rnn_out[:, :t+1, :]          # All past timesteps: (B, t+1, H)

            # Attend over past hidden states to get context
            context, _ = self.attention(query, keys)  # (B, 1, H)

            # Concatenate current hidden state with attention context
            combined = torch.cat([query, context], dim=-1)  # (B, 1, 2H)
            logits_list.append(self.fc(combined))

        logits = torch.cat(logits_list, dim=1)  # (B, T, V)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        """Create a zero-initialized hidden state."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    @torch.no_grad()
    def generate(self, vocab, max_len=50, temperature=0.8):
        """
        Generate a name autoregressively with attention.

        Unlike the forward pass (which processes the full sequence at once),
        generation runs one step at a time, accumulating hidden states for
        the attention mechanism to look back at.
        """
        self.eval()
        device = next(self.parameters()).device
        hidden = self.init_hidden(1, device)
        current_token = torch.tensor([[vocab.sos_idx]], device=device)
        generated_chars = []
        past_hidden_states = []  # Accumulates hidden states for attention

        for _ in range(max_len):
            emb = self.embedding(current_token)            # (1, 1, E)
            out, hidden = self.rnn(emb, hidden)            # (1, 1, H)
            past_hidden_states.append(out)

            # Attend over all hidden states generated so far
            keys = torch.cat(past_hidden_states, dim=1)    # (1, t, H)
            context, _ = self.attention(out, keys)          # (1, 1, H)

            # Combine current state with context for prediction
            combined = torch.cat([out, context], dim=-1)    # (1, 1, 2H)
            logits = self.fc(combined).squeeze(1) / temperature  # (1, V)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)        # (1, 1)

            if next_token.item() == vocab.eos_idx:
                break

            generated_chars.append(vocab.itos[next_token.item()])
            current_token = next_token

        return "".join(generated_chars)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Utility Functions                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def count_parameters(model):
    """Count the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(name, vocab_size, **kwargs):
    """
    Factory function to instantiate a model by name.

    Args:
        name: One of "VanillaRNN", "BLSTM", "AttentionRNN"
        vocab_size: Size of the character vocabulary
        **kwargs: Passed to the model constructor (embed_dim, hidden_size, etc.)
    """
    model_registry = {
        "VanillaRNN": VanillaRNN,
        "BLSTM": CharBLSTM,
        "AttentionRNN": AttentionRNN,
    }
    if name not in model_registry:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(model_registry.keys())}")
    return model_registry[name](vocab_size, **kwargs)
