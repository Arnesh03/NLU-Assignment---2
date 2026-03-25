"""
==============================================================================
Word2Vec From Scratch — PyTorch Implementation
==============================================================================
Welcome! This script builds the entire Problem 1 pipeline from the ground up:
  - Task 1: Preprocess the scraped corpus (tokenize, lowecase, aggressively clean)
  - Task 2: Train CBOW and Skip-gram Word2Vec models from scratch using PyTorch
  - Task 3: Perform Semantic analysis (nearest neighbors, linear analogies)
  - Task 4: Visualize embeddings (PCA and t-SNE) colored by category

Author: Arnesh Singh
Course: NLU Assignment 2 — Problem 1
==============================================================================
"""

import os
import re
import string
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import random

# ==============================================================================
# SET SEEDS FOR REPRODUCIBILITY
# ==============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TASK 1: DATASET PREPARATION                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

STOPWORDS = set("""
    a an the is are was were be been being have has had do does did will would
    shall should may might can could am is are was were to of in for on with at
    by from as into through during before after above below between under
    and but or nor not no so yet both either neither each every all any few
    more most other some such than too very also just about up out if then
    that this these those it its he she they them their his her we our you
    your i me my which who whom whose what when where how why there here
    said mr mrs dr prof st etc vs via e g ie ii iii iv v vi vii viii ix x
""".split())

def load_corpus(data_dir):
    documents = []

    # Load main corpus (academic regulations)
    corpus_path = os.path.join(data_dir, "raw_corpus.txt")
    if os.path.exists(corpus_path):
        with open(corpus_path, "r") as f:
            text = f.read()
        documents.append(("Academic Regulations", text))
        print(f"  ✓ Loaded Academic Regulations ({len(text):,} chars)")

    # Load department/research text
    dept_path = os.path.join(data_dir, "departments_text.txt")
    if os.path.exists(dept_path):
        with open(dept_path, "r") as f:
            content = f.read()
        sections = re.split(r"---SOURCE:\s*(.+?)---", content)
        for i in range(1, len(sections), 2):
            source_name = sections[i].strip()
            text = sections[i + 1].strip()
            if len(text) > 50:
                documents.append((source_name, text))
        print(f"  ✓ Loaded CSE Department pages ({len(sections)//2} sections)")

    # Load any additional .txt files
    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            if fpath not in (corpus_path, dept_path) and fname.endswith(".txt"):
                with open(fpath, "r") as f:
                    text = f.read()
                if len(text) > 100:
                    documents.append((fname, text))
    return documents

def preprocess_text(text):
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\b\d{5,}\b", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = text.lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"--+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    clean_tokens = []
    for token in tokens:
        token = token.strip("-")
        if (len(token) > 1 and token not in STOPWORDS
                and not token.isdigit() and any(c.isalpha() for c in token)):
            clean_tokens.append(token)
    return clean_tokens

def build_corpus(documents):
    all_sentences = []
    all_tokens = []
    for source_name, text in documents:
        raw_sentences = re.split(r"[.\n;]+", text)
        for sent in raw_sentences:
            tokens = preprocess_text(sent)
            if len(tokens) >= 3:
                all_sentences.append(tokens)
                all_tokens.extend(tokens)
    return all_sentences, all_tokens

def report_statistics(documents, sentences, tokens, output_dir):
    vocab = set(tokens)
    freq = Counter(tokens)
    top_words = freq.most_common(20)

    stats_out = []
    stats_out.append("="*50)
    stats_out.append("DATASET STATISTICS")
    stats_out.append("="*50)
    stats_out.append(f"Documents:      {len(documents)}")
    stats_out.append(f"Sentences:      {len(sentences):,}")
    stats_out.append(f"Total tokens:   {len(tokens):,}")
    stats_out.append(f"Vocabulary:     {len(vocab):,}")
    stats_out.append("\nTop 20 words:")
    for word, count in top_words:
        stats_out.append(f"  {word:<20} {count:>6}")
        
    stats_text = "\n".join(stats_out)
    
    print("\n  " + stats_text.replace("\n", "\n  "))
    
    stats_path = os.path.join(output_dir, "..", "data", "dataset_statistics.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(stats_text + "\n")
    print(f"\n  ✓ Statistics saved → {stats_path}")

    wordcloud = WordCloud(
        width=1200, height=600, background_color="white",
        max_words=100, colormap="viridis", contour_width=2,
    ).generate_from_frequencies(freq)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud — IIT Jodhpur Corpus", fontsize=16, fontweight="bold")
    wc_path = os.path.join(output_dir, "wordcloud.png")
    plt.tight_layout()
    plt.savefig(wc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Word cloud saved → {wc_path}")
    return freq


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TASK 2: VOCABULARY, DATASETS & NEURAL MODELS                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class Vocabulary:
    def __init__(self, sentences, min_count=1):
        self.word_freq = Counter()
        for sent in sentences:
            self.word_freq.update(sent)
        self.word2idx = {}
        self.idx2word = {}
        idx = 0
        for word, count in self.word_freq.items():
            if count >= min_count:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        self.vocab_size = len(self.word2idx)
        freqs = np.zeros(self.vocab_size)
        for word, idx in self.word2idx.items():
            freqs[idx] = self.word_freq[word]
        freqs = np.power(freqs, 0.75)
        self.noise_dist = freqs / freqs.sum()

    def __len__(self):
        return self.vocab_size

class CBOWDataset(Dataset):
    def __init__(self, sentences, vocab, window_size=5):
        self.data = []
        for sent in sentences:
            indices = [vocab.word2idx[w] for w in sent if w in vocab.word2idx]
            for i in range(window_size, len(indices) - window_size):
                context = indices[i - window_size:i] + indices[i + 1:i + window_size + 1]
                target = indices[i]
                self.data.append((context, target))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

class SkipGramDataset(Dataset):
    def __init__(self, sentences, vocab, window_size=5):
        self.data = []
        for sent in sentences:
            indices = [vocab.word2idx[w] for w in sent if w in vocab.word2idx]
            for i in range(len(indices)):
                for j in range(max(0, i - window_size), min(len(indices), i + window_size + 1)):
                    if i != j:
                        self.data.append((indices[i], indices[j]))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        target, context = self.data[idx]
        return torch.tensor(target, dtype=torch.long), torch.tensor(context, dtype=torch.long)

class CBOWModelNN(nn.Module):
    """
    🧠 THE CBOW ARCHITECTURE:
    Continuous Bag of Words tries to predict a missing "Target" word using the 
    words surrounding it (the "Context").
    Instead of a massive, slow Softmax over all 2000+ words, we use Negative Sampling:
    We pull the target word closer to the context, and push a randomly chosen 
    set of "noise" words farther away.
    """
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # in_embed is used for the Context words
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        # out_embed is used for Target/Noise words
        self.out_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Initialize with Xavier Uniform to help gradients flow quickly
        nn.init.xavier_uniform_(self.in_embed.weight)
        nn.init.xavier_uniform_(self.out_embed.weight)

    def forward(self, context_ids, pos_ids, neg_ids):
        # 1. Average the context words to get a single summary vector for the window
        context_embeds = self.in_embed(context_ids).mean(dim=1)
        
        # 2. Score the actual true target word (we want this score to be HIGH)
        pos_embeds = self.out_embed(pos_ids)
        pos_score = (context_embeds * pos_embeds).sum(dim=1)
        # Binary Cross Entropy (maximize log sigmoid)
        pos_loss = -torch.nn.functional.logsigmoid(pos_score)
        
        # 3. Score the fake/noise words (we want this score to be LOW -> negative)
        neg_embeds = self.out_embed(neg_ids)
        # bmm = Batch Matrix Multiply (a fast way to dot-product everything at once)
        neg_score = torch.bmm(neg_embeds, context_embeds.unsqueeze(2)).squeeze(2)
        # Maximize log sigmoid of the NEGATED score
        neg_loss = -torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)
        
        # 4. Total Loss is the sum of both objectives
        return (pos_loss + neg_loss).mean()

class SkipGramModelNN(nn.Module):
    """
    🧠 THE SKIP-GRAM ARCHITECTURE:
    Skip-gram flips CBOW upside down. We take a single "Target" word and try 
    to predict all the "Context" words that usually surround it.
    This model generally performs better on smaller datasets and rare words,
    which is why it out-performs CBOW on our specific IITJ corpus!
    """
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)
        
        nn.init.xavier_uniform_(self.in_embed.weight)
        nn.init.xavier_uniform_(self.out_embed.weight)

    def forward(self, target_ids, pos_ids, neg_ids):
        # 1. The input is just the single center target word
        target_embeds = self.in_embed(target_ids)
        
        # 2. Pull the true context words closer
        pos_embeds = self.out_embed(pos_ids)
        pos_score = (target_embeds * pos_embeds).sum(dim=1)
        pos_loss = -torch.nn.functional.logsigmoid(pos_score)
        
        # 3. Push the random fake noise words away
        neg_embeds = self.out_embed(neg_ids)
        neg_score = torch.bmm(neg_embeds, target_embeds.unsqueeze(2)).squeeze(2)
        neg_loss = -torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)
        
        return (pos_loss + neg_loss).mean()

class ScratchWord2Vec:
    """Wrapper offering gensim-like .most_similar() interface"""
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab
        with torch.no_grad():
            self.vectors = model.in_embed.weight.cpu().numpy()
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.normed_vectors = self.vectors / norms

    def __contains__(self, word): return word in self.vocab.word2idx
    def __getitem__(self, word): return self.vectors[self.vocab.word2idx[word]]

    def most_similar(self, word, topn=5):
        if word not in self.vocab.word2idx: return []
        idx = self.vocab.word2idx[word]
        query = self.normed_vectors[idx]
        sims = self.normed_vectors @ query
        sims[idx] = -1.0
        top_indices = np.argsort(sims)[::-1][:topn]
        return [(self.vocab.idx2word[i], float(sims[i])) for i in top_indices]

    def most_similar_analogy(self, positive, negative, topn=3):
        vec = np.zeros(self.vectors.shape[1])
        exclude = set()
        for w in positive:
            if w not in self.vocab.word2idx: return []
            vec += self.normed_vectors[self.vocab.word2idx[w]]
            exclude.add(self.vocab.word2idx[w])
        for w in negative:
            if w not in self.vocab.word2idx: return []
            vec -= self.normed_vectors[self.vocab.word2idx[w]]
            exclude.add(self.vocab.word2idx[w])
        norm = np.linalg.norm(vec)
        if norm > 0: vec = vec / norm
        sims = self.normed_vectors @ vec
        for idx in exclude: sims[idx] = -1.0
        top_indices = np.argsort(sims)[::-1][:topn]
        return [(self.vocab.idx2word[i], float(sims[i])) for i in top_indices]

def train_scratch_models(sentences, output_dir, embed_dim=300, window_size=10,
                         num_neg=10, epochs=20, batch_size=512, lr=0.003):
    print(f"\n  {'='*50}")
    print(f"  MODEL TRAINING (PyTorch From-Scratch)")
    print(f"  {'='*50}")
    
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    vocab = Vocabulary(sentences, min_count=1)
    noise_dist_tensor = torch.tensor(vocab.noise_dist, dtype=torch.float32)
    models = {}

    for mode in ["CBOW", "Skip-gram"]:
        print(f"\n  ── {mode} ────────────────────────────────────────")
        if mode == "CBOW":
            dataset = CBOWDataset(sentences, vocab, window_size)
            model = CBOWModelNN(vocab.vocab_size, embed_dim).to(device)
            print(f"  [Info] CBOW Dataset Size: {len(dataset):,} context-target training pairs")
        else:
            dataset = SkipGramDataset(sentences, vocab, window_size)
            model = SkipGramModelNN(vocab.vocab_size, embed_dim).to(device)
            print(f"  [Info] Skip-gram Dataset Size: {len(dataset):,} context-target training pairs")

        # Optimization: Enable num_workers and pin_memory for faster data transfer!
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                                drop_last=False, num_workers=4, pin_memory=True if torch.cuda.is_available() else False)
        
        # Adam converges quickly and handles sparse gradients gracefully
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            n_batches = 0
            for batch in dataloader:
                if mode == "CBOW":
                    c_ids, t_ids = batch[0].to(device), batch[1].to(device)
                    n_ids = torch.multinomial(noise_dist_tensor, c_ids.size(0) * num_neg, True).view(c_ids.size(0), num_neg).to(device)
                    loss = model(c_ids, t_ids, n_ids)
                else:
                    t_ids, c_ids = batch[0].to(device), batch[1].to(device)
                    n_ids = torch.multinomial(noise_dist_tensor, t_ids.size(0) * num_neg, True).view(t_ids.size(0), num_neg).to(device)
                    loss = model(t_ids, c_ids, n_ids)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(n_batches, 1)
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:>3}/{epochs}  Loss: {avg_loss:.4f}  LR: {scheduler.get_last_lr()[0]:.6f}")

        model.eval()
        models[mode] = ScratchWord2Vec(model, vocab)
        model_path = os.path.join(output_dir, f"word2vec_{mode.lower().replace('-', '_')}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"  ✓ Model saved → {model_path}")

    return models


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TASK 3: SEMANTIC ANALYSIS                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def semantic_analysis(models):
    print(f"\n  {'='*50}")
    print(f"  SEMANTIC ANALYSIS")
    print(f"  {'='*50}")
    
    # Required targets plus one for the accidental duplicate 'exam'
    target_words = ["research", "student", "phd", "exam", "course"]
    
    for mode, model in models.items():
        print(f"\n  ── {mode} — Nearest Neighbors ──────────────")
        for word in target_words:
            if word in model:
                neighbors = model.most_similar(word, topn=5)
                n_str = ", ".join([f"{w}({s:.2f})" for w, s in neighbors])
                print(f"  {word:<12} → {n_str}")
            else:
                print(f"  {word:<12} → (not in vocab)")

    print(f"\n  ── Analogy Experiments ──────────────────────────")
    analogies = [
        ("ug", "btech", "pg", "UG : BTech :: PG : ?"),
        ("student", "exam", "faculty", "student : exam :: faculty : ?"),
        ("research", "phd", "teaching", "research : PhD :: teaching : ?")
    ]
    for mode, model in models.items():
        print(f"\n  {mode}:")
        for pos1, pos2, neg1, desc in analogies:
            if all(w in model for w in [pos1, pos2, neg1]):
                results = model.most_similar_analogy(positive=[pos2, neg1], negative=[pos1], topn=3)
                ans = ", ".join([f"{w}({s:.2f})" for w, s in results])
                print(f"    {desc} → {ans}")
            else:
                print(f"    {desc} → missing vocab")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TASK 4: VISUALIZATION                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def visualize_embeddings(models, output_dir):
    print(f"\n  {'='*50}")
    print(f"  VISUALIZATION")
    print(f"  {'='*50}")

    word_groups = {
        "Academics": ["btech", "mtech", "phd", "undergraduate", "postgraduate", "course", "curriculum"],
        "Research": ["research", "thesis", "publication", "journal", "conference", "project", "laboratory"],
        "People": ["student", "faculty", "professor", "scholar", "advisor", "dean", "director", "department"],
        "Evaluation": ["exam", "grade", "credit", "semester", "registration", "evaluation", "cgpa"],
    }
    colors = {"Academics": "#e74c3c", "Research": "#3498db", "People": "#2ecc71", "Evaluation": "#f39c12"}

    for mode, model in models.items():
        words, word_colors, word_labels = [], [], []
        for gname, gwords in word_groups.items():
            for w in gwords:
                if w in model:
                    words.append(w)
                    word_colors.append(colors[gname])
                    word_labels.append(gname)

        if len(words) < 5: continue
        vectors = np.array([model[w] for w in words])

        pca = PCA(n_components=2)
        coords = pca.fit_transform(vectors)
        fig, ax = plt.subplots(figsize=(12, 8))
        for gname, color in colors.items():
            mask = [l == gname for l in word_labels]
            if any(mask):
                xs, ys = coords[mask, 0], coords[mask, 1]
                ax.scatter(xs, ys, c=color, label=gname, s=80, alpha=0.8, edgecolors="white")
                for x, y, w in zip(xs, ys, [words[i] for i, m in enumerate(mask) if m]):
                    ax.annotate(w, (x, y), fontsize=9, ha="center", va="bottom", xytext=(0, 5), textcoords="offset points")
        
        ax.set_title(f"PCA — {mode} Word Embeddings", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fname = mode.lower().replace("-", "_")
        pca_path = os.path.join(output_dir, f"pca_{fname}.png")
        plt.tight_layout()
        plt.savefig(pca_path, dpi=150)
        plt.close()
        print(f"  ✓ PCA plot saved → {pca_path}")

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(words)-1), n_iter=1000)
        coords_t = tsne.fit_transform(vectors)
        fig, ax = plt.subplots(figsize=(12, 8))
        for gname, color in colors.items():
            mask = [l == gname for l in word_labels]
            if any(mask):
                xs, ys = coords_t[mask, 0], coords_t[mask, 1]
                ax.scatter(xs, ys, c=color, label=gname, s=80, alpha=0.8, edgecolors="white")
                for x, y, w in zip(xs, ys, [words[i] for i, m in enumerate(mask) if m]):
                    ax.annotate(w, (x, y), fontsize=9, ha="center", va="bottom", xytext=(0, 5), textcoords="offset points")
        ax.set_title(f"t-SNE — {mode} Word Embeddings", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        tsne_path = os.path.join(output_dir, f"tsne_{fname}.png")
        plt.tight_layout()
        plt.savefig(tsne_path, dpi=150)
        plt.close()
        print(f"  ✓ t-SNE plot saved → {tsne_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(base_dir, "..")  # Problem1 root
    data_dir = os.path.join(project_dir, "data")
    output_dir = os.path.join(project_dir, "images")
    logs_dir = os.path.join(project_dir, "logs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    import sys
    class Logger(object):
        def __init__(self, log_file):
            self.terminal = sys.stdout
            self.log = open(log_file, "w", encoding="utf-8")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            self.terminal.flush()
            self.log.flush()
            
    sys.stdout = Logger(os.path.join(logs_dir, "training.log"))

    print("=" * 60)
    print("  Word2Vec Pipeline (PyTorch From Scratch) — IIT Jodhpur")
    print("=" * 60)

    # 1. Dataset Preparation
    print("\n  📂 Loading corpus...")
    docs = load_corpus(data_dir)
    sentences, tokens = build_corpus(docs)
    report_statistics(docs, sentences, tokens, output_dir)
    clean_path = os.path.join(project_dir, "data", "cleaned_corpus.txt")
    with open(clean_path, "w") as f:
        for sent in sentences: f.write(" ".join(sent) + "\n")

    # 2. Model Training
    models = train_scratch_models(sentences, output_dir)

    # 3. Semantic Analysis
    semantic_analysis(models)

    # 4. Visualization
    visualize_embeddings(models, output_dir)

    print(f"\n{'='*60}")
    print("  ✓ Complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
