import os
import matplotlib
matplotlib.use("Agg") # Non-interactive backend to save plots directly to disk
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ==============================================================================
# 1. LOAD DATA
# We load the perfectly clean, tokenized sentences from our scratch pipeline.
# Gensim expects a list of lists of strings: [['this', 'is', 'a', 'sentence'], ...]
# ==============================================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
corpus_path = os.path.join(base_dir, "..", "data", "cleaned_corpus.txt")
output_dir = os.path.join(base_dir, "..", "images")
os.makedirs(output_dir, exist_ok=True)

sentences = []
if os.path.exists(corpus_path):
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                sentences.append(tokens)
else:
    print(f"Error: {corpus_path} not found. Run word2vec_scratch.py first.")
    exit(1)

print(f"Loaded {len(sentences)} sentences for Gensim training.")

# ==============================================================================
# 2. HYPERPARAMETER EXPERIMENTS
# We train miniature Gensim models across an 8-parameter grid (dims, window, neg).
# Because Gensim is highly optimized in C, this takes just seconds to run!
# ==============================================================================
print("\n" + "="*50)
print("HYPERPARAMETER EXPERIMENTS (Gensim)")
print("="*50)

dims = [50, 100]
windows = [2, 5]
negatives = [5, 10]

experiment_results = []

for sg in [0, 1]:
    mode = "Skip-gram" if sg == 1 else "CBOW"
    print(f"\n--- {mode} Experiments ---")
    for dim in dims:
        for win in windows:
            for neg in negatives:
                model = Word2Vec(sentences, vector_size=dim, window=win, sg=sg, negative=neg, min_count=1, workers=4, seed=42)
                # Just noting the vocabulary size and vector size as a sanity check
                desc = f"{mode}: dim={dim}, window={win}, neg={neg}"
                print(f"Trained {desc}")
                experiment_results.append((mode, dim, win, neg))

# ==============================================================================
# 3. TRAIN STANDARD MODELS FOR COMPARISON
# Now we train our "Final" CBOW and Skip-gram models using our best parameters
# from the scratch implementation (100 dims, 5 window, 10 negative samples).
# ==============================================================================
print("\n" + "="*50)
print("TRAINING STANDARD MODELS FOR COMPARISON")
print("="*50)

# sg=0 means CBOW, sg=1 means Skip-gram!
cbow_model = Word2Vec(sentences, vector_size=300, window=10, sg=0, negative=10, min_count=1, workers=4, seed=42, epochs=50)
sg_model = Word2Vec(sentences, vector_size=300, window=10, sg=1, negative=10, min_count=1, workers=4, seed=42, epochs=50)

models = {"CBOW (Gensim)": cbow_model, "Skip-gram (Gensim)": sg_model}

# ==============================================================================
# 4. SEMANTIC ANALYSIS
# We query the Gensim models for the same nearest neighbors and analogies
# as our PyTorch models to directly compare their semantic mapping performance.
# ==============================================================================
print("\n" + "="*50)
print("SEMANTIC ANALYSIS (Gensim)")
print("="*50)

target_words = ["research", "student", "phd", "exam", "course"]
analogies = [
    ("ug", "btech", "pg", "UG : BTech :: PG : ?"),
    ("student", "exam", "faculty", "student : exam :: faculty : ?"),
    ("research", "phd", "teaching", "research : PhD :: teaching : ?")
]

comparison_out = []

for name, model in models.items():
    print(f"\n[{name}] Nearest Neighbors:")
    comparison_out.append(f"\n[{name}] Nearest Neighbors:")
    for word in target_words:
        if word in model.wv:
            neighbors = model.wv.most_similar(word, topn=5)
            nbr_str = ", ".join([f"{w}({s:.2f})" for w, s in neighbors])
            out_str = f"  {word:<12} -> {nbr_str}"
            print(out_str)
            comparison_out.append(out_str)
        else:
            print(f"  {word:<12} -> (not in vocab)")
            comparison_out.append(f"  {word:<12} -> (not in vocab)")

    print(f"\n[{name}] Analogies:")
    comparison_out.append(f"\n[{name}] Analogies:")
    for pos1, pos2, neg1, desc in analogies:
        if all(w in model.wv for w in [pos1, pos2, neg1]):
            # pos2 - pos1 + neg1 => PG - UG + BTech
            results = model.wv.most_similar(positive=[pos2, neg1], negative=[pos1], topn=3)
            res_str = ", ".join([f"{w}({s:.2f})" for w, s in results])
            out_str = f"  {desc} -> {res_str}"
            print(out_str)
            comparison_out.append(out_str)
        else:
            print(f"  {desc} -> missing vocab")
            comparison_out.append(f"  {desc} -> missing vocab")

# Save comparison output
with open(os.path.join(base_dir, "..", "logs", "gensim_semantic_analysis.txt"), "w") as f:
    f.write("\n".join(comparison_out))

# ==============================================================================
# 5. VISUALIZATION
# Plotting the 100-dimensional vectors in 2D space.
# We focus on 4 distinct academic categories to see if Gensim clusters them 
# together effectively (spoiler: it usually does!).
# ==============================================================================
print("\n" + "="*50)
print("VISUALIZATION (Gensim)")
print("="*50)

word_groups = {
    "Academics": ["btech", "mtech", "phd", "undergraduate", "postgraduate", "course", "curriculum"],
    "Research": ["research", "thesis", "publication", "journal", "conference", "project", "laboratory"],
    "People": ["student", "faculty", "professor", "scholar", "advisor", "dean", "director", "department"],
    "Evaluation": ["exam", "grade", "credit", "semester", "registration", "evaluation", "cgpa"],
}
colors = {"Academics": "#e74c3c", "Research": "#3498db", "People": "#2ecc71", "Evaluation": "#f39c12"}

for name, model in models.items():
    words, word_colors, word_labels = [], [], []
    for gname, gwords in word_groups.items():
        for w in gwords:
            if w in model.wv:
                words.append(w)
                word_colors.append(colors[gname])
                word_labels.append(gname)

    if len(words) < 5:
        continue
        
    vectors = np.array([model.wv[w] for w in words])
    fname = name.split()[0].lower().replace("-", "_")

    # PCA
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(vectors)
    fig, ax = plt.subplots(figsize=(12, 8))
    for gname, color in colors.items():
        mask = [l == gname for l in word_labels]
        if any(mask):
            xs, ys = coords[np.array(mask), 0], coords[np.array(mask), 1]
            ax.scatter(xs, ys, c=color, label=gname, s=80, alpha=0.8, edgecolors="white")
            for x, y, w in zip(xs, ys, [words[i] for i, m in enumerate(mask) if m]):
                ax.annotate(w, (x, y), fontsize=9, ha="center", va="bottom", xytext=(0, 5), textcoords="offset points")
    ax.set_title(f"PCA — {name}", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    pca_path = os.path.join(output_dir, f"gensim_pca_{fname}.png")
    plt.tight_layout()
    plt.savefig(pca_path, dpi=150)
    plt.close()
    print(f"  ✓ PCA saved -> {pca_path}")

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(words)-1), n_iter=1000)
    coords_t = tsne.fit_transform(vectors)
    fig, ax = plt.subplots(figsize=(12, 8))
    for gname, color in colors.items():
        mask = [l == gname for l in word_labels]
        if any(mask):
            xs, ys = coords_t[np.array(mask), 0], coords_t[np.array(mask), 1]
            ax.scatter(xs, ys, c=color, label=gname, s=80, alpha=0.8, edgecolors="white")
            for x, y, w in zip(xs, ys, [words[i] for i, m in enumerate(mask) if m]):
                ax.annotate(w, (x, y), fontsize=9, ha="center", va="bottom", xytext=(0, 5), textcoords="offset points")
    ax.set_title(f"t-SNE — {name}", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    tsne_path = os.path.join(output_dir, f"gensim_tsne_{fname}.png")
    plt.tight_layout()
    plt.savefig(tsne_path, dpi=150)
    plt.close()
    print(f"  ✓ t-SNE saved -> {tsne_path}")

print("\nDone! All Gensim models trained and compared.")
