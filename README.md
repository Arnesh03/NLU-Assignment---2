# NLU Assignment 2 — Word Embeddings & Character-Level RNN Name Generation

**Author:** Arnesh Sanjeev Singh (M25CSE008)  
**Course:** Natural Language Understanding (NLU), Semester 2  
**Institution:** Indian Institute of Technology Jodhpur

---

## 📁 Project Structure

```
Nlu_Assignment-2/
├── README.md
├── report.tex                  ← Combined LaTeX report (Problem 1 + Problem 2)
│
├── Problem1/                   ← Word2Vec Embeddings
│   ├── report.tex              ← Standalone Problem 1 report
│   ├── src/
│   │   ├── collect_data.py     ← Web scraping + preprocessing pipeline
│   │   ├── word2vec_scratch.py ← From-scratch PyTorch CBOW & Skip-gram
│   │   └── word2vec_gensim.py  ← Gensim-based Word2Vec for comparison
│   ├── data/
│   │   ├── raw_corpus.txt      ← Raw scraped text from IIT Jodhpur
│   │   ├── cleaned_corpus.txt  ← Preprocessed corpus
│   │   └── dataset_statistics.txt
│   ├── images/                 ← All generated visualizations (9 PNGs)
│   │   ├── wordcloud.png
│   │   ├── pca_cbow.png / pca_skip_gram.png
│   │   ├── tsne_cbow.png / tsne_skip_gram.png
│   │   └── gensim_pca_*.png / gensim_tsne_*.png
│   └── logs/
│       ├── training.log
│       ├── collect_data.log
│       └── gensim_semantic_analysis.txt
│
└── Problem2/                   ← Character-Level RNN Name Generation
    ├── report.tex              ← Standalone Problem 2 report
    ├── src/
    │   ├── dataset.py          ← Character vocabulary & dataset utilities
    │   ├── models.py           ← VanillaRNN, BiLSTM, AttentionRNN architectures
    │   ├── train.py            ← Training loop with early stopping
    │   ├── generate.py         ← Autoregressive name generation
    │   ├── evaluate.py         ← Novelty & Diversity metrics
    │   ├── generate_dataset.py ← Indian name dataset generator
    │   └── run_all.py          ← End-to-end pipeline runner
    ├── data/
    │   └── TrainingNames.txt   ← 1000 unique Indian names
    ├── checkpoints/            ← Saved model weights
    │   ├── VanillaRNN.pth
    │   ├── BLSTM.pth
    │   └── AttentionRNN.pth
    └── generated/              ← Generated name outputs
        ├── gen_VanillaRNN.txt
        ├── gen_BLSTM.txt
        └── gen_AttentionRNN.txt
```

---

## 🔧 Prerequisites

```bash
pip install torch numpy matplotlib scikit-learn wordcloud requests beautifulsoup4 gensim
```

> Python 3.8+ required. GPU optional but recommended for faster training.

---

## 🚀 How to Run

### Problem 1: Word2Vec Embeddings

```bash
cd Problem1

# Step 1: Scrape and preprocess data from IIT Jodhpur website
python3 src/collect_data.py

# Step 2: Train Word2Vec from scratch (CBOW + Skip-gram, 300-d embeddings)
#         Generates visualizations (PCA, t-SNE, Word Cloud) + semantic analysis
python3 src/word2vec_scratch.py

# Step 3: Train Gensim Word2Vec for comparison
python3 src/word2vec_gensim.py
```

**Outputs:**
- `images/` — PCA and t-SNE plots, word cloud
- `logs/training.log` — Training progress and semantic analysis results
- `logs/gensim_semantic_analysis.txt` — Gensim model analysis

---

### Problem 2: Character-Level RNN Name Generation

```bash
cd Problem2/src

# Run the full pipeline (train → generate → evaluate) in one command:
python3 run_all.py
```

Or run each step individually:

```bash
# Step 1: Generate the training dataset (if not already present)
python3 generate_dataset.py

# Step 2: Train all 3 models (VanillaRNN, BiLSTM, AttentionRNN)
python3 train.py

# Step 3: Generate 100 names from each model
python3 generate.py

# Step 4: Evaluate novelty and diversity metrics
python3 evaluate.py
```

**Outputs:**
- `checkpoints/` — Trained model weights (`.pth`)
- `generated/` — Generated names from each model
- Console output — Novelty rate and diversity scores

---

## 📊 Key Results

### Problem 1: Word2Vec

| Model | Dimension | Window | Neg. Samples | Final Loss |
|-------|-----------|--------|-------------|------------|
| **CBOW** | **300** | **10** | **10** | **0.8514** |
| **Skip-gram** | **300** | **10** | **10** | **1.3268** |

- Both CBOW and Skip-gram successfully learn academic semantic relationships from IIT Jodhpur data
- Key analogies recovered: `exam → quiz`, `phd → mtech`, `student:exam :: faculty:expertise`

### Problem 2: RNN Name Generation

| Model | Parameters | Novelty | Diversity | Quality |
|-------|-----------|---------|-----------|---------|
| VanillaRNN | ~30K | ~95% | ~0.91 | Readable ✓ |
| BiLSTM | ~330K | ~100% | ~1.00 | Garbled ✗  |
| **AttentionRNN** | **~105K** | **~95%** | **~0.92** | **Best ✓** |

- AttentionRNN produces the most coherent and phonetically realistic Indian names
- BiLSTM failure demonstrates the train/generate architectural mismatch in bidirectional models

---

## 📝 Report

Compile the LaTeX report:

```bash
pdflatex report.tex    # Combined report (both problems)
```

Individual reports are also available inside `Problem1/report.tex` and `Problem2/report.tex`.

---

## 📄 License

This project is part of academic coursework at IIT Jodhpur. For educational purposes only.
