# TG-Transformer

This is a from-scratch reimplementation of *Text Graph Transformer for Document
Classification* (Zhang & Zhang, EMNLP 2020), tested on the R8 and Ohsumed datasets.
The authors didn't release code, so wherever the paper leaves something out, this
README says which choice we made.

## Setup

Everything runs on a CPU. We used Linux with Python 3.11, 12 cores and 14 GB of RAM.

```bash
conda create -n tgt python=3.11
conda activate tgt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The data isn't in this repo, so download it first:

| File | Where to get it | Put it in |
|---|---|---|
| `R8.clean.txt`, `ohsumed.clean.txt` | [TextGCN repo](https://github.com/yao8839836/text_gcn), `data/corpus/` | `data/raw/textgcn/` |
| `R8.txt`, `ohsumed.txt` (train/test split and labels) | same repo, `data/` | `data/raw/textgcn/` |
| `glove.6B.300d.txt` | [GloVe 6B](https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip) | `data/raw/glove/` |

## Running it

```bash
# Full baseline: preprocessing, then R8 and Ohsumed with seeds 0-4
python run.py --config configs/baseline.yaml

# Preprocessing only (add --force_preprocess to rebuild the cache)
python run.py --config configs/baseline.yaml --preprocess_only

# All ablations E1-E5, then the summary tables and plots
python run_experiments.py
python summarize_experiments.py
```

Preprocessing runs once per dataset and is cached in `data/processed/`. It takes
about 35 s for R8 and 90 s for Ohsumed. Results are written to `results/<experiment>/`,
with one JSON file and one training-curve CSV per seed, plus a `summary.md`.

### Ablation switches

Each switch can be set in the config file or on the command line. Use
`--experiment <name>` so each run gets its own results folder.

| Experiment | Flag | Values |
|---|---|---|
| E1: node-type encoding | `--use_type_enc` | `on` / `off` |
| E1: WL encoding | `--use_wl_enc` | `on` / `off` |
| E2: how helper words are picked | `--sampler` | `intimacy` / `tfidf` / `random` |
| E3: number of helper words | `--k` | 1 to 50 |
| E4: readout | `--readout` | `mean` / `target` |
| E5: freeze the input embeddings | `--freeze_embeddings` | `on` / `off` |

```bash
python run.py --config configs/baseline.yaml --experiment e2_tfidf --sampler tfidf
```

## How it works

**The graph.** As in TextGCN, every document and every word is a node. Word nodes link
to the documents they appear in, weighted by TF-IDF, and to other words, weighted by
positive PMI over 20-word sliding windows. The graph includes the test documents, but
their labels are hidden throughout training and model selection. This is the same
transductive setup the paper uses.

**Picking context.** For each document we compute an intimacy score with every node
(personalised PageRank, α = 0.15). The k = 23 words with the highest scores become the
document's helpers. The `tfidf` and `random` samplers are simpler options used in E2.

**The model.** Each input is a short sequence: the document followed by its helper
words. A node's input vector is the sum of three parts: its GloVe-based features, an
encoding for its type (document or word) and a Weisfeiler–Lehman (WL) structural
encoding. The sum is projected down to 32 dimensions and passed through 2 transformer
layers with 4 heads each. A mean over the sequence goes to a linear classifier.

**Training.** Adam with lr 1e-3 and batch size 64. 10% of the training documents are
held out for validation. Training stops early after 10 epochs without improvement in
validation loss. Seeds are fixed and runs are deterministic.

### Choices the paper doesn't spell out

- **PMI** follows the paper's Eq. 1 exactly: each window counts a word pair once.
  TextGCN's code counts repeated pairs more than once.
- **Words missing from GloVe** get random uniform(−0.1, 0.1) vectors. A document's
  features are the mean of its words' vectors.
- **WL codes** are computed on the unweighted graph until the colours stop changing.
  That happens after 2 rounds of refinement on both datasets.
- **The G-Res shortcut** adds a projection of each position's raw features to the
  output of every layer.
- **Weight decay** (5e-4) is applied to everything except the embedding table.
  Otherwise it would slowly shrink the embeddings of test documents, which never
  receive a gradient.
- **Only documents are classified.** Word nodes have no labels, so the paper's
  word-target part isn't implemented.

### A pitfall worth knowing about

WL refinement gives almost every node its own colour: 15,205 colours for 15,362 nodes
on R8. Our first version numbered the colours in the order it first met them. The
Ohsumed file is sorted by class, so the WL code quietly encoded each training
document's label. Validation accuracy hit 98% while test accuracy was 22%. We now
number colours by sorting their signatures, so the codes depend only on the graph's
structure. That fixed it.

## Results

Baseline over 5 seeds, mean ± std (details in [`results/baseline/summary.md`](results/baseline/summary.md)):

| Dataset | Accuracy (%) | Macro-F1 (%) | Paper accuracy (%) |
|---|---|---|---|
| R8 | 95.08 ± 0.29 | 86.50 ± 1.76 | 98.1 |
| Ohsumed | 53.79 ± 1.18 | 34.74 ± 2.43 | 70.4 |

R8 comes close to the paper, but Ohsumed falls well short. The model overfits within
4–6 epochs, and with Ohsumed's long documents, 23 helper words cover only about a
third of each document's vocabulary. We did not tune any hyperparameters to close
the gap.

What the ablations showed (full tables in [`results/experiments/summary.md`](results/experiments/summary.md)):

- **Freezing the embeddings (E5)** gave the biggest gain on Ohsumed: +4.6 points
  accuracy and +7.4 macro-F1. On R8 it made no difference.
- **More helper words (E3)** helped steadily on both datasets, up to k = 40.
- **The plain TF-IDF sampler (E2)** did slightly better than intimacy (+0.7 on R8,
  +1.2 on Ohsumed) and was much cheaper to compute. Random helpers fell to
  near-chance accuracy on Ohsumed.
- **The structural encodings (E1)** mattered on Ohsumed: turning both off cost
  5.6 points. On R8, accuracy barely changed, though macro-F1 dropped 2.7 points.
- **Reading out only the document's own position (E4)** hurt Ohsumed by 8.3 points.

## Code layout

```
data.py                  corpus loading, GloVe features, preprocessing and cache
graph.py                 TF-IDF/PMI graph, intimacy scores, helper selection
wl.py                    Weisfeiler-Lehman codes
sampling.py              intimacy / tfidf / random samplers
model.py                 the TG-Transformer
train.py                 training loop, early stopping, evaluation
run.py                   command-line entry point
run_experiments.py       runs ablations E1-E5
summarize_experiments.py summary tables, significance tests, plots
time_samplers.py         preprocessing cost of each sampler (E2)
configs/baseline.yaml    baseline settings
```
