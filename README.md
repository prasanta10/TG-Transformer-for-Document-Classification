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

# Ablations E1-E5, then their summary tables and plots
python run_experiments.py --only e1 e2 e3 e4 e5
python summarize_experiments.py

# Edge-aware attention (N1), the attention-by-rank check, and its summary
python run_experiments.py --only N1
python n1_attention.py
python summarize_n1.py
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
| N1: graph bias inside attention | `--attn_bias` | `none` / `intimacy` / `edge` |

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
- **β in N1 gets no weight decay**, so it isn't pulled back toward 0 (the baseline)
  just by the regulariser.
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

### Edge-aware attention (N1)

In the original model, graph scores only decide which words enter a document's slice.
Attention itself never sees how strongly those nodes are connected. N1 adds that, in
the style of Graphormer: each head's attention scores get an extra term β·B before the
softmax. B is a 24 × 24 table of graph scores for the slice, z-scored within the
slice, and β is one learned number per layer and head, starting at 0. We tried two
kinds of B:

- **N1a, `intimacy`:** the log intimacy score between every pair of nodes in the slice.
- **N1b, `edge`:** the log graph-edge weight, TF-IDF between the document and a word
  and PMI between two words, with 0 where there is no edge.

Neither helped (full tables in [`results/experiments/N1/summary.md`](results/experiments/N1/summary.md)):

| Dataset | Reference | N1a: intimacy | N1b: edge |
|---|---|---|---|
| R8 | 95.08 ± 0.29 | 95.05 ± 0.33 | 95.07 ± 0.34 |
| Ohsumed | 53.79 ± 1.18 | 53.57 ± 1.15 | 53.70 ± 1.20 |

All differences are within 0.25 points. N1a on Ohsumed is lower by 0.22 points with
p = 0.014, but that is one uncorrected test out of eight and the gap is tiny. Every
learned β came out positive but small: 0.02 to 0.24 across all runs, and 0.09 to 0.19
when averaged over seeds. So the model does lean toward strongly connected pairs, just
not by much. With the bias, the target's
attention to its five closest helpers rose from 0.043 to 0.051 on R8, and from 0.047
to 0.052 on Ohsumed. The best checkpoint still comes at about epoch 6, the same
as the reference. Building B costs about 27 s (R8) and 74 s (Ohsumed) for the intimacy
version and under 4 s for the edge version, with about 17 MB of storage each.

## Code layout

```
data.py                  corpus loading, GloVe features, preprocessing and cache
graph.py                 TF-IDF/PMI graph, intimacy scores, helper selection
wl.py                    Weisfeiler-Lehman codes
sampling.py              intimacy / tfidf / random samplers
model.py                 the TG-Transformer
train.py                 training loop, early stopping, evaluation
run.py                   command-line entry point
run_experiments.py       runs ablations E1-E5 and N1
attn_bias.py             builds the N1 attention-bias matrices B
summarize_experiments.py summary tables, significance tests, plots
time_samplers.py         preprocessing cost of each sampler (E2)
summarize_n1.py          N1 summary: tables, significance tests, learned beta, cost
n1_attention.py          N1 check: attention to helpers by intimacy rank
configs/baseline.yaml    baseline settings
```
