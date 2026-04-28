# Handwritten Maths to LaTeX (Based on TAMER)

This repository focuses on **Handwritten Mathematical Expression Recognition (HMER)** — converting handwritten mathematical expressions into LaTeX code using deep learning.

The project is built upon the original **TAMER** framework (AAAI 2025), with additional personal experiments, modifications, and utilities developed for learning and research purposes.

---

## Overview

Handwritten mathematical expression recognition is a challenging task involving:
- Complex 2D symbol layouts
- Variable handwriting styles
- Large mathematical vocabularies
- Structural ambiguity between symbols

TAMER addresses these challenges through a **Tree-Aware Transformer** that jointly optimises sequence prediction and expression tree structure prediction — combining the flexibility of sequence decoding with the structural awareness of tree decoding.

---

## How TAMER Works (Brief)

The pipeline has four main stages:

**1. Visual Encoding** — A DenseNet backbone extracts spatial features from the input image, downsampled 16× with 2D sinusoidal positional encoding to preserve spatial layout information.

**2. Transformer Decoder with Coverage Attention** — An autoregressive transformer decoder attends to the encoded image features to predict LaTeX tokens one at a time. The ARM (Attention Refinement Module) tracks cumulative attention heatmaps across decoding steps, steering the model away from image regions it has already decoded — preventing symbol repetition and omission.

**3. Tree-Aware Module (TAM)** — Operating on the decoder's hidden states, TAM predicts a relationship score matrix `S ∈ R^{L×L}` where `S[i,j]` represents the likelihood that token `i` is a child of token `j` in the expression tree. This enforces structural awareness (bracket matching, superscript/subscript hierarchy) through a joint training loss without breaking the sequence decoding paradigm.

**4. Inference** — Beam search generates candidate sequences from the sequence head. TAM re-ranks candidates by adding a tree structure coherence score, selecting the final output that is both linguistically fluent and structurally valid.

---

## Base Repository

This work is built upon:

**TAMER (AAAI 2025)**
https://github.com/qingzhenduyu/TAMER

> Zhu et al., *TAMER: Tree-Aware Transformer for Handwritten Mathematical Expression Recognition*, AAAI 2025.

Full credit for the original architecture, training framework, and core implementation belongs to the original authors.

---

## Installation

```bash
cd TAMER

# create and activate environment
conda create -y -n TAMER python=3.7
conda activate TAMER

# core dependencies
conda install pytorch=1.8.1 torchvision=0.2.2 cudatoolkit=11.1 pillow=8.4.0 -c pytorch -c nvidia

# training dependencies
conda install pytorch-lightning=1.4.9 torchmetrics=0.6.0 -c conda-forge

# evaluation dependency
conda install pandoc=1.19.2.1 -c conda-forge

# install project in editable mode
pip install -e .
```

---

## My Contributions

### Dataset & Preprocessing
- InkML stroke parsing and stroke-to-image rasterization pipeline
- Dataset cleaning, image size verification, and pickle generation utilities
- Label retokenization tools for vocabulary alignment

### Vocabulary Expansion
- Extended the model vocabulary from 114 → 248 → 334 tokens to support MathWriting and HME100K datasets
- OOV checking scripts, caption/token length analysis, and filtering of unsupported labels
- Extended token dictionary generation aligned with the TAMER vocab format

### Model Surgery (`Model_surgery.py`)
Checkpoint modification to expand vocab-dependent weight tensors without retraining from scratch. Specifically:
- `decoder.word_embed.0.weight` `[V_old × d]` → `[V_new × d]` — token input embedding
- `decoder.proj.weight` `[V_old × d]` → `[V_new × d]` — sequence prediction head
- `decoder.proj.bias` `[V_old]` → `[V_new]` — sequence prediction head bias

New rows are zero-initialised so existing token behaviour is preserved exactly. All other weights (DenseNet, attention layers, ARM, TAM) are untouched as they operate purely in model-dimension space and are vocabulary-agnostic.

### Fine-tuning (`Model_finetune.py`)
Decoder-only fine-tuning workflow on MathWriting data:
- Encoder frozen — visual features transfer well across handwritten math datasets
- Only decoder weights trained: embedding table, all attention layers, projection head
- Gradient clipping at `max_norm=1.0` to stabilise training from zero-initialised new-token rows
- Lighter pipeline (sequence loss only, no struct loss) for faster iteration

### Web Demo (`app.py`)
A Flask web application for side-by-side comparison of both model versions:
- Upload a handwritten math image via browser
- Both models run beam search inference simultaneously
- Results displayed as raw LaTeX code and live-rendered mathematical notation (via MathJax)
- Handles the vocab singleton swap problem — each model inference temporarily rewrites the global vocab mapping to its own word list before running, preventing index-out-of-bounds errors from the vocab size mismatch between v1 (248) and v4 (334)
- Copy-to-clipboard for each model's LaTeX output

### Evaluation & Inference
- Single image inference pipeline with beam search and TAM re-ranking
- Prediction verification scripts against ground truth
- Custom testing utilities for CROHME / HME100K / MathWriting

---

## Repository Structure

```text
config/                     # Training and dataset configs
eval/                       # Evaluation scripts
tamer/                      # Core TAMER implementation
│   ├── model/
│   │   ├── tamer.py            # Top-level model, bidirectional forward pass
│   │   ├── encoder.py          # DenseNet + Conv1×1 + 2D positional encoding
│   │   ├── decoder.py          # Transformer decoder + TAM
│   │   └── transformer/
│   │       ├── attention.py    # Multi-head attention with ARM hook
│   │       ├── arm.py          # Attention Refinement Module (coverage)
│   │       └── transformer_decoder.py
│   └── datamodule/
│       ├── datamodule.py       # Batching, padding, mask creation
│       ├── dataset.py          # Image transforms and Dataset wrapper
│       ├── vocab.py            # Token ↔ index mapping
│       └── latex2gtd.py        # LaTeX → parse tree → parent index array
app.py                      # Flask web demo, dual-model comparison UI
Model_finetune.py           # Decoder-only fine-tuning on MathWriting
Model_surgery.py            # Vocab expansion via checkpoint weight surgery
build_extended_dict.py      # Extended vocabulary dictionary generation
build_pkl.py                # Dataset pickle creation
single_inference.py         # Single image → LaTeX inference
check_oov.py                # OOV token analysis
inkml_to_image_v2.py        # InkML stroke → rasterized image conversion
```

---

## `app.py` — Flask Web Demo

A self-contained inference server that loads both model checkpoints and serves a browser UI for side-by-side comparison.

### Model loading

Two models are loaded at startup:

**v1** — the base CROHME-trained checkpoint with vocab size 248, loaded directly via `LitTAMER.load_from_checkpoint`.

**v4** — the fine-tuned MathWriting checkpoint with vocab size 334. Loading is done in three steps because the checkpoint's vocab-expanded layers have different shapes than the base architecture:
1. Load the base architecture from the v1 checkpoint (correct layer structure)
2. Replace the two vocab-dependent layers with fresh modules of the correct size: `Embedding(334, 256)` and `Linear(256, 334)`
3. Load fine-tuned weights with `strict=False` (skips shape mismatches), then manually copy the three surgery-expanded tensors by name

### The vocab singleton problem

The TAMER codebase uses a **global vocab singleton** — a single `CROHMEVocab` object shared across the entire process. Both models share this singleton but require different mappings (248 vs 334 tokens). Running v1 inference with the 334-token vocab active would cause index-out-of-bounds errors on the embedding lookup.

The solution: before each model's beam search call, `set_vocab(words)` overwrites the singleton's `word2idx` and `idx2word` dictionaries in-place with the correct word list for that model. Since inference is sequential (v1 then v4), the swap is safe without locks.

### Image preprocessing

Images are converted to grayscale, passed through `ToTensor()` and normalised:
```
mean = 0.7931,  std = 0.1738
```
A zero mask (all valid, no padding) is created since single-image inference has no batching.

### Inference flow per model
```
uploaded image
      ↓
grayscale + normalize → img_tensor [1, 1, H, W]
      ↓
set_vocab(words_vN)          ← swap global vocab
      ↓
model.tamer_model.beam_search(img_tensor, mask, **hparams)
      ↓
hyps[0].seq                  ← best hypothesis token indices
      ↓
vocab.indices2label(seq)     ← indices → LaTeX string
      ↓
returned as JSON → rendered in browser via MathJax
```

### Frontend
Pure HTML/CSS/JS, served as a template string from Flask. Key behaviours:
- Image preview on file select
- Both model cards appear simultaneously after conversion
- MathJax 3 renders the LaTeX string as typeset mathematics in the browser
- Copy-to-clipboard button per model output

---

## Notes

- The vocabulary is **flat token-level** — each LaTeX symbol (`\frac`, `{`, `x`, `^`) is one token. No subword tokenisation.
- Maximum output sequence length is 200 tokens (covers all expressions in CROHME and MathWriting).
- The padding mask travels as a separate boolean tensor throughout the model — it is never concatenated with image features. It is consumed at three points: 2D positional encoding (cumsum over valid pixels only), cross-attention (`memory_key_padding_mask`), and ARM (zero out padded positions in the coverage heatmap).

