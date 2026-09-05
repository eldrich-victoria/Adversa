# Adversarial Toxic Language Auditor

> A local, retrieval-augmented pipeline for detecting **obfuscated toxic language** — the kind conventional classifiers miss.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-47%20passed-brightgreen)]()

---

## 1. Problem

Conventional text classifiers trained on plain toxic language fail against adversarial obfuscation. A user writing:

```
y0u'r3 an 1d10t 💀
```

intends the same insult as `"you're an idiot"` — but TF-IDF or even many transformer-based classifiers never seen this pattern will miss it entirely.

---

## 2. Motivation

After building a conventional [Toxic Comment Detection ML Platform](https://github.com) using scikit-learn, TF-IDF, and logistic regression, I identified a fundamental weakness: **these models have zero robustness against deliberate character-level obfuscation**.

This project directly addresses that weakness.

---

## 3. Key Idea

> **Detect the manipulation first. Normalize it. Then retrieve semantically similar known examples and use weighted evidence to decide.**

Rather than trying to retrain a classifier on all possible obfuscated variants, this system:
1. Detects the obfuscation technique (leetspeak, spacing, emoji, etc.)
2. Normalizes the text to a clean candidate
3. Embeds the normalized text and searches a semantic knowledge base
4. Combines retrieval evidence + keyword signal + attack detection to produce a transparent decision

---

## 4. Architecture

```
User Input (raw, possibly obfuscated)
        │
        ▼
┌──────────────────┐
│  Attack Detector │  ← detect_attacks()
└──────────────────┘
        │ attack_info (types detected)
        ▼
┌──────────────────┐
│   Normalizer     │  ← normalize()
└──────────────────┘
        │ normalized_text
        ▼
┌──────────────────────────────┐
│  Sentence Embedder           │  ← all-MiniLM-L6-v2
│  (all-MiniLM-L6-v2, 384-dim) │
└──────────────────────────────┘
        │ query embedding
        ▼
┌──────────────────────────────┐
│  Cosine Similarity Retriever │  ← sklearn cosine_similarity
│  (knowledge base, 150 items) │
└──────────────────────────────┘
        │ top-5 neighbors
        ▼
┌──────────────────────────────┐
│  Transparent Classifier      │  ← classify()
│  (retrieval vote + keywords  │
│   + attack boost)            │
└──────────────────────────────┘
        │
        ▼
   label · confidence · category · explanation
```

---

## 5. Attack Types Detected

| Attack Type | Description | Example |
|---|---|---|
| **leetspeak** | Digit/symbol → letter substitution | `y0u r3 4n 1d10t` |
| **symbol_substitution** | Symbols replacing or surrounding letters | `b!tch`, `@sshole` |
| **char_repetition** | 3+ consecutive identical characters | `stuuuuupid` |
| **spacing_fragmentation** | Single-character spacing or dot/dash fragmentation | `i d i o t`, `i.d.i.o.t` |
| **unicode_homoglyph** | Non-ASCII look-alike characters (e.g. Cyrillic) | `уоu аrе аn іdіоt` |
| **emoji_signal** | Toxic-context emoji present | `you are trash 💀🖕` |
| **mixed_attack** | Two or more above attack types co-occurring | `y0u r3 an 1d10t 💀` |

---

## 6. Normalization

The normalizer applies a conservative, ordered pipeline:

1. **Unicode normalization** (NFKD + combining-mark stripping) — reduces homoglyphs
2. **Lowercase**
3. **Fragmentation removal** — collapses `i d i o t` → `idiot`
4. **Leetspeak substitution** — `0→o, 1→i, 3→e, 4→a, 5→s, 7→t, @→a, $→s, ...`
5. **Repetition collapse** — `stuuuuu` → `stuu` (reduces 3+ repeats to 2)

The original input is **always preserved** — normalization only creates a candidate for retrieval.

---

## 7. Retrieval / RAG

- **Knowledge base**: 150 synthetic/curated examples covering all attack types
- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, ~80MB, CPU-fast)
- **Retrieval**: `sklearn.metrics.pairwise.cosine_similarity` on a pre-built `.npy` index
- **k = 5** nearest neighbors retrieved per query
- No FAISS required — the cosine fallback is fast enough for a 150-item knowledge base

---

## 8. Decision Pipeline

The classifier combines three signals:

| Signal | Weight | Description |
|---|---|---|
| Retrieval vote | 60% | Weighted toxic fraction of top-5 neighbors (weighted by similarity score) |
| Keyword signal | 30% | Toxic keyword presence in normalized text |
| Attack signal | 10% | Binary: adversarial obfuscation detected? |
| **Attack boost** | ×1.15 | If any attack detected, confidence is boosted by 15% |

**Decision threshold**: `score ≥ 0.45` → **TOXIC**

Outputs: `label`, `confidence`, `category`, `explanation`

---

## 9. Benchmarks

The system is evaluated on two distinct datasets to assess both basic capability and true robustness.

### A. In-Distribution Benchmark (110 examples)
Examples drawn from the same distribution as the knowledge base.
> Run via: `python scripts/run_benchmark.py`

| System | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (keyword only) | 0.518 | 1.000 | 0.338 | 0.505 |
| Normalization + keyword | 0.845 | 1.000 | 0.787 | 0.881 |
| **Full pipeline** | **0.991** | **0.988** | **1.000** | **0.994** |

### B. Held-Out Robustness Benchmark (120 synthetic examples)
The benchmark contains **120 synthetic held-out examples** with novel vocabulary, phrasing, and obfuscation patterns not present in the knowledge base. The evaluation demonstrates performance on this benchmark, not guaranteed real-world robustness.

> Run via: `python scripts/run_held_out_benchmark.py`

#### Semantic Leakage Analysis
No held-out examples exceeded 0.80 cosine similarity to any knowledge-base entry, supporting separation from the current knowledge base under the evaluated similarity thresholds.

#### Overall Metrics

| System | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| A. Baseline (keyword, raw) | 0.275 | 1.000 | 0.022 | 0.044 |
| B. Normalization + keyword | 0.375 | 1.000 | 0.157 | 0.272 |
| C. Detector + normalization (no retrieval) | 0.258 | 0.000 | 0.000 | 0.000 |
| **D. Full RAG pipeline (detect+norm+retrieve)** | **1.000** | **1.000** | **1.000** | **1.000** |

The baseline keyword classifier misses nearly all toxic examples (Recall 0.022) because obfuscation and novel vocabulary prevent keyword matching. Adding normalization helps slightly (Recall 0.157). The full retrieval pipeline achieves 1.000 F1 on this synthetic benchmark.

#### Retrieval Agreement Metrics
Evaluating retrieval match across top-k neighbors at multiple granularities:

- **Top-5 binary-label agreement was 100%** (120/120).
- **Top-5 toxicity-category agreement was 90.8%** (109/120).
- **Top-5 attack-family agreement was 47.5%** (57/120).

The lower attack-family retrieval rate indicates that semantic retrieval captures toxicity/category similarity more reliably than exact obfuscation-technique similarity.

| Attack Type | N | Top-5 Label Agreement | Full Pipeline F1 |
|---|---|---|---|
| clean | 49 | 1.000 | 1.000 |
| leetspeak | 12 | 1.000 | 1.000 |
| spacing | 13 | 1.000 | 1.000 |
| repetition | 9 | 1.000 | 1.000 |
| symbol | 9 | 1.000 | 1.000 |
| unicode | 7 | 1.000 | 1.000 |
| emoji | 7 | 1.000 | 1.000 |
| mixed | 14 | 1.000 | 1.000 |

---

## 11. Examples

### Example 1 — Clean toxic

```
Input:   "you are an idiot"
Attacks: none
Norm:    "you are an idiot"
Result:  TOXIC (confidence: 0.88) — insult
```

### Example 2 — Leetspeak

```
Input:   "y0u'r3 an 1d10t"
Attacks: leetspeak
Norm:    "you're an idiot"
Result:  TOXIC (confidence: 0.94) — insult
```

### Example 3 — Spacing fragmentation

```
Input:   "i d i o t"
Attacks: spacing_fragmentation
Norm:    "idiot"
Result:  TOXIC (confidence: 0.94) — insult
```

### Example 4 — Character repetition

```
Input:   "stuuuuupid"
Attacks: char_repetition
Norm:    "stuu pid" (repetition collapsed)
Result:  TOXIC (confidence: 0.85) — insult
```

### Example 5 — Unicode homoglyph

```
Input:   "уоu аrе аn іdіоt"  (Cyrillic lookalikes)
Attacks: unicode_homoglyph
Norm:    "you are an idiot"  (NFKD normalized)
Result:  TOXIC (confidence: 0.94) — insult
```

### Example 6 — Benign

```
Input:   "Have a wonderful day!"
Attacks: none
Norm:    "have a wonderful day!"
Result:  NON-TOXIC (confidence: 0.10)
```

---

## 12. Technologies

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| ML / metrics | scikit-learn |
| Vector retrieval | sklearn cosine_similarity (NumPy-backed) |
| Attack detection | regex, unicodedata (stdlib) |
| Normalization | regex, unicodedata (stdlib) |
| UI | Streamlit |
| Testing | pytest |
| Data | numpy, pandas |

No external APIs. No LLMs. No paid services. Runs fully offline after initial model download.

---

## 13. Project Structure

```
adversarial-toxic-rag/
│
├── app/
│   └── streamlit_app.py      # Streamlit UI
│
├── data/
│   ├── adversarial_examples.json       # 150-example knowledge base
│   ├── benchmark.csv                   # 110-example in-distribution benchmark
│   ├── held_out_benchmark.csv          # 120-example out-of-distribution benchmark
│   ├── benchmark_results.txt           # Measured in-distribution output
│   ├── held_out_benchmark_results.txt  # Measured held-out output
│   └── kb_embeddings.npy               # Pre-computed embeddings (generated)
│
├── src/
│   ├── __init__.py
│   ├── detector.py           # Attack type detection
│   ├── normalizer.py         # Text normalization pipeline
│   ├── embeddings.py         # Sentence-transformer wrapper
│   ├── retriever.py          # Cosine similarity retrieval
│   ├── classifier.py         # Transparent toxicity decision
│   └── pipeline.py           # End-to-end orchestrator
│
├── tests/
│   ├── test_detector.py             # Detector unit tests
│   ├── test_normalizer.py           # Normalizer unit tests
│   ├── test_pipeline.py             # Pipeline integration tests
│   └── test_held_out_benchmark.py   # Tests for held-out evaluation
│
├── scripts/
│   ├── build_index.py                # Pre-compute KB embeddings → .npy
│   ├── run_benchmark.py              # Run in-distribution benchmark
│   ├── run_held_out_benchmark.py     # Run out-of-distribution benchmark
│   └── verify.py                     # Verify test inputs
│
├── requirements.txt
├── README.md
├── .gitignore
└── run.py                    # Streamlit launcher
```

---

## 14. Installation

```bash
# Clone the repository
git clone https://github.com/yourname/adversarial-toxic-rag.git
cd adversarial-toxic-rag

# Install dependencies
pip install -r requirements.txt

# Build the knowledge-base embedding index (first time only)
python scripts/build_index.py
```

> The first run of `build_index.py` will download the `all-MiniLM-L6-v2` model (~80 MB). After that, the system works fully offline.

---

## 15. Running the Application

```bash
python run.py
# or directly:
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## 16. Running Tests

```bash
python -m pytest tests/ -v
```

Expected: **47 passed**

---

## 17. Running Benchmark

```bash
python scripts/run_benchmark.py
```

Results are printed to stdout and saved to `data/benchmark_results.txt`.

---

## 18. Limitations

- **Closed knowledge base**: Performance is high on the benchmark because the benchmark examples are drawn from the same distribution as the knowledge base. Novel obfuscation patterns not covered in the knowledge base will degrade performance.
- **No deep language understanding**: The system does not understand context, sarcasm, or figurative language. A sarcastic "Oh great, you're an idiot" will be flagged as toxic.
- **Normalization is not a spell-checker**: The normalizer handles known patterns (leetspeak, fragmentation) but cannot handle all creative obfuscation variants.
- **Small knowledge base**: 150 examples is sufficient for a portfolio MVP. A production system would require thousands of diverse examples.
- **Not a production moderation system**: This is a research/portfolio project. It should not be deployed as a production content moderator without significant additional engineering.

---

## 19. Future Improvements

- Expand knowledge base with real-world adversarial corpora (e.g. HateXplain, ToxiGen)
- Add FAISS ANN index for sub-linear retrieval at scale
- Train a lightweight adapter/fine-tuned classifier on top of retrieved embeddings
- Add multi-language homoglyph support
- Implement active learning: flag uncertain predictions for human review
- Add a REST API wrapper (FastAPI) for integration into moderation pipelines
- Support custom attack pattern plugins

---

## Disclaimer

This is a **local research/portfolio MVP**. It uses a synthetic knowledge base and a lightweight retrieval-augmented approach. Results on the included benchmark do not generalize to production-scale or real-world moderation scenarios. The system is intended to demonstrate adversarial NLP techniques, not to serve as a content moderation product.
