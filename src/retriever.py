"""
retriever.py — Semantic Similarity Retrieval

Loads the knowledge-base embeddings from disk (built by scripts/build_index.py)
and retrieves the top-k most semantically similar examples for a given query.

Uses sklearn cosine_similarity as the primary backend (FAISS-free, runs on CPU).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.embeddings import encode_query

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"
KB_PATH = DATA_DIR / "adversarial_examples.json"
EMB_PATH = DATA_DIR / "kb_embeddings.npy"

# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------

_kb_records: list[dict] | None = None
_kb_embeddings: np.ndarray | None = None


def _load_index() -> tuple[list[dict], np.ndarray]:
    """Load knowledge-base records and embeddings from disk (cached in module globals)."""
    global _kb_records, _kb_embeddings

    if _kb_records is None or _kb_embeddings is None:
        with open(KB_PATH, encoding="utf-8") as f:
            _kb_records = json.load(f)

        if not EMB_PATH.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: {EMB_PATH}\n"
                "Run  python scripts/build_index.py  first."
            )
        _kb_embeddings = np.load(str(EMB_PATH)).astype(np.float32)

    return _kb_records, _kb_embeddings


def retrieve(query: str, k: int = 5) -> list[dict]:
    """
    Retrieve the top-k knowledge-base examples most similar to *query*.

    Parameters
    ----------
    query : str
        The (normalized) query text.
    k : int
        Number of results to return.

    Returns
    -------
    list of dicts with keys:
        text, normalized, attack_types, category, label, similarity
    """
    records, embeddings = _load_index()

    q_emb = encode_query(query).reshape(1, -1)          # (1, 384)
    sims = cosine_similarity(q_emb, embeddings)[0]      # (N,)

    top_k_idx = np.argsort(sims)[::-1][:k]

    results = []
    for idx in top_k_idx:
        rec = records[idx]
        results.append({
            "text": rec["text"],
            "normalized": rec["normalized"],
            "attack_types": rec["attack_types"],
            "category": rec["category"],
            "label": rec["label"],
            "similarity": float(sims[idx]),
        })

    return results
