"""
embeddings.py — Sentence Embedding Utilities

Uses sentence-transformers (all-MiniLM-L6-v2) to encode text into
dense semantic vectors. The model is cached so it is loaded only once
per process/Streamlit session.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level cache so the model is loaded once
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return the cached sentence-transformer model, loading it if necessary."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode(texts: list[str] | str, *, batch_size: int = 64, normalize: bool = True) -> np.ndarray:
    """
    Encode one or more texts into L2-normalized embedding vectors.

    Parameters
    ----------
    texts : str or list[str]
        Input text(s) to encode.
    batch_size : int
        Encoding batch size (only relevant for large lists).
    normalize : bool
        If True, L2-normalize embeddings (required for cosine similarity via dot product).

    Returns
    -------
    np.ndarray of shape (n, 384)
    """
    model = get_model()
    if isinstance(texts, str):
        texts = [texts]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    return np.array(embeddings, dtype=np.float32)


def encode_query(text: str) -> np.ndarray:
    """Convenience wrapper: encode a single query string → shape (384,)."""
    return encode([text])[0]
