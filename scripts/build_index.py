"""
build_index.py — Pre-compute and cache knowledge-base embeddings.

Run this once before using the application or benchmark:
    python scripts/build_index.py

Output:
    data/kb_embeddings.npy  — float32 array of shape (N, 384)
"""

import sys
import json
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from src.embeddings import encode, get_model

KB_PATH = ROOT / "data" / "adversarial_examples.json"
OUT_PATH = ROOT / "data" / "kb_embeddings.npy"


def main() -> None:
    print("Loading knowledge base...")
    with open(KB_PATH, encoding="utf-8") as f:
        records = json.load(f)

    print(f"  {len(records)} records loaded.")

    # Encode the normalized text for each example
    texts = [rec["normalized"] for rec in records]

    print(f"Loading embedding model ({get_model().get_sentence_embedding_dimension()}-dim)...")
    embeddings = encode(texts, batch_size=64, normalize=True)

    print(f"Embeddings shape: {embeddings.shape}")
    np.save(str(OUT_PATH), embeddings)
    print(f"Saved embeddings -> {OUT_PATH}")


if __name__ == "__main__":
    main()
