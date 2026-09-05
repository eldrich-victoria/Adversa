"""
pipeline.py — End-to-End Adversarial Toxicity Detection Pipeline

Orchestrates: Detector → Normalizer → Retriever → Classifier

Usage:
    from src.pipeline import run_pipeline
    result = run_pipeline("y0u'r3 an 1d10t 💀")
"""

from __future__ import annotations

from typing import Any

from src.detector import detect_attacks
from src.normalizer import normalize
from src.retriever import retrieve
from src.classifier import classify


def run_pipeline(text: str, k: int = 5) -> dict[str, Any]:
    """
    Run the full adversarial toxicity detection pipeline.

    Parameters
    ----------
    text : str
        Raw user input (may be obfuscated).
    k : int
        Number of retrieved neighbors to use for classification.

    Returns
    -------
    dict with keys:
        original         : str
        attack_info      : dict   (from detector)
        normalization    : dict   (from normalizer)
        retrieved        : list   (from retriever)
        classification   : dict   (from classifier)
    """
    # Step 1: Detect adversarial attacks on the raw input
    attack_info = detect_attacks(text)

    # Step 2: Normalize the text
    normalization = normalize(text)
    normalized_text = normalization["normalized"]

    # Step 3: Retrieve semantically similar examples using the normalized text
    retrieved = retrieve(normalized_text, k=k)

    # Step 4: Classify based on retrieved context + attack info
    classification = classify(normalized_text, retrieved, attack_info)

    return {
        "original": text,
        "attack_info": attack_info,
        "normalization": normalization,
        "retrieved": retrieved,
        "classification": classification,
    }
