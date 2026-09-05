"""
classifier.py — Transparent Local Toxicity Decision

Combines three signals to produce a toxicity decision:
  1. Semantic neighborhood vote from retrieved knowledge-base examples
  2. Normalization-based lexical signal (toxic keywords)
  3. Attack-detection boost (presence of adversarial obfuscation)

No LLM or external API is used. The logic is fully deterministic and auditable.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Toxic keyword lexicon (lightweight signal on normalized text)
# ---------------------------------------------------------------------------

TOXIC_KEYWORDS: set[str] = {
    "idiot", "stupid", "moron", "dumb", "loser", "worthless",
    "garbage", "trash", "disgusting", "hate", "kill", "die",
    "disappear", "failure", "idiocy", "shut up", "shutup",
    "pathetic", "ugly", "screw", "damn", "shit", "fuck",
    "bitch", "asshole", "crap", "hell", "bastard", "jerk",
    "retard", "freak", "dumbass", "dimwit", "imbecile",
    "nobody likes", "get lost", "go away", "leave me alone",
    "dead", "threat", "regret", "embarrassment",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _keyword_score(normalized_text: str) -> float:
    """Return fraction of toxic keywords present in the normalized text."""
    lower = normalized_text.lower()
    hits = sum(1 for kw in TOXIC_KEYWORDS if kw in lower)
    return min(hits / 3.0, 1.0)   # cap at 1.0; 3+ keywords → full signal


def _retrieval_vote(retrieved: list[dict]) -> tuple[float, float]:
    """
    Weighted vote from retrieved neighbors.

    Returns
    -------
    (toxic_score, total_weight)
        toxic_score : weighted proportion of toxic neighbors
    """
    if not retrieved:
        return 0.0, 0.0

    total_weight = sum(r["similarity"] for r in retrieved)
    if total_weight == 0:
        return 0.0, 0.0

    toxic_weight = sum(
        r["similarity"] for r in retrieved if r["label"] == "toxic"
    )
    return toxic_weight / total_weight, total_weight


def _infer_category(retrieved: list[dict], normalized: str) -> str:
    """Infer the most likely toxicity category from retrieved neighbors."""
    # Count categories from toxic neighbors only
    from collections import Counter
    cats = [r["category"] for r in retrieved if r["label"] == "toxic"]
    if not cats:
        return "benign"
    return Counter(cats).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(
    normalized_text: str,
    retrieved: list[dict],
    attack_info: dict,
) -> dict[str, Any]:
    """
    Produce a toxicity classification decision.

    Parameters
    ----------
    normalized_text : str
        Cleaned/normalized version of the user input.
    retrieved : list[dict]
        Top-k retrieved knowledge-base examples.
    attack_info : dict
        Output from detector.detect_attacks().

    Returns
    -------
    dict with keys:
        label         : "toxic" | "non-toxic"
        confidence    : float in [0, 1]
        category      : str
        explanation   : str
    """
    retrieval_score, retrieval_weight = _retrieval_vote(retrieved)
    keyword_score = _keyword_score(normalized_text)
    attack_detected = attack_info.get("attack_detected", False)

    # Combine signals: 60% retrieval, 30% keyword, 10% attack signal
    base_score = (
        0.60 * retrieval_score
        + 0.30 * keyword_score
        + 0.10 * (1.0 if attack_detected else 0.0)
    )

    # Apply adversarial attack boost — if obfuscation detected,
    # we assume the intent is more likely malicious
    if attack_detected:
        base_score = min(base_score * 1.15, 1.0)

    label = "toxic" if base_score >= 0.45 else "non-toxic"
    confidence = round(base_score, 3)
    category = _infer_category(retrieved, normalized_text)

    # Build explanation
    explanation_parts: list[str] = []

    if retrieval_weight > 0:
        toxic_neighbors = sum(1 for r in retrieved if r["label"] == "toxic")
        explanation_parts.append(
            f"Semantic retrieval: {toxic_neighbors}/{len(retrieved)} nearest neighbors "
            f"are labeled toxic (weighted vote score: {retrieval_score:.2f})."
        )
    if keyword_score > 0:
        explanation_parts.append(
            f"Lexical signal: normalized text contains toxic keyword pattern(s) "
            f"(signal strength: {keyword_score:.2f})."
        )
    if attack_detected:
        attack_types = ", ".join(attack_info.get("attack_types", []))
        explanation_parts.append(
            f"Adversarial obfuscation detected ({attack_types}): "
            f"confidence boosted by 15%."
        )
    if not explanation_parts:
        explanation_parts.append(
            "No strong toxic signals detected in retrieved neighbors or keywords."
        )

    explanation = " ".join(explanation_parts)

    return {
        "label": label,
        "confidence": confidence,
        "category": category if label == "toxic" else "benign",
        "explanation": explanation,
    }
