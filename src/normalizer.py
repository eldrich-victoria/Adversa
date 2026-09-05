"""
normalizer.py — Text Normalization for Adversarial Inputs

Converts obfuscated adversarial text into a clean normalized candidate
while preserving the original. Conservative approach — does not attempt
sophisticated spell-correction.
"""

import re
import unicodedata
from typing import Any

# ---------------------------------------------------------------------------
# Leet → Latin mapping (single-char substitutions)
# ---------------------------------------------------------------------------
LEET_TO_LATIN: dict[str, str] = {
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
    "6": "g", "7": "t", "8": "b", "@": "a", "$": "s",
    "|": "l",
}

# Compiled once for efficiency
_LEET_RE = re.compile("|".join(re.escape(k) for k in LEET_TO_LATIN))
_REPEAT_RE = re.compile(r"(.)\1{2,}")


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def _unicode_normalize(text: str) -> tuple[str, bool]:
    """
    Apply NFKD decomposition then strip combining marks to approximate
    ASCII equivalents of homoglyph characters.
    """
    normalized = unicodedata.normalize("NFKD", text)
    result = "".join(c for c in normalized if not unicodedata.combining(c))
    changed = result != text
    return result, changed


def _apply_leet(text: str) -> tuple[str, bool]:
    """Replace common leet substitution characters with their Latin equivalents."""
    result = _LEET_RE.sub(lambda m: LEET_TO_LATIN[m.group(0)], text)
    changed = result != text
    return result, changed


def _collapse_repetition(text: str) -> tuple[str, bool]:
    """Collapse 3+ consecutive identical characters down to 2 (preserves doubles like 'cool')."""
    result = _REPEAT_RE.sub(lambda m: m.group(1) * 2, text)
    changed = result != text
    return result, changed


def _remove_fragmentation(text: str) -> tuple[str, bool]:
    """
    Remove deliberate single-character spacing fragmentation.
    Conservative: only collapse patterns clearly matching fragmented words.
    e.g.  'i d i o t'  →  'idiot'
          'i.d.i.o.t'  →  'idiot'
    """
    # Pattern: single letters separated by a consistent delimiter (space, dot, dash, underscore)
    frag_pattern = re.compile(r"\b([a-zA-Z])([\s.\-_,])(?:[a-zA-Z]\2){2,}[a-zA-Z]\b")

    def _join_frag(m: re.Match) -> str:
        delim = re.escape(m.group(2))
        return re.sub(delim, "", m.group(0)).replace(" ", "")

    result = frag_pattern.sub(_join_frag, text)
    changed = result != text
    return result, changed


def _lowercase(text: str) -> tuple[str, bool]:
    result = text.lower()
    return result, result != text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize(text: str) -> dict[str, Any]:
    """
    Normalize adversarial text through a conservative pipeline.

    Processing order:
      1. Unicode normalization (NFKD homoglyph reduction)
      2. Lowercase
      3. Fragmentation removal
      4. Leet substitution
      5. Repetition collapse

    Returns
    -------
    dict with keys:
        original        : str   — unchanged input
        normalized      : str   — cleaned text
        transformations : list[str]  — names of applied transforms
    """
    original = text
    current = text
    transformations: list[str] = []

    steps: list[tuple[str, Any]] = [
        ("unicode_normalization", _unicode_normalize),
        ("lowercase", _lowercase),
        ("fragmentation_removal", _remove_fragmentation),
        ("leetspeak_normalization", _apply_leet),
        ("repetition_collapse", _collapse_repetition),
    ]

    for name, fn in steps:
        current, changed = fn(current)
        if changed:
            transformations.append(name)

    return {
        "original": original,
        "normalized": current,
        "transformations": transformations,
    }
