"""
detector.py — Adversarial Attack Detector

Detects common text obfuscation and adversarial manipulation patterns
used to evade conventional toxicity classifiers.

Supported attack types:
  - leetspeak        : digit/symbol replacements (a→4, e→3, i→1, o→0, etc.)
  - symbol_subst     : non-alphanumeric characters substituted for letters
  - char_repetition  : excessive repeated characters (aaaa, !!!!!)
  - spacing          : deliberate spacing/fragmentation (i d i o t)
  - unicode_homoglyph: non-ASCII lookalike characters
  - emoji_signal     : emoji commonly associated with abusive context
  - mixed_attack     : two or more attack types co-occurring
"""

import re
import unicodedata
from typing import Any

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

# Common leet substitutions: character(s) → latin letter
LEET_MAP: dict[str, str] = {
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
    "6": "g", "7": "t", "8": "b", "9": "g", "@": "a",
    "$": "s", "|": "l",
}

# Regex: digits/special chars that are typical leet substitutes
_LEET_CHARS = re.compile(r"[0134578@$|]")

# Regex: letter-punctuation-letter fragmentation patterns ("i.d.i.o.t", "i d i o t")
_FRAG_PATTERN = re.compile(
    r"\b([a-zA-Z][\s.\-_,]{1,3}){3,}[a-zA-Z]\b"
)

# Regex: three or more identical consecutive characters
_REPETITION_PATTERN = re.compile(r"(.)\1{2,}")

# Toxic-context emoji (a curated subset — not exhaustive)
TOXIC_EMOJI: set[str] = {
    "💀", "🖕", "😡", "🤬", "👊", "💢", "😤", "🔫", "💣", "☠️",
    "🤮", "🤡", "🤣", "😈", "👿", "🤫", "🤥", "😒", "😑", "🙄",
}


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

def _has_leet(text: str) -> bool:
    """Return True if the text contains typical leet substitution characters."""
    # Count leet chars as proportion of alphabetic+digit content
    leet_hits = len(_LEET_CHARS.findall(text))
    alpha = sum(c.isalpha() for c in text)
    if alpha == 0:
        return False
    return leet_hits >= 1 and (leet_hits / (alpha + leet_hits)) >= 0.05


def _has_symbol_substitution(text: str) -> bool:
    """Return True if non-standard symbols appear in word positions."""
    # Look for symbols that interrupt or replace letters in a word context
    pattern = re.compile(r"[a-zA-Z][^a-zA-Z\s]{1,2}[a-zA-Z]")
    return bool(pattern.search(text))


def _has_char_repetition(text: str) -> bool:
    """Return True if any character repeats 3+ times consecutively."""
    return bool(_REPETITION_PATTERN.search(text))


def _has_spacing_fragmentation(text: str) -> bool:
    """Return True if deliberate character-level spacing/fragmentation is present."""
    return bool(_FRAG_PATTERN.search(text))


def _has_unicode_homoglyph(text: str) -> bool:
    """Return True if the text contains non-ASCII characters that resemble Latin letters."""
    for char in text:
        if ord(char) > 127:
            # Check if it's a letter-like character but not standard ASCII
            cat = unicodedata.category(char)
            name = unicodedata.name(char, "")
            if cat.startswith("L") and "LATIN" not in name and "DIGIT" not in name:
                return True
    return False


def _has_emoji_signal(text: str) -> bool:
    """Return True if any toxic-context emoji is present."""
    for char in text:
        if char in TOXIC_EMOJI:
            return True
    # Also catch via unicode categories: So (Symbol,other) and emoji ranges
    for char in text:
        cp = ord(char)
        if (0x1F600 <= cp <= 0x1F64F or  # emoticons
                0x1F300 <= cp <= 0x1F5FF or  # misc symbols
                0x1F680 <= cp <= 0x1F6FF or  # transport
                0x2600 <= cp <= 0x26FF or    # misc symbols
                0x2700 <= cp <= 0x27BF or    # dingbats
                0xFE00 <= cp <= 0xFE0F):     # variation selectors
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_attacks(text: str) -> dict[str, Any]:
    """
    Analyze *text* for adversarial obfuscation patterns.

    Returns
    -------
    dict with keys:
        attack_detected : bool
        attack_types    : list[str]   — names of detected attack types
        attack_details  : dict        — per-type boolean flags
    """
    details: dict[str, bool] = {
        "leetspeak": _has_leet(text),
        "symbol_substitution": _has_symbol_substitution(text),
        "char_repetition": _has_char_repetition(text),
        "spacing_fragmentation": _has_spacing_fragmentation(text),
        "unicode_homoglyph": _has_unicode_homoglyph(text),
        "emoji_signal": _has_emoji_signal(text),
    }
    attack_types = [k for k, v in details.items() if v]

    # Mixed attack = 2 or more individual types detected
    if len(attack_types) >= 2:
        attack_types.append("mixed_attack")

    return {
        "attack_detected": bool(attack_types),
        "attack_types": attack_types,
        "attack_details": details,
    }
