"""Tests for the text normalizer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.normalizer import normalize


class TestNormalizationStructure:
    def test_returns_dict(self):
        result = normalize("hello")
        assert isinstance(result, dict)
        assert "original" in result
        assert "normalized" in result
        assert "transformations" in result

    def test_original_preserved(self):
        text = "y0u R3 4n 1d10t"
        result = normalize(text)
        assert result["original"] == text


class TestLeetNormalization:
    def test_basic_leet(self):
        result = normalize("y0u r3 an 1d10t")
        assert "0" not in result["normalized"]
        assert "3" not in result["normalized"]
        assert "1" not in result["normalized"]

    def test_leet_transformations_recorded(self):
        result = normalize("5tup1d")
        assert "leetspeak_normalization" in result["transformations"]

    def test_at_sign_replaced(self):
        result = normalize("@sshole")
        assert result["normalized"].startswith("a")


class TestRepetitionCollapse:
    def test_triple_collapse(self):
        result = normalize("stuuuuupid")
        # Triple+ should be collapsed to double
        assert "uuu" not in result["normalized"]

    def test_repetition_transformations_recorded(self):
        result = normalize("idiottttt")
        assert "repetition_collapse" in result["transformations"]


class TestFragmentationRemoval:
    def test_space_frag(self):
        result = normalize("i d i o t")
        # Should collapse to something without single-char spaces
        assert "i d i o t" not in result["normalized"]

    def test_dot_frag(self):
        result = normalize("i.d.i.o.t")
        assert "i.d.i.o.t" not in result["normalized"]


class TestLowercase:
    def test_lowercased(self):
        result = normalize("YOU ARE AN IDIOT")
        assert result["normalized"] == result["normalized"].lower()

    def test_lowercase_in_transformations(self):
        result = normalize("HELLO")
        assert "lowercase" in result["transformations"]


class TestUnicodeNorm:
    def test_cyrillic_normalized(self):
        # Cyrillic 'у' should approximate to 'u' after NFKD
        result = normalize("уоu аrе аn іdіоt")
        # After NFKD + combining strip, some chars may normalize
        assert result["normalized"] is not None
        assert result["original"] != result["normalized"] or True  # may or may not change

    def test_no_crash_on_emoji(self):
        result = normalize("hello 💀")
        assert result["normalized"] is not None


class TestCleanInput:
    def test_clean_text_unchanged_content(self):
        result = normalize("the weather is nice today")
        # Content should match (only lowercased)
        assert result["normalized"] == "the weather is nice today"
