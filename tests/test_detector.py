"""Tests for the adversarial attack detector."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector import detect_attacks


class TestLeetspeak:
    def test_basic_leet(self):
        result = detect_attacks("y0u r3 an 1d10t")
        assert result["attack_detected"] is True
        assert "leetspeak" in result["attack_types"]

    def test_heavy_leet(self):
        result = detect_attacks("5tup1d m0r0n")
        assert "leetspeak" in result["attack_types"]

    def test_clean_text_no_leet(self):
        result = detect_attacks("you are so nice today")
        assert "leetspeak" not in result["attack_types"]

    def test_leet_with_number_words(self):
        # "I have 10 apples" — numbers here but not leet proportion
        result = detect_attacks("I will be there in 10 minutes")
        # May or may not trigger — just check it returns a valid dict
        assert "attack_detected" in result
        assert "attack_types" in result


class TestCharRepetition:
    def test_repeated_chars(self):
        result = detect_attacks("stuuuuupid")
        assert result["attack_detected"] is True
        assert "char_repetition" in result["attack_types"]

    def test_idiottttt(self):
        result = detect_attacks("idiottttt")
        assert "char_repetition" in result["attack_types"]

    def test_normal_double(self):
        # "cool" and "look" have doubles but not triples
        result = detect_attacks("cool looking book")
        assert "char_repetition" not in result["attack_types"]


class TestSpacingFragmentation:
    def test_space_fragmented(self):
        result = detect_attacks("i d i o t")
        assert result["attack_detected"] is True
        assert "spacing_fragmentation" in result["attack_types"]

    def test_dot_fragmented(self):
        result = detect_attacks("i.d.i.o.t")
        assert "spacing_fragmentation" in result["attack_types"]

    def test_dash_fragmented(self):
        result = detect_attacks("i-d-i-o-t")
        assert "spacing_fragmentation" in result["attack_types"]

    def test_normal_sentence(self):
        result = detect_attacks("Have a great day")
        assert "spacing_fragmentation" not in result["attack_types"]


class TestUnicodeHomoglyph:
    def test_cyrillic_lookalike(self):
        # Cyrillic characters resembling Latin
        result = detect_attacks("уоu аrе аn іdіоt")
        assert result["attack_detected"] is True
        assert "unicode_homoglyph" in result["attack_types"]

    def test_ascii_clean(self):
        result = detect_attacks("you are clean")
        assert "unicode_homoglyph" not in result["attack_types"]


class TestEmojiSignal:
    def test_skull_emoji(self):
        result = detect_attacks("you are an idiot 💀")
        assert result["attack_detected"] is True
        assert "emoji_signal" in result["attack_types"]

    def test_middle_finger(self):
        result = detect_attacks("go away 🖕")
        assert "emoji_signal" in result["attack_types"]

    def test_no_emoji(self):
        result = detect_attacks("you are an idiot")
        assert "emoji_signal" not in result["attack_types"]


class TestMixedAttack:
    def test_mixed_leet_emoji(self):
        result = detect_attacks("y0u r3 an 1d10t 💀")
        assert "mixed_attack" in result["attack_types"]

    def test_single_attack_not_mixed(self):
        result = detect_attacks("stuuuuupid")
        assert "mixed_attack" not in result["attack_types"]


class TestReturnStructure:
    def test_structure(self):
        result = detect_attacks("hello world")
        assert "attack_detected" in result
        assert "attack_types" in result
        assert "attack_details" in result
        assert isinstance(result["attack_detected"], bool)
        assert isinstance(result["attack_types"], list)
