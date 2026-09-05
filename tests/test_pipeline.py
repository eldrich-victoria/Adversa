"""Tests for the end-to-end pipeline output structure and behavior."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import run_pipeline


class TestPipelineStructure:
    """Verify that the pipeline returns the expected structure."""

    def test_returns_dict(self):
        result = run_pipeline("you are an idiot")
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = run_pipeline("hello world")
        required = {"original", "attack_info", "normalization", "retrieved", "classification"}
        assert required.issubset(result.keys())

    def test_original_preserved(self):
        text = "y0u'r3 an 1d10t 💀"
        result = run_pipeline(text)
        assert result["original"] == text

    def test_attack_info_structure(self):
        result = run_pipeline("y0u r3 4n 1d10t")
        ai = result["attack_info"]
        assert "attack_detected" in ai
        assert "attack_types" in ai
        assert isinstance(ai["attack_detected"], bool)

    def test_normalization_structure(self):
        result = run_pipeline("5tup1d")
        n = result["normalization"]
        assert "original" in n
        assert "normalized" in n
        assert "transformations" in n

    def test_retrieved_list(self):
        result = run_pipeline("idiot")
        assert isinstance(result["retrieved"], list)
        assert len(result["retrieved"]) > 0

    def test_retrieved_item_keys(self):
        result = run_pipeline("you are an idiot")
        item = result["retrieved"][0]
        assert "text" in item
        assert "label" in item
        assert "similarity" in item
        assert "category" in item

    def test_classification_structure(self):
        result = run_pipeline("you are an idiot")
        clf = result["classification"]
        assert "label" in clf
        assert "confidence" in clf
        assert "category" in clf
        assert "explanation" in clf

    def test_label_values(self):
        result = run_pipeline("you are an idiot")
        assert result["classification"]["label"] in ("toxic", "non-toxic")

    def test_confidence_range(self):
        result = run_pipeline("you are an idiot")
        conf = result["classification"]["confidence"]
        assert 0.0 <= conf <= 1.0


class TestPipelineBehavior:
    """Sanity-check the pipeline's directional behavior."""

    def test_clear_toxic_detected(self):
        result = run_pipeline("you are an idiot")
        assert result["classification"]["label"] == "toxic"

    def test_leet_toxic_detected(self):
        result = run_pipeline("y0u r3 an 1d10t")
        # Attack should be detected
        assert result["attack_info"]["attack_detected"] is True
        # And the result should be toxic
        assert result["classification"]["label"] == "toxic"

    def test_benign_not_toxic(self):
        result = run_pipeline("Have a wonderful day!")
        assert result["classification"]["label"] == "non-toxic"

    def test_explanation_non_empty(self):
        result = run_pipeline("stuuuuupid")
        assert len(result["classification"]["explanation"]) > 10
