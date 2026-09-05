"""
test_held_out_benchmark.py — Tests for held-out benchmark evaluation utilities.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_held_out_benchmark import (
    evaluate_metrics,
    evaluate_retrieval,
    analyze_semantic_leakage,
    predict_baseline,
    predict_normalized,
    predict_detect_norm_only,
    predict_full,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
HELD_OUT_PATH = ROOT / "data" / "held_out_benchmark.csv"
KB_PATH       = ROOT / "data" / "adversarial_examples.json"


@pytest.fixture(scope="module")
def held_out_df() -> pd.DataFrame:
    return pd.read_csv(HELD_OUT_PATH).reset_index(drop=True)

@pytest.fixture(scope="module")
def sample_df(held_out_df) -> pd.DataFrame:
    return held_out_df.head(5)

# ---------------------------------------------------------------------------
# Semantic Leakage Tests
# ---------------------------------------------------------------------------

class TestSemanticLeakage:
    def test_semantic_leakage_output_structure(self, sample_df):
        stats, df = analyze_semantic_leakage(sample_df)
        
        # Test stats
        assert "mean" in stats
        assert "median" in stats
        assert "min" in stats
        assert "max" in stats
        assert "std" in stats
        assert "pct_gt_80" in stats
        
        for k, v in stats.items():
            assert isinstance(v, float)
            
        # Test DataFrame
        assert len(df) == len(sample_df)
        required_cols = {
            "query_id", "query_text", "query_label", "max_similarity",
            "nearest_kb_id", "nearest_kb_text", "nearest_kb_label",
            "nearest_kb_category", "nearest_kb_attack"
        }
        assert required_cols.issubset(df.columns)

# ---------------------------------------------------------------------------
# Retrieval Evaluation Tests
# ---------------------------------------------------------------------------

class TestRetrievalEvaluation:
    def test_retrieval_output_structure(self, sample_df):
        results = evaluate_retrieval(sample_df)
        
        expected_keys = {
            "top1_label", "top5_label",
            "top1_category", "top5_category",
            "top1_attack", "top5_attack",
            "total"
        }
        assert expected_keys.issubset(results.keys())
        assert results["total"] == len(sample_df)
        
        for k in expected_keys:
            if k != "total":
                assert 0 <= results[k] <= results["total"]

# ---------------------------------------------------------------------------
# Ablation Prediction Tests
# ---------------------------------------------------------------------------

class TestAblationPredictions:
    def test_predict_detect_norm_only_returns_valid_label(self):
        label = predict_detect_norm_only("you are a fool")
        assert label in ("toxic", "benign")

    def test_predict_detect_norm_only_no_retrieval_influence(self):
        # Even with no retrieval, a high keyword score with attack boost can trigger toxic
        label = predict_detect_norm_only("y0u 4r3 4n 1d10t stup1d l0s3r")
        # In this specific case, 3 keywords (idiot, stupid, loser) -> keyword_score = 1.0. 
        # base_score = 0.3 * 1.0 + 0.10 = 0.40. With attack boost: 0.40 * 1.15 = 0.46 -> toxic!
        assert label == "toxic"

# ---------------------------------------------------------------------------
# Existing Tests
# ---------------------------------------------------------------------------

class TestNoLeakage:
    def test_no_exact_kb_overlap(self, held_out_df):
        with open(KB_PATH, encoding="utf-8") as f:
            records = json.load(f)
        kb_texts = {r["text"].strip().lower() for r in records}
        overlaps = []
        for _, row in held_out_df.iterrows():
            if row["text"].strip().lower() in kb_texts:
                overlaps.append(row["text"])
        assert len(overlaps) == 0

class TestEvaluateMetrics:
    def test_perfect_predictions(self):
        y = ["toxic", "toxic", "benign", "benign"]
        m = evaluate_metrics(y, y)
        assert m["accuracy"] == pytest.approx(1.0)
