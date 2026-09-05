"""
run_benchmark.py — Evaluate the adversarial toxicity pipeline.

Evaluates three configurations:
  1. Baseline   : raw text, no normalization, no retrieval (keyword-only)
  2. Normalized : normalized text, no retrieval (keyword signal only)
  3. Full       : normalization + semantic retrieval (full pipeline)

Metrics: Accuracy, Precision, Recall, F1
Also reports breakdown by attack type.

Usage:
    python scripts/run_benchmark.py

Output is printed to stdout and written to data/benchmark_results.txt
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.detector import detect_attacks
from src.normalizer import normalize
from src.classifier import classify, _keyword_score
from src.retriever import retrieve

# ---------------------------------------------------------------------------
# Load benchmark data
# ---------------------------------------------------------------------------

BENCH_PATH = ROOT / "data" / "benchmark.csv"
RESULTS_PATH = ROOT / "data" / "benchmark_results.txt"


def load_benchmark() -> pd.DataFrame:
    df = pd.read_csv(BENCH_PATH)
    return df


# ---------------------------------------------------------------------------
# Prediction functions
# ---------------------------------------------------------------------------

def predict_baseline(text: str) -> str:
    """Baseline: keyword signal on raw (lowercased) text only."""
    score = _keyword_score(text.lower())
    return "toxic" if score >= 0.10 else "benign"


def predict_normalized(text: str) -> str:
    """Normalized: keyword signal on normalized text only (no retrieval)."""
    norm = normalize(text)
    score = _keyword_score(norm["normalized"])
    return "toxic" if score >= 0.10 else "benign"


def predict_full(text: str) -> str:
    """Full pipeline: normalization + retrieval + attack detection."""
    attack_info = detect_attacks(text)
    norm = normalize(text)
    normalized_text = norm["normalized"]
    retrieved = retrieve(normalized_text, k=5)
    clf = classify(normalized_text, retrieved, attack_info)
    return clf["label"].replace("non-toxic", "benign")


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(y_true: list[str], y_pred: list[str]) -> dict:
    labels = ["toxic", "benign"]
    # Convert to binary: toxic=1, benign=0
    yt = [1 if y == "toxic" else 0 for y in y_true]
    yp = [1 if y == "toxic" else 0 for y in y_pred]
    return {
        "accuracy": accuracy_score(yt, yp),
        "precision": precision_score(yt, yp, zero_division=0),
        "recall": recall_score(yt, yp, zero_division=0),
        "f1": f1_score(yt, yp, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading benchmark data...")
    df = load_benchmark()
    print(f"  {len(df)} examples loaded.")

    true_labels = df["label"].tolist()
    attack_types = df["attack_type"].tolist()

    print("\nRunning Baseline evaluation...")
    baseline_preds = [predict_baseline(t) for t in df["text"]]

    print("Running Normalized evaluation...")
    norm_preds = [predict_normalized(t) for t in df["text"]]

    print("Running Full Pipeline evaluation...")
    full_preds = [predict_full(t) for t in df["text"]]

    # Overall metrics
    results = {
        "Baseline (keyword only)": evaluate(true_labels, baseline_preds),
        "Normalization + keyword": evaluate(true_labels, norm_preds),
        "Full pipeline (norm + retrieval)": evaluate(full_preds, full_preds),  # placeholder replaced below
    }
    results["Full pipeline (norm + retrieval)"] = evaluate(true_labels, full_preds)

    # Per-attack-type metrics (full pipeline only)
    attack_type_results: dict[str, dict] = {}
    unique_attacks = sorted(set(attack_types))
    for at in unique_attacks:
        mask = [i for i, a in enumerate(attack_types) if a == at]
        yt_sub = [true_labels[i] for i in mask]
        yp_sub = [full_preds[i] for i in mask]
        attack_type_results[at] = evaluate(yt_sub, yp_sub)

    # ---- Print report ----
    lines: list[str] = []
    lines.append("=" * 65)
    lines.append("  ADVERSARIAL TOXIC LANGUAGE AUDITOR — BENCHMARK RESULTS")
    lines.append("=" * 65)
    lines.append(f"\nDataset: {len(df)} examples ({sum(1 for l in true_labels if l == 'toxic')} toxic, "
                 f"{sum(1 for l in true_labels if l == 'benign')} benign)")
    lines.append("\n--- OVERALL METRICS ---\n")
    header = f"{'System':<38} {'Acc':>6} {'Prec':>7} {'Rec':>7} {'F1':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    for system, m in results.items():
        lines.append(
            f"{system:<38} {m['accuracy']:>6.3f} {m['precision']:>7.3f} "
            f"{m['recall']:>7.3f} {m['f1']:>7.3f}"
        )

    lines.append("\n--- PER ATTACK TYPE (Full Pipeline) ---\n")
    header2 = f"{'Attack Type':<20} {'N':>4} {'Acc':>6} {'Prec':>7} {'Rec':>7} {'F1':>7}"
    lines.append(header2)
    lines.append("-" * len(header2))
    for at in unique_attacks:
        m = attack_type_results[at]
        count = sum(1 for a in attack_types if a == at)
        lines.append(
            f"{at:<20} {count:>4} {m['accuracy']:>6.3f} {m['precision']:>7.3f} "
            f"{m['recall']:>7.3f} {m['f1']:>7.3f}"
        )

    lines.append("\n" + "=" * 65)

    report = "\n".join(lines)
    print("\n" + report)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nResults saved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
