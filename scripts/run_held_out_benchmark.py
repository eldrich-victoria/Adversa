"""
run_held_out_benchmark.py — Held-Out Robustness Benchmark

Evaluates the pipeline on examples that are NOT in the knowledge base,
providing a more honest assessment of out-of-distribution robustness.

Includes:
1. Semantic Leakage Analysis
2. Improved Retrieval Evaluation (Label, Category, Attack Family)
3. RAG Ablation Study

Usage:
    python scripts/run_held_out_benchmark.py
"""

import sys
from pathlib import Path
import json

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

from src.detector import detect_attacks
from src.normalizer import normalize
from src.classifier import classify, _keyword_score
from src.retriever import retrieve, _load_index
from src.embeddings import encode

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HELD_OUT_PATH = ROOT / "data" / "held_out_benchmark.csv"
RESULTS_PATH  = ROOT / "data" / "held_out_benchmark_results.txt"
LEAKAGE_CSV_PATH = ROOT / "data" / "semantic_leakage.csv"
KB_PATH = ROOT / "data" / "adversarial_examples.json"


# ---------------------------------------------------------------------------
# Prediction functions (Ablation)
# ---------------------------------------------------------------------------

def predict_baseline(text: str) -> str:
    """System A: raw text keyword signal only."""
    score = _keyword_score(text.lower())
    return "toxic" if score >= 0.10 else "benign"


def predict_normalized(text: str) -> str:
    """System B: normalized text keyword signal only."""
    norm = normalize(text)
    score = _keyword_score(norm["normalized"])
    return "toxic" if score >= 0.10 else "benign"


def predict_detect_norm_only(text: str) -> str:
    """System C: detect + norm + classify (NO retrieval)."""
    attack_info = detect_attacks(text)
    norm = normalize(text)
    clf = classify(norm["normalized"], [], attack_info)
    return clf["label"].replace("non-toxic", "benign")


def predict_full(text: str) -> str:
    """System D: full pipeline — detect + normalize + retrieve + classify."""
    attack_info = detect_attacks(text)
    norm = normalize(text)
    retrieved = retrieve(norm["normalized"], k=5)
    clf = classify(norm["normalized"], retrieved, attack_info)
    return clf["label"].replace("non-toxic", "benign")


# ---------------------------------------------------------------------------
# Semantic Leakage Analysis
# ---------------------------------------------------------------------------

def analyze_semantic_leakage(df: pd.DataFrame) -> dict:
    """
    Computes max similarity of each held-out query against the KB.
    Saves a detailed CSV and returns summary statistics.
    """
    kb_records, kb_embeddings = _load_index()
    if kb_embeddings is None:
        raise RuntimeError("KB embeddings not found. Run scripts/build_index.py first.")

    records = []
    max_sims = []

    for idx, row in df.iterrows():
        text = row["text"]
        q_emb = encode(text).reshape(1, -1)
        sims = cosine_similarity(q_emb, kb_embeddings)[0]
        max_idx = np.argmax(sims)
        max_sim = sims[max_idx]
        nearest_kb = kb_records[max_idx]

        records.append({
            "query_id": row["id"],
            "query_text": text,
            "query_label": row["label"],
            "query_category": row.get("category", "N/A"),
            "query_attack_type": row["attack_type"],
            "max_similarity": float(max_sim),
            "nearest_kb_id": nearest_kb.get("id", max_idx),
            "nearest_kb_text": nearest_kb["text"],
            "nearest_kb_label": nearest_kb["label"],
            "nearest_kb_category": nearest_kb["category"],
            "nearest_kb_attack": nearest_kb.get("attack_types", ["clean"])[0] if nearest_kb.get("attack_types") else "clean"
        })
        max_sims.append(max_sim)

    # Save detailed records to CSV
    leakage_df = pd.DataFrame(records)
    leakage_df.to_csv(LEAKAGE_CSV_PATH, index=False)

    max_sims = np.array(max_sims)
    stats = {
        "mean": float(np.mean(max_sims)),
        "median": float(np.median(max_sims)),
        "min": float(np.min(max_sims)),
        "max": float(np.max(max_sims)),
        "std": float(np.std(max_sims)),
        "pct_gt_80": float(np.mean(max_sims > 0.80) * 100),
        "pct_gt_85": float(np.mean(max_sims > 0.85) * 100),
        "pct_gt_90": float(np.mean(max_sims > 0.90) * 100),
        "pct_gt_95": float(np.mean(max_sims > 0.95) * 100),
    }

    return stats, leakage_df


# ---------------------------------------------------------------------------
# Improved Retrieval Evaluation
# ---------------------------------------------------------------------------

def evaluate_retrieval(df: pd.DataFrame) -> dict:
    """
    Evaluate retrieval quality on held-out examples.
    """
    results = {
        "top1_label": 0, "top5_label": 0,
        "top1_category": 0, "top5_category": 0,
        "top1_attack": 0, "top5_attack": 0,
        "total": len(df)
    }

    for _, row in df.iterrows():
        text = row["text"]
        true_label = row["label"]
        true_category = row.get("category", "insult" if true_label=="toxic" else "benign")
        true_attack = row["attack_type"]
        
        norm = normalize(text)
        retrieved = retrieve(norm["normalized"], k=5)

        if not retrieved:
            continue

        # Top-1 Checks
        top1 = retrieved[0]
        if top1["label"] == true_label:
            results["top1_label"] += 1
        if top1.get("category", "") == true_category:
            results["top1_category"] += 1
        top1_attacks = top1.get("attack_types", ["clean"])
        if not top1_attacks:
            top1_attacks = ["clean"]
        if true_attack in top1_attacks:
            results["top1_attack"] += 1

        # Top-5 Checks
        if any(r["label"] == true_label for r in retrieved):
            results["top5_label"] += 1
        if any(r.get("category", "") == true_category for r in retrieved):
            results["top5_category"] += 1
            
        found_attack = False
        for r in retrieved:
            r_attacks = r.get("attack_types", ["clean"])
            if not r_attacks:
                r_attacks = ["clean"]
            if true_attack in r_attacks:
                found_attack = True
                break
        if found_attack:
            results["top5_attack"] += 1

    return results


# ---------------------------------------------------------------------------
# Metric helper
# ---------------------------------------------------------------------------

def evaluate_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    yt = [1 if y == "toxic" else 0 for y in y_true]
    yp = [1 if y == "toxic" else 0 for y in y_pred]
    return {
        "accuracy":  accuracy_score(yt, yp),
        "precision": precision_score(yt, yp, zero_division=0),
        "recall":    recall_score(yt, yp, zero_division=0),
        "f1":        f1_score(yt, yp, zero_division=0),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Loading held-out benchmark data...")
    df = pd.read_csv(HELD_OUT_PATH)
    df = df.reset_index(drop=True)
    print(f"  {len(df)} held-out examples")

    true_labels = df["label"].tolist()

    # ---- 1. Semantic Leakage Analysis ----
    print("\nRunning Semantic Leakage Analysis...")
    leakage_stats, leakage_df = analyze_semantic_leakage(df)
    
    # ---- 2. Retrieval Evaluation ----
    print("Evaluating Retrieval Metrics...")
    retrieval = evaluate_retrieval(df)

    # ---- 3. Ablation Predictions ----
    print("\nRunning Ablation Study...")
    print("  System A: Baseline...")
    preds_a = [predict_baseline(t) for t in df["text"]]

    print("  System B: Norm + baseline...")
    preds_b = [predict_normalized(t) for t in df["text"]]

    print("  System C: Detect + norm (no retrieval)...")
    preds_c = [predict_detect_norm_only(t) for t in df["text"]]

    print("  System D: Full pipeline...")
    preds_d = [predict_full(t) for t in df["text"]]

    systems = {
        "A. Baseline (keyword, raw)":                 evaluate_metrics(true_labels, preds_a),
        "B. Normalization + keyword":                 evaluate_metrics(true_labels, preds_b),
        "C. Detector + normalization (no retrieval)": evaluate_metrics(true_labels, preds_c),
        "D. Full pipeline (detect+norm+retrieve)":    evaluate_metrics(true_labels, preds_d),
    }

    # ---- Build report ----
    lines: list[str] = []
    lines.append("=" * 68)
    lines.append("  ADVERSARIAL TOXIC LANGUAGE AUDITOR")
    lines.append("  FINAL HELD-OUT ROBUSTNESS BENCHMARK")
    lines.append("=" * 68)
    lines.append(f"\nDataset: {len(df)} held-out examples")

    lines.append("\n--- 1. SEMANTIC LEAKAGE ANALYSIS ---\n")
    lines.append("Max Cosine Similarity vs Knowledge Base:")
    lines.append(f"  Mean:   {leakage_stats['mean']:.4f}")
    lines.append(f"  Median: {leakage_stats['median']:.4f}")
    lines.append(f"  Min:    {leakage_stats['min']:.4f}")
    lines.append(f"  Max:    {leakage_stats['max']:.4f}")
    lines.append(f"  StdDev: {leakage_stats['std']:.4f}")
    lines.append(f"  > 0.80: {leakage_stats['pct_gt_80']:.1f}%")
    lines.append(f"  > 0.85: {leakage_stats['pct_gt_85']:.1f}%")
    lines.append(f"  > 0.90: {leakage_stats['pct_gt_90']:.1f}%")
    lines.append(f"  > 0.95: {leakage_stats['pct_gt_95']:.1f}%")
    lines.append(f"\nDetailed pairs saved to: {LEAKAGE_CSV_PATH.name}")

    lines.append("\nTop 3 Highest-Similarity Pairs:")
    top_3 = leakage_df.sort_values("max_similarity", ascending=False).head(3)
    for _, r in top_3.iterrows():
        lines.append(f"  Sim: {r['max_similarity']:.4f}")
        lines.append(f"  Qry: {r['query_text']}")
        lines.append(f"  KB:  {r['nearest_kb_text']}\n")

    lines.append("--- 2. RETRIEVAL METRICS (top-k) ---\n")
    total = retrieval['total']
    lines.append(f"Top-1 Label Agreement:         {retrieval['top1_label']}/{total} ({retrieval['top1_label']/total:.3f})")
    lines.append(f"Top-5 Label Agreement:         {retrieval['top5_label']}/{total} ({retrieval['top5_label']/total:.3f})")
    lines.append(f"Top-1 Category Agreement:      {retrieval['top1_category']}/{total} ({retrieval['top1_category']/total:.3f})")
    lines.append(f"Top-5 Category Agreement:      {retrieval['top5_category']}/{total} ({retrieval['top5_category']/total:.3f})")
    lines.append(f"Top-1 Attack-Family Agreement: {retrieval['top1_attack']}/{total} ({retrieval['top1_attack']/total:.3f})")
    lines.append(f"Top-5 Attack-Family Agreement: {retrieval['top5_attack']}/{total} ({retrieval['top5_attack']/total:.3f})")

    lines.append("\n--- 3. ABLATION EXPERIMENT ---\n")
    hdr = f"{'System':<44} {'Acc':>6} {'Prec':>7} {'Rec':>7} {'F1':>7}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for name, m in systems.items():
        lines.append(
            f"{name:<44} {m['accuracy']:>6.3f} {m['precision']:>7.3f} "
            f"{m['recall']:>7.3f} {m['f1']:>7.3f}"
        )

    lines.append("\n" + "=" * 68)
    report = "\n".join(lines)
    print("\n" + report)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nResults saved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
