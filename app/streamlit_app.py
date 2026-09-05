"""
streamlit_app.py — Adversarial Toxic Language Auditor

A single-page Streamlit UI that demonstrates the full adversarial
toxicity detection pipeline.
"""

import sys
from pathlib import Path

# Ensure the project root is importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from src.pipeline import run_pipeline

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Adversarial Toxic Language Auditor",
    page_icon="🔍",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1rem;
        color: #94a3b8;
        margin-top: 0.25rem;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 0.25rem;
    }
    .verdict-toxic {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        color: #fca5a5;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: 700;
        display: inline-block;
    }
    .verdict-nontoxic {
        background: linear-gradient(135deg, #14532d, #166534);
        color: #86efac;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: 700;
        display: inline-block;
    }
    .attack-badge {
        background: #1e293b;
        border: 1px solid #334155;
        color: #f59e0b;
        border-radius: 999px;
        padding: 0.2rem 0.75rem;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.15rem;
    }
    .retrieved-card {
        background: #1e293b;
        border-left: 3px solid #3b82f6;
        border-radius: 6px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.5rem;
        font-size: 0.875rem;
    }
    .retrieved-card.toxic {
        border-left-color: #ef4444;
    }
    .retrieved-card.benign {
        border-left-color: #22c55e;
    }
    .sim-score {
        color: #94a3b8;
        font-size: 0.75rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown('<p class="main-title">🔍 Adversarial Toxic Language Auditor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Detecting obfuscated toxic language using normalization and semantic retrieval.</p>',
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

EXAMPLES = [
    "you are an idiot",
    "y0u'r3 an 1d10t 💀",
    "i d i o t",
    "stuuuuupid",
    "уоu аrе аn іdіоt",
    "Great work on the project!",
]

def on_example_select():
    chosen = st.session_state.get("example_select")
    if chosen and chosen != "— select —":
        st.session_state["input_text"] = chosen


col_input, col_example = st.columns([3, 1])

with col_input:
    user_input = st.text_area(
        "Enter text to analyze:",
        height=100,
        placeholder='e.g.  "y0u\'r3 an 1d10t 💀"',
        key="input_text",
    )

with col_example:
    st.markdown("**Try an example:**")
    st.selectbox(
        "Examples",
        options=["— select —"] + EXAMPLES,
        label_visibility="collapsed",
        key="example_select",
        on_change=on_example_select,
    )

analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

if analyze_btn and user_input.strip():
    with st.spinner("Running pipeline…"):
        try:
            result = run_pipeline(user_input.strip(), k=5)
        except FileNotFoundError as e:
            st.error(f"⚠️ Index not built yet.\n\n```\n{e}\n```\n\nRun: `python scripts/build_index.py`")
            st.stop()

    attack_info = result["attack_info"]
    norm = result["normalization"]
    clf = result["classification"]
    retrieved = result["retrieved"]

    st.divider()

    # ---- 1. Original Input ----
    st.markdown('<p class="section-header">1 · Original Input</p>', unsafe_allow_html=True)
    st.code(result["original"], language=None)

    # ---- 2. Attack Detection ----
    st.markdown('<p class="section-header">2 · Adversarial Attack Detected</p>', unsafe_allow_html=True)
    if attack_info["attack_detected"]:
        st.warning("⚠️ Adversarial manipulation detected.")
    else:
        st.success("✅ No adversarial attacks detected.")

    # ---- 3. Attack Types ----
    if attack_info["attack_types"]:
        st.markdown('<p class="section-header">3 · Detected Attack Types</p>', unsafe_allow_html=True)
        badges_html = "".join(
            f'<span class="attack-badge">{t}</span>'
            for t in attack_info["attack_types"]
        )
        st.markdown(badges_html, unsafe_allow_html=True)

    # ---- 4. Normalized Text ----
    st.markdown('<p class="section-header">4 · Normalized Candidate</p>', unsafe_allow_html=True)
    col_n1, col_n2 = st.columns(2)
    with col_n1:
        st.text_input("Normalized", value=norm["normalized"], disabled=True, key="norm_out")
    with col_n2:
        if norm["transformations"]:
            st.markdown("**Transformations applied:**")
            for t in norm["transformations"]:
                st.markdown(f"- `{t}`")
        else:
            st.markdown("*No transformations applied.*")

    st.divider()

    # ---- 5. Verdict ----
    st.markdown('<p class="section-header">5 · Toxicity Verdict</p>', unsafe_allow_html=True)
    if clf["label"] == "toxic":
        st.markdown('<span class="verdict-toxic">🚨 TOXIC</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="verdict-nontoxic">✅ NON-TOXIC</span>', unsafe_allow_html=True)

    # ---- 6. Confidence ----
    st.markdown('<p class="section-header">6 · Confidence</p>', unsafe_allow_html=True)
    st.progress(clf["confidence"], text=f"{clf['confidence']:.1%}")

    # ---- 7. Category ----
    st.markdown('<p class="section-header">7 · Toxicity Category</p>', unsafe_allow_html=True)
    st.markdown(f"`{clf['category']}`")

    # ---- 8. Explanation ----
    st.markdown('<p class="section-header">8 · Explanation</p>', unsafe_allow_html=True)
    st.info(clf["explanation"])

    # ---- 9. Retrieved Examples ----
    st.markdown('<p class="section-header">9 · Top Retrieved Similar Examples</p>', unsafe_allow_html=True)
    for i, r in enumerate(retrieved, 1):
        card_class = "retrieved-card toxic" if r["label"] == "toxic" else "retrieved-card benign"
        attack_str = ", ".join(r["attack_types"]) if r["attack_types"] else "none"
        st.markdown(
            f"""
<div class="{card_class}">
  <strong>#{i} [{r['label'].upper()}]</strong> — {r['category']}<br>
  <em>{r['text']}</em><br>
  <span class="sim-score">Similarity: {r['similarity']:.3f} · Attacks: {attack_str}</span>
</div>
""",
            unsafe_allow_html=True,
        )

elif analyze_btn:
    st.warning("Please enter some text to analyze.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.markdown(
    """
<small style="color:#475569">
  ⚠️ This is a local research/portfolio MVP. It uses synthetic knowledge base examples
  and a lightweight retrieval-augmented approach. It is not a production-grade moderation system.
  Model: <code>all-MiniLM-L6-v2</code> · Retrieval: sklearn cosine similarity · No external APIs.
</small>
""",
    unsafe_allow_html=True,
)
