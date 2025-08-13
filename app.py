import os
import json
import time
import uuid
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import difflib
import re

import plotly.graph_objects as go

from metrics import (
    embed_texts,
    semantic_similarity,
    ai_judge_translation,
    ai_extract_terms,
    normalize_score,
)

# --------------------
# Page config
# --------------------
st.set_page_config(
    page_title="EduTransAI – Translation Comparison",
    page_icon="🧭",
    layout="wide",
)

st.title("EduTransAI – AI-Powered Translation Comparison")
st.caption("Compare translations for Accuracy, Fluency, Style, Terminology, Similarity — now with error highlighting and instructor rubric.")

# -------- Helpers for highlighting --------
def tokenize(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text.split(" ")

def html_diff(source: str, translation: str, key_terms: List[str]) -> str:
    """
    Produce an HTML side-by-side-style inline diff for translation against source tokens.
    - insertions: <span class='ins'>word</span>
    - deletions: <span class='del'>word</span>  (from source POV)
    - replacements: show del then ins
    - key terms present in translation are highlighted with <span class='term'>
    """
    s_tokens = tokenize(source)
    t_tokens = tokenize(translation)
    sm = difflib.SequenceMatcher(None, s_tokens, t_tokens)
    html = []
    key_set = {kt.lower() for kt in key_terms}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for w in t_tokens[j1:j2]:
                w_out = w
                if any(kt in w.lower() for kt in key_set):
                    w_out = f"<span class='term'>{w}</span>"
                html.append(w_out)
        elif tag == 'insert':
            for w in t_tokens[j1:j2]:
                w_out = w
                if any(kt in w.lower() for kt in key_set):
                    w_out = f"<span class='term ins'>{w}</span>"
                else:
                    w_out = f"<span class='ins'>{w}</span>"
                html.append(w_out)
        elif tag == 'delete':
            for w in s_tokens[i1:i2]:
                html.append(f"<span class='del'>{w}</span>")
        elif tag == 'replace':
            # show deleted source and inserted target
            for w in s_tokens[i1:i2]:
                html.append(f"<span class='del'>{w}</span>")
            for w in t_tokens[j1:j2]:
                w_out = w
                if any(kt in w.lower() for kt in key_set):
                    w_out = f"<span class='term rep'>{w}</span>"
                else:
                    w_out = f"<span class='rep'>{w}</span>"
                html.append(w_out)
    return " ".join(html)

HIGHLIGHT_CSS = """
<style>
.diffbox {font-family: ui-sans-serif,system-ui,-apple-system; line-height:1.9; padding:12px 14px; border:1px solid #e5e7eb; border-radius:12px; background:#fff;}
.del {text-decoration: line-through; opacity: 0.6;}
.ins {text-decoration: underline; font-weight: 600;}
.rep {background: rgba(255, 196, 0, 0.25); border-radius: 6px; padding: 0 2px;}
.term {background: rgba(34,197,94,0.18); border-radius:6px; padding:0 2px;}
.kbd {font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; background:#f3f4f6; padding:2px 6px; border-radius:6px; border:1px solid #e5e7eb;}
.small {color:#6b7280; font-size: 0.9rem;}
</style>
"""
st.markdown(HIGHLIGHT_CSS, unsafe_allow_html=True)

# Sidebar: settings + instructor rubric
with st.sidebar:
    st.header("Settings")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    st.write("**Model settings**")
    openai_model = st.text_input("OpenAI judge model", value=openai_model)
    embed_model = st.text_input("Embedding model", value=embed_model)
    temperature = st.slider("Judge temperature", 0.0, 1.0, 0.2, 0.1)
    max_terms = st.slider("Max key terms to extract", 3, 30, 10, 1)
    chunk_limit = st.slider("Max chars per chunk (for very long texts)", 500, 6000, 2200, 100)

    st.write("---")
    st.subheader("Instructor Rubric (weights)")
    w_acc = st.slider("Accuracy weight", 0, 100, 30, 5)
    w_flu = st.slider("Fluency weight", 0, 100, 20, 5)
    w_style = st.slider("Style weight", 0, 100, 20, 5)
    w_term = st.slider("Terminology weight", 0, 100, 15, 5)
    w_sim = st.slider("Similarity weight", 0, 100, 15, 5)
    w_sum = w_acc + w_flu + w_style + w_term + w_sim
    if w_sum != 100:
        st.warning(f"Current weights sum to {w_sum}. They will be normalized to 100.", icon="⚖️")

    st.write("---")
    st.subheader("Error Penalty")
    penalty_per_error = st.slider("Penalty per error (points)", 0, 10, 1, 1)
    max_penalty = st.slider("Max total penalty", 0, 40, 10, 1)

    st.write("---")
    st.write("**Upload CSV (optional)**")
    uploaded = st.file_uploader("CSV with columns: translation_id, translation_text, (optional) source", type=["csv"])
    st.markdown("[Download sample CSV](sample_data/sample_translations.csv)")

# Main input
col1, col2 = st.columns([1, 1])
with col1:
    source_text = st.text_area("Source text", height=160, key="source_text")

with col2:
    default_n = 2
    n = st.number_input("Number of translations to compare", min_value=2, max_value=10, value=default_n, step=1)
    translations_inputs = []
    for i in range(int(n)):
        t = st.text_area(f"Translation {i+1} text", height=120, key=f"t_{i}")
        label = st.text_input(f"Translation {i+1} label", value=f"T{i+1}", key=f"l_{i}")
        translations_inputs.append((label, t))

# If CSV uploaded, merge/override
if uploaded is not None:
    try:
        df_csv = pd.read_csv(uploaded)
        needed = {"translation_id", "translation_text"}
        if not needed.issubset(set(df_csv.columns)):
            st.error("CSV must include columns: translation_id, translation_text (optional: source).")
        else:
            if not source_text and "source" in df_csv.columns and df_csv["source"].notna().any():
                uniq_src = [s for s in df_csv["source"].dropna().unique() if isinstance(s, str) and s.strip()]
                if uniq_src:
                    source_text = uniq_src[0]
            translations_inputs = [(row["translation_id"], row["translation_text"]) for _, row in df_csv.iterrows()]
            st.success(f"Loaded {len(translations_inputs)} translations from CSV.")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

st.write("---")

def make_radar(scores_by_label: Dict[str, Dict[str, float]]):
    metrics = ["Accuracy", "Fluency", "Style", "Terminology", "Similarity"]
    fig = go.Figure()
    for label, sc in scores_by_label.items():
        values = [sc.get(m, 0) for m in metrics]
        values.append(values[0])  # close shape
        fig.add_trace(go.Scatterpolar(r=values, theta=metrics + [metrics[0]], fill='toself', name=label))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        margin=dict(l=10, r=10, t=10, b=10),
        height=460
    )
    return fig

def to_download(df: pd.DataFrame, per_tr_feedback: Dict[str, Any], rubric: Dict[str, Any]) -> Tuple[bytes, bytes]:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    bundle = {"rubric": rubric, "details": per_tr_feedback}
    json_bytes = json.dumps(bundle, ensure_ascii=False, indent=2).encode("utf-8")
    return csv_bytes, json_bytes

# Run evaluation
run = st.button("Run AI Evaluation", type="primary", use_container_width=True)

if run:
    if not source_text or not any(t.strip() for _, t in translations_inputs):
        st.error("Please provide the source text and at least two translations.")
        st.stop()

    # Filter blank translations
    translations = [(lbl, txt) for lbl, txt in translations_inputs if isinstance(txt, str) and txt.strip()]
    if len(translations) < 2:
        st.error("Please provide at least two non-empty translations.")
        st.stop()

    st.info("Extracting key terms & computing embeddings...", icon="🔍")
    # Extract key terms from source via AI
    try:
        key_terms = ai_extract_terms(source_text, max_terms=max_terms, model=openai_model, temperature=temperature)
    except Exception as e:
        st.warning(f"AI term extraction failed ({e}). Falling back to simple terms.", icon="⚠️")
        toks = [w.lower() for w in source_text.split() if w.isalpha()]
        freq = pd.Series(toks).value_counts().head(max_terms)
        key_terms = freq.index.tolist()

    # Embeddings for similarity
    texts_for_embed = [source_text] + [t for _, t in translations]
    embeds = embed_texts(texts_for_embed, model=embed_model)
    src_vec = embeds[0]
    tr_vecs = embeds[1:]

    # Normalize rubric weights
    w = np.array([w_acc, w_flu, w_style, w_term, w_sim], dtype=float)
    if w.sum() == 0:
        w = np.array([30,20,20,15,15], dtype=float)
    w = w / w.sum()

    rubric = {
        "weights": {"accuracy": float(w[0]), "fluency": float(w[1]), "style": float(w[2]), "terminology": float(w[3]), "similarity": float(w[4])},
        "penalty_per_error": penalty_per_error,
        "max_penalty": max_penalty,
        "descriptors": {
            "accuracy": "Faithful meaning transfer; nuances preserved; no omissions/additions.",
            "fluency": "Grammatical correctness; natural, readable prose; idiomaticity.",
            "style": "Appropriate register and voice; cohesion and coherence.",
            "terminology": "Key terms translated consistently and appropriately.",
            "similarity": "Semantic overlap with source via embeddings."
        }
    }

    scores_rows = []
    per_translation_feedback = {}

    progress = st.progress(0.0)
    for idx, ((label, tr_text), tr_vec) in enumerate(zip(translations, tr_vecs)):
        progress.progress((idx+1)/len(translations))
        try:
            judge = ai_judge_translation(
                source_text=source_text,
                translation_text=tr_text,
                key_terms=key_terms,
                model=openai_model,
                temperature=temperature,
                chunk_limit=int(chunk_limit),
            )
        except Exception as e:
            st.error(f"AI judge failed for {label}: {e}")
            continue

        # Similarity score (0–100)
        sim = semantic_similarity(src_vec, tr_vec)
        sim_score = normalize_score(sim)

        # Base scores (0–100)
        acc = judge["scores"]["accuracy"]
        flu = judge["scores"]["fluency"]
        sty = judge["scores"]["style"]
        term = judge["scores"]["terminology"]

        # Error penalty
        n_errors = len(judge.get("errors", []))
        total_penalty = min(max_penalty, n_errors * penalty_per_error)

        weighted = w[0]*acc + w[1]*flu + w[2]*sty + w[3]*term + w[4]*sim_score
        overall = max(0.0, weighted - total_penalty)

        scores_rows.append({
            "Label": label,
            "Accuracy": acc,
            "Fluency": flu,
            "Style": sty,
            "Terminology": term,
            "Similarity": sim_score,
            "Errors": n_errors,
            "Penalty": total_penalty,
            "Overall (weighted - penalty)": round(overall, 2),
        })
        judge["key_terms"] = key_terms
        per_translation_feedback[label] = judge

    if not scores_rows:
        st.stop()

    scores_df = pd.DataFrame(scores_rows).sort_values("Overall (weighted - penalty)", ascending=False)
    st.subheader("Results")
    st.dataframe(scores_df, use_container_width=True)

    # Radar chart
    radar_data = {row["Label"]: {k: row[k] for k in ["Accuracy", "Fluency", "Style", "Terminology", "Similarity"]} for _, row in scores_df.iterrows()}
    fig = make_radar(radar_data)
    st.plotly_chart(fig, use_container_width=True)

    # Detailed feedback + highlighting
    st.subheader("AI Feedback, Error Analysis, and Highlighting")
    for label in scores_df["Label"]:
        fb = per_translation_feedback[label]
        with st.expander(f"Details – {label}", expanded=False):
            st.markdown("**Key terms** detected in source: " + ", ".join(fb.get("key_terms", [])))
            st.markdown("**Term coverage**: " + fb["explanations"]["terminology_coverage"])
            st.markdown("**Accuracy notes**: " + fb["explanations"]["accuracy_notes"])
            st.markdown("**Fluency notes**: " + fb["explanations"]["fluency_notes"])
            st.markdown("**Style notes**: " + fb["explanations"]["style_notes"])

            err_df = pd.DataFrame(fb.get("errors", []))
            st.markdown("**Error categories**:")
            if not err_df.empty:
                st.dataframe(err_df, use_container_width=True)
            else:
                st.write("No major errors detected.")

            st.markdown("<div class='small'>Legend: <span class='del kbd'>deleted (from source)</span> <span class='ins kbd'>inserted</span> <span class='rep kbd'>replaced</span> <span class='term kbd'>key term</span></div>", unsafe_allow_html=True)
            html = html_diff(source_text, translations[[l for l,_ in translations].index(label)][1], fb.get("key_terms", []))
            st.markdown(f"<div class='diffbox'>{html}</div>", unsafe_allow_html=True)

    # Download buttons
    st.subheader("Export")
    csv_bytes, json_bytes = to_download(scores_df, per_translation_feedback, rubric)
    st.download_button("Download scores CSV", data=csv_bytes, file_name="scores.csv", mime="text/csv")
    st.download_button("Download detailed JSON (with rubric)", data=json_bytes, file_name="detailed_feedback.json", mime="application/json")

    st.success("Done.")
else:
    st.info("Enter/paste your texts or upload a CSV, then click **Run AI Evaluation**.", icon="💡")
