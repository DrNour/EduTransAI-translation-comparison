# app.py (improved)
# ---------------------------------
# EduTransAI - Translation Comparison & Student Assessment
# This version adds: robust normalization, stronger hybrid metric (cosine+CHRF/BLEU with length penalty),
# safer/faster embedding cache, clearer token-level diffs, adaptive thresholds, and lighter UI rendering.

import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
from difflib import SequenceMatcher
from hashlib import blake2b
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from docx import Document

# Optional lexical metric: sacrebleu CHRF (preferred)
try:
    from sacrebleu.metrics import CHRF
    _CHRF = CHRF(word_order=0)
except Exception:
    _CHRF = None

# ===========================
# Streamlit page config
# ===========================
st.set_page_config(page_title="EduTransAI - Translation Assessment", layout="wide")
st.title("📊 EduTransAI - Translation Comparison & Student Assessment")

# ===========================
# Sidebar controls
# ===========================
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox(
        "Embedding model",
        options=[
            "all-MiniLM-L6-v2",
            "paraphrase-multilingual-MiniLM-L12-v2",  # better cross-lingual
        ],
        index=0,
        help="Choose a multilingual model if your data spans languages.",
    )
    semantic_weight = st.slider("Semantic weight (cosine)", 0.0, 1.0, 0.65, 0.05)
    lexical_weight = 1.0 - semantic_weight
    use_chrf = st.checkbox(
        "Use CHRF for lexical overlap (fallback to BLEU if unavailable)",
        value=True,
    )

# ===========================
# Load model (cached per model name)
# ===========================
@st.cache_resource(show_spinner=True)
def load_model(name: str):
    return SentenceTransformer(name)

model = load_model(model_name)

# ===========================
# Helper functions
# ===========================
word_or_punct = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+|[^\w\s]", re.UNICODE)


def normalize_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", str(t)).strip().lower()
    t = re.sub(r"[“”]", '"', t)
    t = re.sub(r"[‘’]", "'", t)
    t = re.sub(r"\s+", " ", t)
    return t


def simple_tokenize(text: str):
    return word_or_punct.findall(text)


def text_key(t: str) -> str:
    return blake2b(t.encode("utf-8"), digest_size=12).hexdigest()


@st.cache_data(show_spinner=False)
def batch_encode_unique(texts: list[str], model_name_for_cache: str):
    # cache also depends on model name
    vecs = load_model(model_name_for_cache).encode(
        texts,
        batch_size=128,
        show_progress_bar=False,
        normalize_embeddings=True,  # unit vectors -> dot is cosine
    )
    return {text_key(t): v for t, v in zip(texts, vecs)}


def get_vec(t: str, cache: dict):
    return cache.get(text_key(t))


def chrf_score(ref: str, hyp: str) -> float:
    if _CHRF is None:
        raise NameError("CHRF not available")
    return _CHRF.sentence_score(hyp, [ref]).score / 100.0  # 0..1


def length_ratio_penalty(ref: str, hyp: str) -> float:
    r = max(1e-6, len(hyp.split())) / max(1e-6, len(ref.split()))
    return float(np.exp(-abs(np.log(r))))  # 1.0 when equal; ~0.61 at 2x or 0.5x


def semantic_accuracy_score(ref: str, hyp: str, embed_cache: dict,
                            w_sem: float, w_lex: float, prefer_chrf: bool):
    ref_n, hyp_n = normalize_text(ref), normalize_text(hyp)
    v_ref, v_hyp = get_vec(ref_n, embed_cache), get_vec(hyp_n, embed_cache)
    cosine = float(np.dot(v_ref, v_hyp)) if (v_ref is not None and v_hyp is not None) else 0.0

    lexical = 0.0
    used_metric = "BLEU"
    if prefer_chrf and _CHRF is not None:
        lexical = chrf_score(ref_n, hyp_n)
        used_metric = "CHRF"
    else:
        lexical = float(
            sentence_bleu([simple_tokenize(ref_n)], simple_tokenize(hyp_n),
                           smoothing_function=SmoothingFunction().method4)
        )
    penalty = length_ratio_penalty(ref_n, hyp_n)
    hybrid = np.clip(w_sem * cosine + w_lex * lexical, 0, 1) * penalty
    return round(cosine, 3), used_metric, round(lexical, 3), round(hybrid, 3)


def fluency_score(text: str) -> float:
    t = normalize_text(text)
    if not t:
        return 1.0
    tokens = t.split()
    long_token_ratio = sum(len(w) > 24 for w in tokens) / max(1, len(tokens))
    punct_endings = len(re.findall(r"[.!?]", t))
    sentences = max(1, punct_endings)
    avg_sent_len = len(tokens) / sentences
    repeats = len(re.findall(r"([!?.,])\1{1,}", t))
    weird_ws = 1 if re.search(r"\s{2,}", t) else 0

    score = 5.0
    score -= min(2.0, abs(avg_sent_len - 18) / 18) * 1.0
    score -= long_token_ratio * 2.0
    score -= repeats * 0.3
    score -= weird_ws * 0.5
    return round(float(np.clip(score, 1.0, 5.0)), 2)


def token_diff(a: str, b: str) -> str:
    a_t, b_t = a.split(), b.split()
    sm = SequenceMatcher(None, a_t, b_t)
    parts = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            parts.extend(b_t[j1:j2])
        elif tag == "replace":
            if i1 != i2:
                parts.append(f"<span style='background:#ffe6e6;text-decoration:line-through'>{' '.join(a_t[i1:i2])}</span>")
            if j1 != j2:
                parts.append(f"<span style='background:#e6ffe6;'>{' '.join(b_t[j1:j2])}</span>")
        elif tag == "delete":
            parts.append(f"<span style='background:#ffe6e6;text-decoration:line-through'>{' '.join(a_t[i1:i2])}</span>")
        elif tag == "insert":
            parts.append(f"<span style='background:#e6ffe6;'>{' '.join(b_t[j1:j2])}</span>")
    return " ".join(parts)


def dynamic_thresholds(ref_len_tokens: int):
    acc = 0.60
    lex = 0.50
    if ref_len_tokens >= 25:
        acc -= 0.05; lex -= 0.05
    elif ref_len_tokens <= 6:
        acc += 0.05; lex += 0.05
    return acc, lex


# ===========================
# File upload
# ===========================
uploaded_file = st.file_uploader(
    "Upload CSV, Excel, or Word file (translations)",
    type=["csv", "xlsx", "xls", "docx"]
)

if uploaded_file:
    try:
        # ---------------------------
        # Read file content
        # ---------------------------
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding="utf-8", na_filter=False)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file, na_filter=False)
        elif uploaded_file.name.endswith(".docx"):
            doc = Document(uploaded_file)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            df = pd.DataFrame({"Text": paragraphs})
        else:
            st.error("Unsupported file type.")
            st.stop()

        df.columns = df.columns.str.strip().fillna("Unnamed")
        df = df.fillna("")

        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())

        # Keep a normalized copy for scoring; keep originals for display/export
        df_norm = df.applymap(normalize_text)

        # ---------------------------
        # Assessment mode
        # ---------------------------
        mode = st.radio(
            "Select assessment mode:",
            ["Reference-based", "Pairwise Comparison", "Standalone Student Assessment"],
        )
        source_col = None
        translation_cols = None
        if mode == "Reference-based" and len(df.columns) > 1:
            source_col = st.selectbox("Reference / Source Column", df.columns)
            translation_cols = st.multiselect(
                "Translations / Student Submissions",
                [c for c in df.columns if c != source_col],
            )
            if not translation_cols:
                st.warning("Select at least one translation column.")
        else:
            translation_cols = st.multiselect("Translations / Student Submissions", df.columns)
            if not translation_cols:
                st.warning("Select at least one translation column.")

        if translation_cols and st.button("Run Analysis"):
            st.subheader("✅ Analysis Results")
            results = []

            # ---------------------------
            # Precompute embeddings in batch (on normalized texts)
            # ---------------------------
            all_texts = []
            if source_col:
                all_texts.extend(df_norm[source_col].astype(str).tolist())
            for c in translation_cols:
                all_texts.extend(df_norm[c].astype(str).tolist())
            all_texts = list({t for t in all_texts if t.strip()})
            embed_cache = batch_encode_unique(all_texts, model_name)

            # Reference lengths for adaptive thresholds
            ref_lens = None
            if mode == "Reference-based" and source_col:
                ref_lens = df_norm[source_col].apply(lambda t: len(str(t).split())).tolist()

            # ---------------------------
            # Analyze each row
            # ---------------------------
            for idx, row in df.iterrows():
                row_result = {}
                for t_col in translation_cols:
                    trans_text = str(row[t_col])
                    trans_text_norm = str(df_norm.iloc[idx][t_col])

                    if source_col and mode == "Reference-based":
                        source_text = str(row[source_col])
                        source_text_norm = str(df_norm.iloc[idx][source_col])
                        cosine_sim, used_metric, lexical, hybrid = semantic_accuracy_score(
                            source_text_norm,
                            trans_text_norm,
                            embed_cache,
                            semantic_weight,
                            lexical_weight,
                            prefer_chrf=use_chrf,
                        )
                        flu = fluency_score(trans_text)

                        # Adaptive thresholds
                        acc_thr, lex_thr = dynamic_thresholds(ref_lens[idx])
                        err = []
                        if flu < 3:
                            err.append("Fluency/Grammar")
                        if hybrid < acc_thr:
                            err.append("Semantic Deviation")
                        if lexical < lex_thr:
                            err.append("Low Lexical Overlap")
                        n_words = len(trans_text_norm.split())
                        if n_words < 3:
                            err.append("Too Short")
                        elif n_words > 60:
                            err.append("Too Long / Verbosity")

                        row_result.update({
                            f"{t_col}_LexicalMetric": used_metric,
                            f"{t_col}_Lexical": lexical,
                            f"{t_col}_Cosine": cosine_sim,
                            f"{t_col}_Accuracy": hybrid,
                            f"{t_col}_Fluency": flu,
                            f"{t_col}_Errors": ", ".join(sorted(set(err))) if err else "None",
                        })

                        # Inline diff (HTML, not exported)
                        with st.expander(f"Diff: {t_col} vs {source_col} (row {idx})", expanded=False):
                            st.markdown(token_diff(source_text, trans_text), unsafe_allow_html=True)

                    elif mode == "Pairwise Comparison":
                        for other_col in translation_cols:
                            if other_col == t_col:
                                continue
                            other_text = str(row[other_col])
                            other_text_norm = str(df_norm.iloc[idx][other_col])
                            cosine_sim, used_metric, lexical, hybrid = semantic_accuracy_score(
                                other_text_norm,
                                trans_text_norm,
                                embed_cache,
                                semantic_weight,
                                lexical_weight,
                                prefer_chrf=use_chrf,
                            )
                            flu = fluency_score(trans_text)
                            err = []
                            if flu < 3:
                                err.append("Fluency/Grammar")
                            n_words = len(trans_text_norm.split())
                            if n_words < 3:
                                err.append("Too Short")
                            elif n_words > 60:
                                err.append("Too Long / Verbosity")

                            row_result.update({
                                f"{t_col}_vs_{other_col}_LexicalMetric": used_metric,
                                f"{t_col}_vs_{other_col}_Lexical": lexical,
                                f"{t_col}_vs_{other_col}_Cosine": cosine_sim,
                                f"{t_col}_vs_{other_col}_Accuracy": hybrid,
                                f"{t_col}_vs_{other_col}_Fluency": flu,
                                f"{t_col}_vs_{other_col}_Errors": ", ".join(sorted(set(err))) if err else "None",
                            })

                    elif mode == "Standalone Student Assessment":
                        flu = fluency_score(trans_text)
                        # No reference: cosine/lexical/hybrid are None
                        row_result.update({
                            f"{t_col}_Fluency": flu,
                            f"{t_col}_Cosine": None,
                            f"{t_col}_Lexical": None,
                            f"{t_col}_Accuracy": None,
                            f"{t_col}_Errors": "None" if flu >= 3 else "Fluency/Grammar",
                        })

                results.append(row_result)

            res_df = pd.DataFrame(results)

            # ---------------------------
            # Identify Best Translation per sentence (Reference-based)
            # ---------------------------
            if mode == "Reference-based":
                acc_cols = [c for c in res_df.columns if c.endswith("_Accuracy")]
                if acc_cols:
                    res_df["Best_Translation"] = res_df[acc_cols].idxmax(axis=1)
                    res_df["Best_Translation"] = res_df["Best_Translation"].str.replace("_Accuracy", "", regex=False)

            st.dataframe(res_df.head(20))

            # ---------------------------
            # Low-Quality Flags (using adaptive thresholds when applicable)
            # ---------------------------
            st.subheader("⚠️ Low-Quality Translation Flags")
            flagged_cols = []
            if mode == "Reference-based" and source_col:
                for i in range(len(res_df)):
                    acc_thr, lex_thr = dynamic_thresholds(ref_lens[i])
                    for c in translation_cols:
                        acc_col = f"{c}_Accuracy"
                        lex_col = f"{c}_Lexical"
                        if acc_col in res_df.columns:
                            flag_c = acc_col.replace("_Accuracy", "_Low_Accuracy_Flag")
                            res_df.loc[i, flag_c] = "⚠️" if pd.notna(res_df.loc[i, acc_col]) and res_df.loc[i, acc_col] < acc_thr else ""
                            flagged_cols.append(flag_c)
                        if lex_col in res_df.columns:
                            flag_l = lex_col.replace("_Lexical", "_Low_Lexical_Flag")
                            res_df.loc[i, flag_l] = "⚠️" if pd.notna(res_df.loc[i, lex_col]) and res_df.loc[i, lex_col] < lex_thr else ""
                            flagged_cols.append(flag_l)
            # Fluency flags (all modes)
            for col in res_df.columns:
                if col.endswith("_Fluency"):
                    flag_col = col.replace("_Fluency", "_Low_Fluency_Flag")
                    res_df[flag_col] = res_df[col].apply(lambda x: "⚠️" if pd.notna(x) and x < 3 else "")
                    flagged_cols.append(flag_col)

            if flagged_cols:
                st.dataframe(res_df[sorted(set(flagged_cols))].head(20))

            # ---------------------------
            # Heatmaps (styled DataFrames for speed)
            # ---------------------------
            if mode == "Reference-based":
                st.subheader("📈 Per-Sentence Similarity Heatmaps")
                def style_heatmap(df_num):
                    try:
                        return df_num.style.background_gradient(cmap="YlGnBu").format("{:.2f}")
                    except Exception:
                        return df_num

                metric_map = {
                    "Lexical": [c for c in res_df.columns if c.endswith("_Lexical")],
                    "Cosine": [c for c in res_df.columns if c.endswith("_Cosine")],
                    "Accuracy": [c for c in res_df.columns if c.endswith("_Accuracy")],
                }
                for metric, cols in metric_map.items():
                    if cols:
                        st.markdown(f"**{metric} per Sentence**")
                        st.dataframe(style_heatmap(res_df[cols]))

            # ---------------------------
            # Dashboard Metrics (compact)
            # ---------------------------
            st.subheader("📊 Dashboard Summary")
            if mode == "Reference-based":
                metrics = ["Lexical", "Cosine", "Accuracy", "Fluency"]
            elif mode == "Pairwise Comparison":
                metrics = ["Lexical", "Cosine", "Accuracy"]
            else:
                metrics = ["Fluency"]

            for metric in metrics:
                metric_cols = [c for c in res_df.columns if c.endswith(metric)]
                if metric_cols:
                    plt.figure(figsize=(10, 4))
                    sns.boxplot(data=res_df[metric_cols])
                    plt.ylabel(metric)
                    plt.title(f"{metric} Distribution Across Students / Translations")
                    st.pyplot(plt)

            # ---------------------------
            # Clean CSV Export (exclude HTML diffs)
            # ---------------------------
            st.subheader("📥 Export Cleaned Results")
            preferred_order = []
            for base in translation_cols:
                for metric in ["Accuracy", "Lexical", "Cosine", "Fluency", "Errors"]:
                    matches = [c for c in res_df.columns if c.startswith(base) and c.endswith(metric)]
                    preferred_order.extend(matches)
                # include metric name column if present
                metric_name_col = f"{base}_LexicalMetric"
                if metric_name_col in res_df.columns:
                    preferred_order.append(metric_name_col)
            flag_cols = [c for c in res_df.columns if "Flag" in c]
            if "Best_Translation" in res_df.columns:
                preferred_order = ["Best_Translation"] + preferred_order
            preferred_order.extend(flag_cols)
            other_cols = [c for c in res_df.columns if c not in preferred_order]
            ordered_cols = preferred_order + other_cols
            export_df = res_df[ordered_cols].copy()

            # Human-readable column names
            export_df.columns = (
                export_df.columns
                .str.replace("_", " ")
                .str.replace(" Lexical", " (Lexical)")
                .str.replace(" Accuracy", " (Hybrid Accuracy)")
                .str.replace(" Cosine", " (Semantic Cosine)")
                .str.replace(" Fluency", " (Fluency)")
                .str.replace(" Errors", " (Error Categories)")
                .str.replace(" Flag", " ⚠️", regex=False)
            )
            export_df = export_df.applymap(lambda x: round(x, 3) if isinstance(x, (float, int)) else x)

            st.dataframe(export_df.head(20))
            csv = export_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "Download Full Analysis Results (Clean CSV)",
                csv,
                "translation_analysis_clean.csv",
                "text/csv",
            )

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload CSV, Excel, or Word file to begin analysis.")
