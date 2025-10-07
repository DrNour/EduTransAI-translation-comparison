# app.py (improved + streamlined metrics, thresholds, labels, metadata)
# ---------------------------------------------------------------------------------
# EduTransAI - Translation Comparison & Student Assessment
# Adds the requested features:
# 1) Streamline metrics – one representative semantic score (Hybrid Accuracy) + separate Fluency.
#    (Already present as `*_Accuracy` and `*_Fluency`; clarified in labels/exports.)
# 2) Recalibrate thresholds – quantile-based cutoffs with sensible defaults
#    (BLEU/CHRF < 0.15, Fluency < 3.0, Hybrid < 20th pct).
# 3) Normalize error labels – canonical label vocabulary for aggregation/ML.
# 4) Augment metadata – language/domain/genre + per-domain diagnostics.

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

    st.markdown("---")
    st.subheader("🧩 Metadata (optional)")
    st.caption("Add metadata for per-domain diagnostics and export.")

    meta_mode = st.radio("Metadata source", ["None", "Use existing columns", "Set constants"], index=0)

    language_val = domain_val = genre_val = None
    language_col = domain_col = genre_col = None

    if meta_mode == "Use existing columns":
        # Detect likely columns
        cols = []
        try:
            # Will be filled after upload – we guard with try/except and handle later.
            pass
        except Exception:
            pass
        st.caption("You'll be able to select columns after uploading.")
    elif meta_mode == "Set constants":
        language_val = st.text_input("Language (constant for all rows)", value="")
        domain_val = st.text_input("Domain (constant for all rows)", value="")
        genre_val = st.text_input("Genre (constant for all rows)", value="")

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

    if prefer_chrf and _CHRF is not None:
        lexical = chrf_score(ref_n, hyp_n)
        used_metric = "CHRF"
    else:
        lexical = float(
            sentence_bleu([simple_tokenize(ref_n)], simple_tokenize(hyp_n),
                           smoothing_function=SmoothingFunction().method4)
        )
        used_metric = "BLEU"
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
    # retained for row-wise adaptivity; global low cutoffs are added later
    acc = 0.60
    lex = 0.50
    if ref_len_tokens >= 25:
        acc -= 0.05; lex -= 0.05
    elif ref_len_tokens <= 6:
        acc += 0.05; lex += 0.05
    return acc, lex

# -----------------------------
# Error label normalization
# -----------------------------
# Canonical vocabulary tailored for this app
LABEL_CANON = {
    "fluency": ["fluency/grammar", "grammar", "fluency issues"],
    "semantic": ["semantic deviation", "meaning error", "mismatch"],
    "lexical": ["low lexical overlap", "lexical low", "overlap low"],
    "length_short": ["too short", "short"],
    "length_long": ["too long / verbosity", "too long", "verbosity"],
}
# Build lookup
_LABEL_LOOKUP = {k: k for k in LABEL_CANON}
for k, aliases in LABEL_CANON.items():
    for a in aliases:
        _LABEL_LOOKUP[a.lower()] = k


def normalize_labels(raw_list):
    norm = []
    for raw in (raw_list or []):
        canon = _LABEL_LOOKUP.get(str(raw).strip().lower())
        norm.append(canon if canon else f"other:{raw}")
    # dedup keep order
    seen, out = set(), []
    for x in norm:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

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

        # Metadata column selection (if user chose existing columns)
        if meta_mode == "Use existing columns":
            language_col = st.selectbox("Language column (optional)", ["<none>"] + list(df.columns), index=0)
            domain_col = st.selectbox("Domain column (optional)", ["<none>"] + list(df.columns), index=0)
            genre_col = st.selectbox("Genre column (optional)", ["<none>"] + list(df.columns), index=0)
            language_col = None if language_col == "<none>" else language_col
            domain_col = None if domain_col == "<none>" else domain_col
            genre_col = None if genre_col == "<none>" else genre_col

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
            per_row_meta = []
            for idx, row in df.iterrows():
                row_result = {}

                # row metadata (constant or columns)
                lang_val = (language_val if meta_mode == "Set constants" else (row.get(language_col, "") if language_col else ""))
                dom_val = (domain_val if meta_mode == "Set constants" else (row.get(domain_col, "") if domain_col else ""))
                gen_val = (genre_val if meta_mode == "Set constants" else (row.get(genre_col, "") if genre_col else ""))
                per_row_meta.append({"Language": lang_val or "", "Domain": dom_val or "", "Genre": gen_val or ""})

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

                        # Adaptive thresholds (length-aware)
                        acc_thr, lex_thr = dynamic_thresholds(ref_lens[idx])
                        errs_raw = []
                        if flu < 3:
                            errs_raw.append("Fluency/Grammar")
                        if hybrid < acc_thr:
                            errs_raw.append("Semantic Deviation")
                        if lexical < lex_thr:
                            errs_raw.append("Low Lexical Overlap")
                        n_words = len(trans_text_norm.split())
                        if n_words < 3:
                            errs_raw.append("Too Short")
                        elif n_words > 60:
                            errs_raw.append("Too Long / Verbosity")

                        errs_norm = normalize_labels(errs_raw)

                        row_result.update({
                            f"{t_col}_LexicalMetric": used_metric,
                            f"{t_col}_Lexical": lexical,
                            f"{t_col}_Cosine": cosine_sim,
                            f"{t_col}_Accuracy": hybrid,
                            f"{t_col}_Fluency": flu,
                            f"{t_col}_Errors": ", ".join(sorted(set(errs_raw))) if errs_raw else "None",
                            f"{t_col}_ErrorsNorm": ",".join(errs_norm) if errs_norm else "",
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
                            errs_raw = []
                            if flu < 3:
                                errs_raw.append("Fluency/Grammar")
                            n_words = len(trans_text_norm.split())
                            if n_words < 3:
                                errs_raw.append("Too Short")
                            elif n_words > 60:
                                errs_raw.append("Too Long / Verbosity")
                            errs_norm = normalize_labels(errs_raw)

                            row_result.update({
                                f"{t_col}_vs_{other_col}_LexicalMetric": used_metric,
                                f"{t_col}_vs_{other_col}_Lexical": lexical,
                                f"{t_col}_vs_{other_col}_Cosine": cosine_sim,
                                f"{t_col}_vs_{other_col}_Accuracy": hybrid,
                                f"{t_col}_vs_{other_col}_Fluency": flu,
                                f"{t_col}_vs_{other_col}_Errors": ", ".join(sorted(set(errs_raw))) if errs_raw else "None",
                                f"{t_col}_vs_{other_col}_ErrorsNorm": ",".join(errs_norm) if errs_norm else "",
                            })

                    elif mode == "Standalone Student Assessment":
                        flu = fluency_score(trans_text)
                        errs_raw = [] if flu >= 3 else ["Fluency/Grammar"]
                        errs_norm = normalize_labels(errs_raw)
                        # No reference: cosine/lexical/hybrid are None
                        row_result.update({
                            f"{t_col}_Fluency": flu,
                            f"{t_col}_Cosine": None,
                            f"{t_col}_Lexical": None,
                            f"{t_col}_Accuracy": None,
                            f"{t_col}_Errors": "None" if flu >= 3 else "Fluency/Grammar",
                            f"{t_col}_ErrorsNorm": ",".join(errs_norm) if errs_norm else "",
                        })

                results.append(row_result)

            res_df = pd.DataFrame(results)
            meta_df = pd.DataFrame(per_row_meta) if per_row_meta else pd.DataFrame()

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
            # Quantile-based global thresholds (recalibration)
            # ---------------------------
            def _collect_cols(suffix):
                return [c for c in res_df.columns if c.endswith(suffix)]

            lex_vals = pd.concat([res_df[c] for c in _collect_cols("_Lexical")], axis=0) if _collect_cols("_Lexical") else pd.Series(dtype=float)
            flu_vals = pd.concat([res_df[c] for c in _collect_cols("_Fluency")], axis=0) if _collect_cols("_Fluency") else pd.Series(dtype=float)
            acc_vals = pd.concat([res_df[c] for c in _collect_cols("_Accuracy")], axis=0) if _collect_cols("_Accuracy") else pd.Series(dtype=float)

            def safe_q(s, q, default):
                s = pd.to_numeric(s, errors='coerce').dropna()
                if s.empty:
                    return default
                try:
                    return float(s.quantile(q))
                except Exception:
                    return default

            # Defaults as examples
            default_bleu_or_chrf_low = 0.15
            default_fluency_low = 3.0
            default_hybrid_low = 0.50

            q_lex = safe_q(lex_vals, 0.10, default_bleu_or_chrf_low)
            q_flu = safe_q(flu_vals, 0.10, default_fluency_low)
            q_acc = safe_q(acc_vals, 0.20, default_hybrid_low)

            # choose the stricter (max) between empirical and defaults to avoid too-lenient cutoffs
            LEX_LOW = max(default_bleu_or_chrf_low, q_lex)
            FLU_LOW = max(default_fluency_low, q_flu)
            ACC_LOW = q_acc  # 20th percentile is already data-driven; keep default only if NaN handled above

            st.markdown(f"**Calibrated Low Cutoffs:** Lexical < `{LEX_LOW:.2f}`, Fluency < `{FLU_LOW:.2f}`, Hybrid Accuracy < `{ACC_LOW:.2f}`")

            # ---------------------------
            # Low-Quality Flags – Global (quantiles) + Adaptive (length)
            # ---------------------------
            st.subheader("⚠️ Low-Quality Translation Flags")
            flagged_cols = []

            if mode == "Reference-based" and source_col:
                for i in range(len(res_df)):
                    # adaptive thresholds by sentence length
                    acc_thr_adapt, lex_thr_adapt = dynamic_thresholds(ref_lens[i])
                    for c in translation_cols:
                        acc_col = f"{c}_Accuracy"; lex_col = f"{c}_Lexical"; flu_col = f"{c}_Fluency"

                        if acc_col in res_df.columns:
                            # Two flags: global and adaptive
                            flag_c_g = acc_col.replace("_Accuracy", "_Low_Hybrid_Flag_Global")
                            flag_c_a = acc_col.replace("_Accuracy", "_Low_Hybrid_Flag_Adaptive")
                            val = res_df.loc[i, acc_col]
                            res_df.loc[i, flag_c_g] = "⚠️" if pd.notna(val) and val < ACC_LOW else ""
                            res_df.loc[i, flag_c_a] = "⚠️" if pd.notna(val) and val < acc_thr_adapt else ""
                            flagged_cols += [flag_c_g, flag_c_a]

                        if lex_col in res_df.columns:
                            flag_l_g = lex_col.replace("_Lexical", "_Low_Lexical_Flag_Global")
                            flag_l_a = lex_col.replace("_Lexical", "_Low_Lexical_Flag_Adaptive")
                            val = res_df.loc[i, lex_col]
                            res_df.loc[i, flag_l_g] = "⚠️" if pd.notna(val) and val < LEX_LOW else ""
                            res_df.loc[i, flag_l_a] = "⚠️" if pd.notna(val) and val < lex_thr_adapt else ""
                            flagged_cols += [flag_l_g, flag_l_a]

                        if flu_col in res_df.columns:
                            flag_f_g = flu_col.replace("_Fluency", "_Low_Fluency_Flag_Global")
                            val = res_df.loc[i, flu_col]
                            res_df.loc[i, flag_f_g] = "⚠️" if pd.notna(val) and val < FLU_LOW else ""
                            flagged_cols.append(flag_f_g)

            # Fluency flags (all modes) – global only when not reference-based
            if mode != "Reference-based":
                for col in res_df.columns:
                    if col.endswith("_Fluency"):
                        flag_col = col.replace("_Fluency", "_Low_Fluency_Flag_Global")
                        res_df[flag_col] = res_df[col].apply(lambda x: "⚠️" if pd.notna(x) and x < FLU_LOW else "")
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
            # Per-domain diagnostics (using metadata)
            # ---------------------------
            st.subheader("🗂️ Per-Domain Diagnostics")
            if meta_df.shape[0] == res_df.shape[0]:
                # Build long format per translation
                long_rows = []
                for i in range(len(res_df)):
                    meta_row = meta_df.iloc[i].to_dict() if not meta_df.empty else {}
                    for base in translation_cols:
                        rec = {
                            **meta_row,
                            "Student": base,
                            "Accuracy": res_df.loc[i, f"{base}_Accuracy"] if f"{base}_Accuracy" in res_df.columns else np.nan,
                            "Fluency": res_df.loc[i, f"{base}_Fluency"] if f"{base}_Fluency" in res_df.columns else np.nan,
                            "Lexical": res_df.loc[i, f"{base}_Lexical"] if f"{base}_Lexical" in res_df.columns else np.nan,
                            "ErrorsNorm": res_df.loc[i, f"{base}_ErrorsNorm"] if f"{base}_ErrorsNorm" in res_df.columns else "",
                        }
                        long_rows.append(rec)
                long_df = pd.DataFrame(long_rows)

                # Aggregate
                if not long_df.empty:
                    group_keys = [k for k in ["Domain", "Language", "Genre"] if k in long_df.columns]
                    if not group_keys:
                        st.caption("No metadata provided; skipping per-domain aggregates.")
                    else:
                        # label prevalence
                        def top_labels(sub):
                            labels = ",".join(sub["ErrorsNorm"].dropna().astype(str)).split(",")
                            labels = [x for x in labels if x]
                            if not labels:
                                return ""
                            s = pd.Series(labels).value_counts(normalize=True)
                            pairs = [f"{lab}:{share:.0%}" for lab, share in s.head(3).items()]
                            return ", ".join(pairs)

                        agg = long_df.groupby(group_keys).agg(
                            Count=("Accuracy", "count"),
                            Accuracy_Mean=("Accuracy", "mean"),
                            Accuracy_Std=("Accuracy", "std"),
                            Fluency_Mean=("Fluency", "mean"),
                            Fluency_Std=("Fluency", "std"),
                            Lexical_Mean=("Lexical", "mean"),
                            Top_Labels=("ErrorsNorm", top_labels),
                        ).reset_index()
                        st.dataframe(agg)
            else:
                st.caption("Metadata shape mismatch; per-domain diagnostics skipped.")

            # ---------------------------
            # Clean CSV Export (exclude HTML diffs)
            # ---------------------------
            st.subheader("📥 Export Cleaned Results")
            preferred_order = []
            for base in translation_cols:
                for metric in ["Accuracy", "Lexical", "Cosine", "Fluency", "Errors", "ErrorsNorm"]:
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
            # attach metadata to export
            if not meta_df.empty:
                export_df = pd.concat([meta_df, export_df], axis=1)

            # Human-readable column names
            export_df.columns = (
                export_df.columns
                .str.replace("_", " ")
                .str.replace(" Lexical", " (Lexical)")
                .str.replace(" Accuracy", " (Hybrid Accuracy)")
                .str.replace(" Cosine", " (Semantic Cosine)")
                .str.replace(" Fluency", " (Fluency)")
                .str.replace(" ErrorsNorm", " (Error Labels – Canonical)")
                .str.replace(" Errors", " (Error Categories – Raw)")
                .str.replace(" Flag", " ⚠️", regex=False)
            )
            export_df = export_df.applymap(lambda x: round(x, 3) if isinstance(x, (float, int)) else x)

            # Show a peek
            st.dataframe(export_df.head(20))
            csv = export_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "Download Full Analysis Results (Clean CSV)",
                csv,
                "translation_analysis_clean.csv",
                "text/csv",
            )

            # show calibrated thresholds summary
            with st.expander("Thresholds (Quantile-based) Details", expanded=False):
                st.write({
                    "Lexical_Low": LEX_LOW,
                    "Fluency_Low": FLU_LOW,
                    "Hybrid_Low": ACC_LOW,
                })

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload CSV, Excel, or Word file to begin analysis.")
