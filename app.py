# app.py
# ---------------------------------------------------------------------------------
# EduTransAI - Translation Comparison & Student Assessment (Light build)
# Features:
# - Semantic metrics: Cosine, BERTScore, or Cosine+BERTScore (no COMET)
# - Lexical metric: CHRF (preferred) or BLEU fallback
# - Hybrid Accuracy = weighted (semantic, lexical) × length penalty
# - Paraphrase vs Meaning Drift labeling (semantic-based)
# - Fluency + Style scoring; optional floors stabilize Hybrid
# - Quantile-based "Low" cutoffs + length-adaptive thresholds
# - Normalized error labels for ML; metadata-aware diagnostics
# - Two-gate triage (Semantic first, then Lexical/Fluency)
# - SQI (Semantic Quality Index) and LI (Literalness Index)
# - Heatmaps/boxplots; clean CSV export; disagreement examples & outlier monitoring

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

# Optional semantic metric: BERTScore
try:
    from bert_score import score as bertscore_score
    _HAS_BERTSCORE = True
except Exception:
    _HAS_BERTSCORE = False

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
        help="Choose the multilingual model for cross-lingual comparisons.",
    )
    semantic_weight = st.slider("Semantic weight (for Hybrid)", 0.0, 1.0, 0.65, 0.05)
    lexical_weight = 1.0 - semantic_weight

    sem_choice = st.selectbox(
        "Semantic metric",
        options=["Cosine", "BERTScore", "Cosine+BERTScore"],
        index=0,
        help="BERTScore offers fairer semantics than raw lexical metrics; falls back if unavailable.",
    )

    use_chrf = st.checkbox(
        "Use CHRF for lexical overlap (fallback to BLEU if unavailable)",
        value=True,
    )

    st.markdown("---")
    st.subheader("🧪 Stability & Thresholds")
    fluency_floor = st.number_input("Fluency floor to trust Hybrid", 1.0, 5.0, 3.0, 0.1)
    style_floor = st.number_input("Style floor to trust Hybrid", 1.0, 5.0, 3.0, 0.1)
    consistency_tolerance = st.slider("Consistency tolerance (|Cosine − BERTScore|)", 0.0, 1.0, 0.20, 0.01)

    st.markdown("---")
    st.subheader("🧭 Evaluation Strategy")
    rebalance_sem = st.checkbox(
        "Emphasize semantic + fluency in Hybrid (rebalance)",
        value=True,
        help="Hybrid ≈ 0.8*Semantic + 0.2*Lexical, stabilized by Fluency/Style floors.",
    )
    use_composite = st.checkbox(
        "Report composite indices (SQI & LI)",
        value=True,
        help="SQI = 0.7*Semantic + 0.3*(Fluency/5). LI = Lexical.",
    )
    gateA_threshold = st.slider("Gate A – Semantic threshold", 0.50, 0.99, 0.80, 0.01)
    gateB_lex_threshold = st.slider("Gate B – Lexical threshold", 0.00, 0.80, 0.20, 0.01)
    gateB_flu_threshold = st.slider("Gate B – Fluency threshold", 1.0, 5.0, 3.0, 0.1)

    st.markdown("---")
    st.subheader("🧩 Paraphrase vs Drift")
    paraphrase_sem_hi = st.slider("Paraphrase: semantic high ≥", 0.70, 0.99, 0.85, 0.01)
    drift_sem_lo = st.slider("Meaning drift: semantic low <", 0.40, 0.95, 0.70, 0.01)
    low_lexical_for_paraphrase = st.slider("Paraphrase lexical threshold <", 0.0, 0.6, 0.20, 0.01)

    st.markdown("---")
    st.subheader("🧩 Metadata (optional)")
    st.caption("Add metadata for per-domain diagnostics and export.")
    meta_mode = st.radio("Metadata source", ["None", "Use existing columns", "Set constants"], index=0)

    language_val = domain_val = genre_val = None
    language_col = domain_col = genre_col = None

    if meta_mode == "Use existing columns":
        st.caption("You'll be able to select columns after uploading.")
    elif meta_mode == "Set constants":
        language_val = st.text_input("Language (constant for all rows)", value="")
        domain_val = st.text_input("Domain (constant for all rows)", value="")
        genre_val = st.text_input("Genre (constant for all rows)", value="")

# ===========================
# Load model (cached)
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

def compute_style_features(t: str) -> dict:
    """Deterministic style features + 1–5 style score."""
    t = normalize_text(t)
    t = re.sub(r"\s+", " ", t.strip())
    sents = max(1, len(re.findall(r"[.!?]", t)))
    tokens = t.split()
    n_tok = max(1, len(tokens))

    avg_sent_len = n_tok / sents
    long_tok_ratio = sum(len(w) > 24 for w in tokens) / n_tok
    repeat_punct = 1 if re.search(r"([!?.,])\1{1,}", t) else 0
    double_space = 1 if re.search(r"\s{2,}", t) else 0
    trailing_space = 1 if re.search(r"\s$", t) else 0
    ttr = len(set(tokens)) / n_tok  # type–token ratio

    score = 5.0
    score -= min(2.0, abs(avg_sent_len - 18) / 18) * 1.2
    score -= long_tok_ratio * 2.0
    score -= (repeat_punct + double_space + trailing_space) * 0.5
    score -= max(0, 0.35 - ttr) * 2.0  # penalize very low lexical variety
    score = float(np.clip(score, 1.0, 5.0))

    return {
        "avg_sent_len": avg_sent_len,
        "long_tok_ratio": long_tok_ratio,
        "repeat_punct": repeat_punct,
        "double_space": double_space,
        "trailing_space": trailing_space,
        "ttr": round(ttr, 3),
        "style_score": round(score, 2),
    }

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

def _safe_clip01(x):
    try:
        return float(np.clip(x, 0.0, 1.0))
    except Exception:
        return 0.0

def compute_semantic_signals(ref: str, hyp: str, embed_cache: dict, sem_choice: str):
    """Compute available semantic signals: Cosine and/or BERTScore-F1 (0..1)."""
    ref_n, hyp_n = normalize_text(ref), normalize_text(hyp)
    # Cosine via cached embeddings
    v_ref, v_hyp = get_vec(ref_n, embed_cache), get_vec(hyp_n, embed_cache)
    cosine = float(np.dot(v_ref, v_hyp)) if (v_ref is not None and v_hyp is not None) else None

    bert_f1 = None
    if _HAS_BERTSCORE and sem_choice in ("BERTScore", "Cosine+BERTScore"):
        try:
            P, R, F1 = bertscore_score([hyp_n], [ref_n], verbose=False, rescale_with_baseline=False)
            bert_f1 = float(F1[0].item())
            bert_f1 = _safe_clip01(bert_f1)
        except Exception:
            bert_f1 = None

    return {"cosine": cosine, "bertscore_f1": bert_f1}

def semantic_accuracy_score(
    ref: str,
    hyp: str,
    embed_cache: dict,
    w_sem: float,
    w_lex: float,
    prefer_chrf: bool,
    sem_choice: str,
    fluency: float | None = None,
    style: float | None = None,
    fluency_floor: float = 3.0,
    style_floor: float = 3.0,
    consistency_tolerance: float = 0.20,
):
    """
    Compute lexical + selected semantic metric + hybrid with stabilizers.
    Returns: (semantic_signal, used_sem_name, lexical, hybrid, extras_dict)
    extras include: cosine, bertscore_f1, length_penalty, lexical_metric
    """
    ref_n, hyp_n = normalize_text(ref), normalize_text(hyp)

    # lexical
    if prefer_chrf and _CHRF is not None:
        lexical = chrf_score(ref_n, hyp_n)
        used_lex = "CHRF"
    else:
        lexical = float(
            sentence_bleu([simple_tokenize(ref_n)], simple_tokenize(hyp_n),
                          smoothing_function=SmoothingFunction().method4)
        )
        used_lex = "BLEU"

    # semantic signals
    sigs = compute_semantic_signals(ref_n, hyp_n, embed_cache, sem_choice)
    cosine = sigs.get("cosine")
    bert_f1 = sigs.get("bertscore_f1")

    # pick semantic signal
    if sem_choice == "BERTScore" and bert_f1 is not None:
        sem_val, used_sem = bert_f1, "BERTScore"
    elif sem_choice == "Cosine+BERTScore" and (cosine is not None or bert_f1 is not None):
        vals = [v for v in [cosine, bert_f1] if v is not None]
        sem_val, used_sem = float(np.mean(vals)), "Cosine+BERTScore"
    else:
        sem_val, used_sem = (cosine if cosine is not None else 0.0), "Cosine"

    # length penalty (unchanged)
    penalty = length_ratio_penalty(ref_n, hyp_n)

    # hybrid before stabilization
    hybrid = _safe_clip01(w_sem * sem_val + w_lex * lexical) * penalty

    # Stabilize by fluency/style floors: if below floor, dampen hybrid
    if fluency is not None and fluency < fluency_floor:
        factor = max(0.1, float(fluency) / float(fluency_floor))
        hybrid *= factor
    if style is not None and style < style_floor:
        factor = max(0.1, float(style) / float(style_floor))
        hybrid *= factor

    # Consistency stabilization between cosine and BERTScore when both exist
    if cosine is not None and bert_f1 is not None:
        gap = abs(cosine - bert_f1)
        if gap > consistency_tolerance:
            # shrink hybrid proportionally to the excess gap (cap 30% shrink)
            hybrid *= (1.0 - min(0.30, (gap - consistency_tolerance)))

    extras = {
        "cosine": None if cosine is None else round(cosine, 3),
        "bertscore_f1": None if bert_f1 is None else round(bert_f1, 3),
        "length_penalty": round(penalty, 3),
        "lexical_metric": used_lex,
    }
    return round(sem_val, 3), used_sem, round(lexical, 3), round(hybrid, 3), extras

def dynamic_thresholds(ref_len_tokens: int):
    # retained for row-wise adaptivity; global low cutoffs are added later
    acc = 0.60
    lex = 0.50
    if ref_len_tokens >= 25:
        acc -= 0.05; lex -= 0.05
    elif ref_len_tokens <= 6:
        acc += 0.05; lex += 0.05
    return acc, lex

# Paraphrase vs. meaning drift classifier
def classify_semantic_deviation(
    lexical: float,
    cosine: float | None,
    bert_f1: float | None,
    paraphrase_sem_hi: float = 0.85,
    low_lex: float = 0.20,
    drift_sem_lo: float = 0.70,
) -> str | None:
    """Return 'Paraphrase' when semantic strong but lexical low; 'Meaning Drift' when semantic weak.
    Prefers BERTScore if available, else cosine."""
    sem_signal = None
    if bert_f1 is not None:
        sem_signal = bert_f1
    elif cosine is not None:
        sem_signal = cosine

    if sem_signal is None:
        return None

    if sem_signal >= paraphrase_sem_hi and lexical < low_lex:
        return "Paraphrase"
    if sem_signal < drift_sem_lo:
        return "Meaning Drift"
    return None

# -----------------------------
# Error label normalization
# -----------------------------
LABEL_CANON = {
    "fluency": ["fluency/grammar", "grammar", "fluency issues"],
    "style": ["style", "formatting", "layout"],
    "semantic": ["semantic deviation", "meaning error", "mismatch"],
    "lexical": ["low lexical overlap", "lexical low", "overlap low"],
    "length_short": ["too short", "short"],
    "length_long": ["too long / verbosity", "too long", "verbosity"],
    "paraphrase": ["paraphrase", "lexically diverse paraphrase"],
    "meaning_drift": ["meaning drift", "semantic shift"],
}
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

# ---------------------------
# Human-readable column names (robust, order-safe) + de-dup
# ---------------------------
def _pretty_metric_suffix(suffix: str) -> str:
    # Exact matches first
    mapping_exact = {
        "Accuracy": " (Hybrid Accuracy)",
        "Lexical": " (Lexical)",
        "Semantic": " (Semantic Score)",
        "Cosine": " (Semantic Cosine)",
        "BERTScoreF1": " (BERTScore F1)",
        "Fluency": " (Fluency)",
        "Style": " (Style)",
        "SQI": " (Semantic Quality Index)",
        "LI": " (Literalness Index)",
        "Errors": " (Error Categories – Raw)",
        "ErrorsNorm": " (Error Labels – Canonical)",
        "SemanticMetric": " (Semantic Metric)",
        "LexicalMetric": " (Lexical Metric)",
        "GateA_SemanticOK": " (Gate A – Semantic OK)",
        "GateB_Flag": " (Gate B – Needs Review)",
    }
    if suffix in mapping_exact:
        return mapping_exact[suffix]

    # Low flags / patterns
    if suffix == "Low_Hybrid_Flag_Global":
        return " Low Hybrid ⚠️ Global"
    if suffix == "Low_Hybrid_Flag_Adaptive":
        return " Low Hybrid ⚠️ Adaptive"
    if suffix == "Low_Lexical_Flag_Global":
        return " Low (Lexical) ⚠️ Global"
    if suffix == "Low_Lexical_Flag_Adaptive":
        return " Low (Lexical) ⚠️ Adaptive"
    if suffix == "Low_Fluency_Flag_Global":
        return " Low (Fluency) ⚠️ Global"
    if suffix == "Low_Style_Flag_Global":
        return " Low (Style) ⚠️ Global"

    # Fallback: show raw suffix with spaces
    return " " + suffix.replace("_", " ")

def humanize_export_columns(cols) -> list:
    """Map technical column names like 'Google Translate_Accuracy'
    to 'Google Translate (Hybrid Accuracy)' in a collision-free way."""
    out = []
    seen = set()
    for col in cols:
        # Keep metadata columns as-is
        if isinstance(col, str) and col in {"Language", "Domain", "Genre"}:
            new = col
        elif isinstance(col, str) and "_" in col:
            # Split only on the FIRST underscore to keep pairwise labels intact:
            # e.g., "A_vs_B_Accuracy" -> base="A_vs_B", suffix="Accuracy"
            base, suffix = col.split("_", 1)
            base_pretty = base.replace("_vs_", " vs ")
            new = base_pretty + _pretty_metric_suffix(suffix)
        else:
            new = col

        # Ensure uniqueness
        candidate = new
        k = 2
        while candidate in seen:
            candidate = f"{new} #{k}"
            k += 1
        seen.add(candidate)
        out.append(candidate)
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

                        # fluency/style first (stabilizers)
                        flu = fluency_score(trans_text)
                        style_feats = compute_style_features(trans_text)
                        style = float(style_feats["style_score"])

                        sem_sig, used_sem, lexical, hybrid, extras = semantic_accuracy_score(
                            source_text_norm,
                            trans_text_norm,
                            embed_cache,
                            semantic_weight,
                            lexical_weight,
                            prefer_chrf=use_chrf,
                            sem_choice=sem_choice,
                            fluency=flu,
                            style=style,
                            fluency_floor=fluency_floor,
                            style_floor=style_floor,
                            consistency_tolerance=consistency_tolerance,
                        )

                        # Adaptive thresholds (length-aware)
                        acc_thr, lex_thr = dynamic_thresholds(ref_lens[idx])
                        errs_raw = []
                        if flu < 3:
                            errs_raw.append("Fluency/Grammar")
                        if style < 3:
                            errs_raw.append("Style")

                        # semantic deviation labelling (paraphrase vs meaning drift)
                        sem_tag = classify_semantic_deviation(
                            lexical=lexical,
                            cosine=extras.get("cosine"),
                            bert_f1=extras.get("bertscore_f1"),
                            paraphrase_sem_hi=paraphrase_sem_hi,
                            low_lex=low_lexical_for_paraphrase,
                            drift_sem_lo=drift_sem_lo,
                        )
                        if sem_tag == "Paraphrase":
                            errs_raw.append("Paraphrase")
                        elif sem_tag == "Meaning Drift":
                            errs_raw.append("Semantic Deviation")

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

                        # Optional rebalanced hybrid
                        if rebalance_sem:
                            lp = extras.get("length_penalty", 1.0) or 1.0
                            hyb_reb = float(np.clip(0.8 * (sem_sig or 0.0) + 0.2 * (lexical or 0.0), 0, 1)) * lp
                            if flu < fluency_floor:
                                hyb_reb *= max(0.1, float(flu) / float(fluency_floor))
                            if style < style_floor:
                                hyb_reb *= max(0.1, float(style) / float(style_floor))
                            hybrid = round(hyb_reb, 3)

                        # Composite indices
                        SQI = round(float(np.clip(0.7 * (sem_sig or 0.0) + 0.3 * (flu / 5.0), 0, 1)), 3) if use_composite else None
                        LI = round(float(np.clip(lexical or 0.0, 0, 1)), 3) if use_composite else None

                        # Triage (Gate A/B)
                        gateA_pass = bool((sem_sig or 0.0) >= gateA_threshold)
                        gateB_flag = False
                        if not gateA_pass:
                            gateB_flag = bool(((lexical or 0.0) < gateB_lex_threshold) or (flu < gateB_flu_threshold))

                        row_result.update({
                            f"{t_col}_LexicalMetric": extras.get("lexical_metric"),
                            f"{t_col}_Lexical": lexical,
                            f"{t_col}_SemanticMetric": used_sem,
                            f"{t_col}_Semantic": sem_sig,
                            f"{t_col}_Cosine": extras.get("cosine"),
                            f"{t_col}_BERTScoreF1": extras.get("bertscore_f1"),
                            f"{t_col}_Accuracy": hybrid,
                            f"{t_col}_Fluency": flu,
                            f"{t_col}_Style": style,
                            f"{t_col}_SQI": SQI,
                            f"{t_col}_LI": LI,
                            f"{t_col}_GateA_SemanticOK": "✅" if gateA_pass else "",
                            f"{t_col}_GateB_Flag": "⚠️" if gateB_flag else "",
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

                            flu = fluency_score(trans_text)
                            style_feats = compute_style_features(trans_text)
                            style = float(style_feats["style_score"])

                            sem_sig, used_sem, lexical, hybrid, extras = semantic_accuracy_score(
                                other_text_norm,
                                trans_text_norm,
                                embed_cache,
                                semantic_weight,
                                lexical_weight,
                                prefer_chrf=use_chrf,
                                sem_choice=sem_choice,
                                fluency=flu,
                                style=style,
                                fluency_floor=fluency_floor,
                                style_floor=style_floor,
                                consistency_tolerance=consistency_tolerance,
                            )

                            errs_raw = []
                            if flu < 3:
                                errs_raw.append("Fluency/Grammar")
                            if style < 3:
                                errs_raw.append("Style")
                            n_words = len(trans_text_norm.split())
                            if n_words < 3:
                                errs_raw.append("Too Short")
                            elif n_words > 60:
                                errs_raw.append("Too Long / Verbosity")
                            errs_norm = normalize_labels(errs_raw)

                            # Optional rebalanced hybrid
                            if rebalance_sem:
                                lp = extras.get("length_penalty", 1.0) or 1.0
                                hyb_reb = float(np.clip(0.8 * (sem_sig or 0.0) + 0.2 * (lexical or 0.0), 0, 1)) * lp
                                if flu < fluency_floor:
                                    hyb_reb *= max(0.1, float(flu) / float(fluency_floor))
                                if style < style_floor:
                                    hyb_reb *= max(0.1, float(style) / float(style_floor))
                                hybrid = round(hyb_reb, 3)

                            SQI = round(float(np.clip(0.7 * (sem_sig or 0.0) + 0.3 * (flu / 5.0), 0, 1)), 3) if use_composite else None
                            LI = round(float(np.clip(lexical or 0.0, 0, 1)), 3) if use_composite else None
                            gateA_pass = bool((sem_sig or 0.0) >= gateA_threshold)
                            gateB_flag = False
                            if not gateA_pass:
                                gateB_flag = bool(((lexical or 0.0) < gateB_lex_threshold) or (flu < gateB_flu_threshold))

                            row_result.update({
                                f"{t_col}_vs_{other_col}_LexicalMetric": extras.get("lexical_metric"),
                                f"{t_col}_vs_{other_col}_Lexical": lexical,
                                f"{t_col}_vs_{other_col}_SemanticMetric": used_sem,
                                f"{t_col}_vs_{other_col}_Semantic": sem_sig,
                                f"{t_col}_vs_{other_col}_Cosine": extras.get("cosine"),
                                f"{t_col}_vs_{other_col}_BERTScoreF1": extras.get("bertscore_f1"),
                                f"{t_col}_vs_{other_col}_Accuracy": hybrid,
                                f"{t_col}_vs_{other_col}_Fluency": flu,
                                f"{t_col}_vs_{other_col}_Style": style,
                                f"{t_col}_vs_{other_col}_SQI": SQI,
                                f"{t_col}_vs_{other_col}_LI": LI,
                                f"{t_col}_vs_{other_col}_GateA_SemanticOK": "✅" if gateA_pass else "",
                                f"{t_col}_vs_{other_col}_GateB_Flag": "⚠️" if gateB_flag else "",
                                f"{t_col}_vs_{other_col}_Errors": ", ".join(sorted(set(errs_raw))) if errs_raw else "None",
                                f"{t_col}_vs_{other_col}_ErrorsNorm": ",".join(errs_norm) if errs_norm else "",
                            })

                    elif mode == "Standalone Student Assessment":
                        flu = fluency_score(trans_text)
                        style_feats = compute_style_features(trans_text)
                        style = float(style_feats["style_score"])
                        errs_raw = []
                        if flu < 3:
                            errs_raw.append("Fluency/Grammar")
                        if style < 3:
                            errs_raw.append("Style")
                        errs_norm = normalize_labels(errs_raw)
                        # No reference: cosine/lexical/hybrid/semantic are None
                        row_result.update({
                            f"{t_col}_Fluency": flu,
                            f"{t_col}_Style": style,
                            f"{t_col}_Cosine": None,
                            f"{t_col}_Lexical": None,
                            f"{t_col}_Semantic": None,
                            f"{t_col}_Accuracy": None,
                            f"{t_col}_Errors": "None" if not errs_raw else ", ".join(sorted(set(errs_raw))),
                            f"{t_col}_ErrorsNorm": ",".join(errs_norm) if errs_norm else "",
                        })

                results.append(row_result)

            res_df = pd.DataFrame(results)

            # Build metadata DataFrame
            meta_rows = []
            if meta_mode != "None":
                for i in range(len(res_df)):
                    meta_rows.append({
                        "Language": per_row_meta[i]["Language"] if meta_mode != "None" else "",
                        "Domain": per_row_meta[i]["Domain"] if meta_mode != "None" else "",
                        "Genre": per_row_meta[i]["Genre"] if meta_mode != "None" else "",
                    })
            meta_df = pd.DataFrame(meta_rows) if meta_rows else pd.DataFrame()

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
            sty_vals = pd.concat([res_df[c] for c in _collect_cols("_Style")], axis=0) if _collect_cols("_Style") else pd.Series(dtype=float)
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
            default_lex_low = 0.15
            default_fluency_low = 3.0
            default_style_low = 3.0
            default_hybrid_low = 0.50

            q_lex = safe_q(lex_vals, 0.10, default_lex_low)
            q_flu = safe_q(flu_vals, 0.10, default_fluency_low)
            q_sty = safe_q(sty_vals, 0.10, default_style_low)
            q_acc = safe_q(acc_vals, 0.20, default_hybrid_low)

            # choose stricter (max) for lexical/fluency/style; keep acc as empirical
            LEX_LOW = max(default_lex_low, q_lex)
            FLU_LOW = max(default_fluency_low, q_flu)
            STYLE_LOW = max(default_style_low, q_sty)
            ACC_LOW = q_acc

            st.markdown(
                f"**Calibrated Low Cutoffs:** Lexical < `{LEX_LOW:.2f}`, Fluency < `{FLU_LOW:.2f}`, "
                f"Style < `{STYLE_LOW:.2f}`, Hybrid Accuracy < `{ACC_LOW:.2f}`"
            )

            # ---------------------------
            # Low-Quality Flags – Global (quantiles) + Adaptive (length)
            # ---------------------------
            st.subheader("⚠️ Low-Quality Translation Flags")
            flagged_cols = []

            if mode == "Reference-based" and source_col:
                for i in range(len(res_df)):
                    acc_thr_adapt, lex_thr_adapt = dynamic_thresholds(ref_lens[i])
                    for c in translation_cols:
                        acc_col = f"{c}_Accuracy"; lex_col = f"{c}_Lexical"; flu_col = f"{c}_Fluency"; sty_col = f"{c}_Style"

                        if acc_col in res_df.columns:
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

                        if sty_col in res_df.columns:
                            flag_s_g = sty_col.replace("_Style", "_Low_Style_Flag_Global")
                            val = res_df.loc[i, sty_col]
                            res_df.loc[i, flag_s_g] = "⚠️" if pd.notna(val) and val < STYLE_LOW else ""
                            flagged_cols.append(flag_s_g)

            # Fluency/Style flags (all modes) – global only when not reference-based
            if mode != "Reference-based":
                for col in res_df.columns:
                    if col.endswith("_Fluency"):
                        flag_col = col.replace("_Fluency", "_Low_Fluency_Flag_Global")
                        res_df[flag_col] = res_df[col].apply(lambda x: "⚠️" if pd.notna(x) and x < FLU_LOW else "")
                        flagged_cols.append(flag_col)
                    if col.endswith("_Style"):
                        flag_col = col.replace("_Style", "_Low_Style_Flag_Global")
                        res_df[flag_col] = res_df[col].apply(lambda x: "⚠️" if pd.notna(x) and x < STYLE_LOW else "")
                        flagged_cols.append(flag_col)

            if flagged_cols:
                st.dataframe(res_df[sorted(set(flagged_cols))].head(20))

            # ---------------------------
            # Triage Summary & Disagreements
            # ---------------------------
            st.subheader("🧷 Triage Summary")
            if mode == "Reference-based":
                tri_rows = []
                for base in translation_cols:
                    ga_col = f"{base}_GateA_SemanticOK"
                    gb_col = f"{base}_GateB_Flag"
                    par_col = f"{base}_ErrorsNorm"
                    ga_ok_pct = float((res_df[ga_col] == "✅").mean() * 100) if ga_col in res_df.columns else np.nan
                    gb_flag_pct = float((res_df[gb_col] == "⚠️").mean() * 100) if gb_col in res_df.columns else np.nan
                    paraphrase_cases = int(res_df[par_col].astype(str).str.contains("paraphrase").sum()) if par_col in res_df.columns else 0
                    drift_cases = int(res_df[par_col].astype(str).str.contains("meaning_drift|semantic").sum()) if par_col in res_df.columns else 0
                    tri_rows.append({
                        "Student": base,
                        "GateA_OK_%": ga_ok_pct,
                        "GateB_Flag_%": gb_flag_pct,
                        "Paraphrase_cases": paraphrase_cases,
                        "MeaningDrift_cases": drift_cases,
                    })
                st.dataframe(pd.DataFrame(tri_rows))

            st.subheader("🧩 Disagreements & Teaching Cases")
            disagree_examples = []
            max_examples = 12
            for i in range(len(res_df)):
                for base in translation_cols:
                    if f"{base}_Semantic" not in res_df.columns:
                        continue
                    sem_val = res_df.loc[i, f"{base}_Semantic"]
                    lex_val = res_df.loc[i, f"{base}_Lexical"] if f"{base}_Lexical" in res_df.columns else np.nan
                    cos_val = res_df.loc[i, f"{base}_Cosine"] if f"{base}_Cosine" in res_df.columns else np.nan
                    bs_val = res_df.loc[i, f"{base}_BERTScoreF1"] if f"{base}_BERTScoreF1" in res_df.columns else np.nan

                    cond_paraphrase_like = (pd.notna(sem_val) and sem_val >= paraphrase_sem_hi) and (pd.notna(lex_val) and lex_val < low_lexical_for_paraphrase)
                    cond_metric_disagree = (pd.notna(cos_val) and pd.notna(bs_val) and abs(cos_val - bs_val) > consistency_tolerance)
                    if cond_paraphrase_like or cond_metric_disagree:
                        src_txt = str(df.loc[i, source_col]) if (mode == "Reference-based" and source_col) else ""
                        hyp_txt = str(df.loc[i, base]) if base in df.columns else ""
                        disagree_examples.append({
                            "Row": i,
                            "Student": base,
                            "Semantic": round(sem_val, 3) if pd.notna(sem_val) else np.nan,
                            "Lexical": round(lex_val, 3) if pd.notna(lex_val) else np.nan,
                            "Cosine": round(cos_val, 3) if pd.notna(cos_val) else np.nan,
                            "BERTScoreF1": round(bs_val, 3) if pd.notna(bs_val) else np.nan,
                            "Source": src_txt[:220],
                            "Translation": hyp_txt[:220],
                        })
                    if len(disagree_examples) >= max_examples:
                        break
                if len(disagree_examples) >= max_examples:
                    break
            if disagree_examples:
                st.caption("Examples to guide reviewers — avoid over-correcting valid paraphrases.")
                st.dataframe(pd.DataFrame(disagree_examples))

            # ---------------------------
            # Continuous Monitoring – Outlier shares
            # ---------------------------
            st.subheader("📉 Outlier Monitoring")
            def iqr_outlier_share(series):
                s = pd.to_numeric(series, errors='coerce').dropna()
                if s.empty:
                    return np.nan
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
                return float(((s < low) | (s > high)).mean() * 100)

            lex_all = pd.concat([res_df[c] for c in res_df.columns if c.endswith('_Lexical')], axis=0) if any(res_df.columns.str.endswith('_Lexical')) else pd.Series(dtype=float)
            acc_all = pd.concat([res_df[c] for c in res_df.columns if c.endswith('_Accuracy')], axis=0) if any(res_df.columns.str.endswith('_Accuracy')) else pd.Series(dtype=float)

            lex_out_pct = iqr_outlier_share(lex_all)
            acc_out_pct = iqr_outlier_share(acc_all)

            st.write({"Lexical_outliers_%": lex_out_pct, "Hybrid_outliers_%": acc_out_pct})

            if 'outlier_history' not in st.session_state:
                st.session_state.outlier_history = []
            st.session_state.outlier_history.append({"Lexical": lex_out_pct, "Hybrid": acc_out_pct, "n": len(res_df)})
            hist_df = pd.DataFrame(st.session_state.outlier_history)
            if not hist_df.empty:
                st.line_chart(hist_df[["Lexical", "Hybrid"]])

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
                    "Semantic": [c for c in res_df.columns if c.endswith("_Semantic")],
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
                metrics = ["Lexical", "Semantic", "Cosine", "Accuracy", "Fluency", "Style", "SQI", "LI"]
            elif mode == "Pairwise Comparison":
                metrics = ["Lexical", "Semantic", "Cosine", "Accuracy", "Fluency", "Style", "SQI", "LI"]
            else:
                metrics = ["Fluency", "Style"]

            # Ensure composite metrics included when present
            if "SQI" not in metrics:
                metrics = metrics + ["SQI", "LI"]

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
            if not meta_df.empty and meta_df.shape[0] == res_df.shape[0]:
                long_rows = []
                for i in range(len(res_df)):
                    meta_row = meta_df.iloc[i].to_dict()
                    for base in translation_cols:
                        rec = {
                            **meta_row,
                            "Student": base,
                            "Accuracy": res_df.get(f"{base}_Accuracy", pd.Series([np.nan]*len(res_df))).iloc[i],
                            "Fluency": res_df.get(f"{base}_Fluency", pd.Series([np.nan]*len(res_df))).iloc[i],
                            "Style": res_df.get(f"{base}_Style", pd.Series([np.nan]*len(res_df))).iloc[i],
                            "Lexical": res_df.get(f"{base}_Lexical", pd.Series([np.nan]*len(res_df))).iloc[i],
                            "Semantic": res_df.get(f"{base}_Semantic", pd.Series([np.nan]*len(res_df))).iloc[i],
                            "ErrorsNorm": res_df.get(f"{base}_ErrorsNorm", pd.Series([""]*len(res_df))).iloc[i],
                        }
                        long_rows.append(rec)
                long_df = pd.DataFrame(long_rows)

                if not long_df.empty:
                    group_keys = [k for k in ["Domain", "Language", "Genre"] if k in long_df.columns]
                    if not group_keys:
                        st.caption("No metadata provided; skipping per-domain aggregates.")
                    else:
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
                            Style_Mean=("Style", "mean"),
                            Lexical_Mean=("Lexical", "mean"),
                            Semantic_Mean=("Semantic", "mean"),
                            Top_Labels=("ErrorsNorm", top_labels),
                        ).reset_index()
                        st.dataframe(agg)
            else:
                st.caption("No metadata or shape mismatch; per-domain diagnostics skipped.")

            # ---------------------------
            # Clean CSV Export (exclude HTML diffs)
            # ---------------------------
            st.subheader("📥 Export Cleaned Results")
            preferred_order = []
            for base in translation_cols:
                for metric in [
                    "Accuracy", "Lexical", "Semantic", "Cosine", "BERTScoreF1",
                    "Fluency", "Style", "SQI", "LI", "GateA_SemanticOK", "GateB_Flag",
                    "Errors", "ErrorsNorm", "SemanticMetric", "LexicalMetric"
                ]:
                    matches = [c for c in res_df.columns if c.startswith(base + "_") and c.endswith(metric)]
                    preferred_order.extend(matches)

            # Only add flag columns that are NOT already in preferred_order (prevents duplicates)
            flag_cols = [c for c in res_df.columns if ("Flag" in c) and (c not in preferred_order)]

            if "Best_Translation" in res_df.columns:
                preferred_order = ["Best_Translation"] + preferred_order

            preferred_order.extend(flag_cols)

            other_cols = [c for c in res_df.columns if c not in preferred_order]
            ordered_cols = preferred_order + other_cols

            export_df = res_df[ordered_cols].copy()
            if not meta_df.empty:
                export_df = pd.concat([meta_df, export_df], axis=1)

            # Human-readable, collision-safe column names
            export_df.columns = humanize_export_columns(export_df.columns)

            # Round numeric values
            export_df = export_df.applymap(lambda x: round(x, 3) if isinstance(x, (float, int)) else x)

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
                    "Style_Low": STYLE_LOW,
                    "Hybrid_Low": ACC_LOW,
                })

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload CSV, Excel, or Word file to begin analysis.")
