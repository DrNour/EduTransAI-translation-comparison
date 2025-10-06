# app.py
import streamlit as st
import pandas as pd
import numpy as np
import difflib
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from docx import Document

# ===========================
# Streamlit page config
# ===========================
st.set_page_config(page_title="EduTransAI - Translation Assessment", layout="wide")
st.title("📊 EduTransAI - Translation Comparison & Student Assessment")

# ===========================
# Load model
# ===========================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ===========================
# Helper functions
# ===========================
def simple_tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

def semantic_accuracy_score(text1, text2, embed_cache):
    """Return cosine, BLEU, and hybrid Accuracy"""
    vec1, vec2 = embed_cache[text1], embed_cache[text2]
    cosine_sim = cosine_similarity(vec1, vec2)
    bleu = sentence_bleu([simple_tokenize(text1)], simple_tokenize(text2),
                         smoothing_function=SmoothingFunction().method4)
    hybrid = round(0.7 * cosine_sim + 0.3 * bleu, 3)
    return round(cosine_sim, 3), round(bleu, 3), hybrid

def fluency_score(text):
    """Estimate fluency based on length balance, punctuation, and surface cleanliness"""
    if not text.strip():
        return 1.0
    words = text.split()
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    sentence_endings = len(re.findall(r'[.!?]', text))
    num_sentences = max(1, sentence_endings)
    avg_sentence_len = len(words) / num_sentences
    penalty = len(re.findall(r'\s{2,}', text)) + len(re.findall(r'[^\w\s.,;:!?\'"-]', text))
    score = 5 - (abs(avg_sentence_len - 20)/20 + penalty*0.3 + abs(avg_word_len - 5)*0.2)
    return round(max(1, min(score, 5)), 2)

def enhanced_error_detection(text, reference=None, embed_cache=None):
    """Detect fluency and semantic issues"""
    fluency = fluency_score(text)
    errors = []
    cosine_sim = bleu = hybrid = None

    if fluency < 3:
        errors.append("Fluency/Grammar")
    if reference and embed_cache:
        cosine_sim, bleu, hybrid = semantic_accuracy_score(reference, text, embed_cache)
        if hybrid < 0.6:
            errors.append("Semantic Deviation")
        elif hybrid < 0.8:
            errors.append("Partial Accuracy")
    if len(text.split()) < 3:
        errors.append("Too Short")
    elif len(text.split()) > 60:
        errors.append("Too Long / Verbosity")

    return fluency, cosine_sim, bleu, hybrid, ", ".join(sorted(set(errors))) if errors else "None"

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

        # ---------------------------
        # Assessment mode
        # ---------------------------
        mode = st.radio("Select assessment mode:", ["Reference-based", "Pairwise Comparison", "Standalone Student Assessment"])
        source_col = None
        translation_cols = None
        if mode == "Reference-based" and len(df.columns) > 1:
            source_col = st.selectbox("Reference / Source Column", df.columns)
            translation_cols = st.multiselect("Translations / Student Submissions", [c for c in df.columns if c != source_col])
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
            # Precompute embeddings in batch
            # ---------------------------
            all_texts = list(set(df[translation_cols].astype(str).values.ravel()))
            if source_col:
                all_texts += df[source_col].astype(str).tolist()
            all_texts = list(set(all_texts))
            embeddings = model.encode(all_texts, batch_size=64, show_progress_bar=True)
            embed_cache = {text: emb for text, emb in zip(all_texts, embeddings)}

            # ---------------------------
            # Analyze each row
            # ---------------------------
            for idx, row in df.iterrows():
                row_result = {}
                for t_col in translation_cols:
                    trans_text = str(row[t_col])
                    if source_col and mode == "Reference-based":
                        source_text = str(row[source_col])
                        fluency, cosine_sim, bleu, hybrid, error_str = enhanced_error_detection(
                            trans_text,
                            reference=source_text,
                            embed_cache=embed_cache
                        )
                        seq = difflib.ndiff(source_text.split(), trans_text.split())
                        diff = ' '.join([f"[{s}]" if s.startswith('-') or s.startswith('+') else s for s in seq])
                        row_result.update({
                            f"{t_col}_BLEU": bleu,
                            f"{t_col}_Cosine": cosine_sim,
                            f"{t_col}_Accuracy": hybrid,
                            f"{t_col}_Fluency": fluency,
                            f"{t_col}_Errors": error_str,
                            f"{t_col}_Diff": diff
                        })
                    elif mode == "Pairwise Comparison":
                        for other_col in translation_cols:
                            if other_col == t_col:
                                continue
                            other_text = str(row[other_col])
                            fluency, cosine_sim, bleu, hybrid, error_str = enhanced_error_detection(
                                trans_text,
                                reference=other_text,
                                embed_cache=embed_cache
                            )
                            seq = difflib.ndiff(other_text.split(), trans_text.split())
                            diff = ' '.join([f"[{s}]" if s.startswith('-') or s.startswith('+') else s for s in seq])
                            row_result.update({
                                f"{t_col}_vs_{other_col}_BLEU": bleu,
                                f"{t_col}_vs_{other_col}_Cosine": cosine_sim,
                                f"{t_col}_vs_{other_col}_Accuracy": hybrid,
                                f"{t_col}_vs_{other_col}_Fluency": fluency,
                                f"{t_col}_vs_{other_col}_Errors": error_str,
                                f"{t_col}_vs_{other_col}_Diff": diff
                            })
                    elif mode == "Standalone Student Assessment":
                        fluency, cosine_sim, bleu, hybrid, error_str = enhanced_error_detection(
                            trans_text,
                            embed_cache=embed_cache
                        )
                        row_result.update({
                            f"{t_col}_Fluency": fluency,
                            f"{t_col}_Cosine": cosine_sim,
                            f"{t_col}_BLEU": bleu,
                            f"{t_col}_Accuracy": hybrid,
                            f"{t_col}_Errors": error_str
                        })
                results.append(row_result)

            res_df = pd.DataFrame(results)

            # ---------------------------
            # Identify Best Translation per sentence
            # ---------------------------
            if mode == "Reference-based":
                acc_cols = [c for c in res_df.columns if c.endswith("_Accuracy")]
                res_df["Best_Translation"] = res_df[acc_cols].idxmax(axis=1)
                res_df["Best_Translation"] = res_df["Best_Translation"].str.replace("_Accuracy","")

            st.dataframe(res_df.head(20))

            # ---------------------------
            # Low-Quality Flags
            # ---------------------------
            st.subheader("⚠️ Low-Quality Translation Flags")
            flagged_cols = []
            for col in res_df.columns:
                if col.endswith("_Fluency"):
                    flag_col = col.replace("_Fluency", "_Low_Fluency_Flag")
                    res_df[flag_col] = res_df[col].apply(lambda x: "⚠️" if x < 3 else "")
                    flagged_cols.append(flag_col)
                elif col.endswith("_Accuracy"):
                    flag_col = col.replace("_Accuracy", "_Low_Accuracy_Flag")
                    res_df[flag_col] = res_df[col].apply(lambda x: "⚠️" if x < 0.6 else "")
                    flagged_cols.append(flag_col)
                elif col.endswith("_BLEU"):
                    flag_col = col.replace("_BLEU", "_Low_BLEU_Flag")
                    res_df[flag_col] = res_df[col].apply(lambda x: "⚠️" if x < 0.5 else "")
                    flagged_cols.append(flag_col)
            if flagged_cols:
                st.dataframe(res_df[flagged_cols].head(20))

            # ---------------------------
            # Heatmaps
            # ---------------------------
            if mode == "Reference-based":
                st.subheader("📈 Per-Sentence Similarity Heatmaps")
                metric_sets = {
                    "BLEU": [c for c in res_df.columns if c.endswith("_BLEU")],
                    "Cosine": [c for c in res_df.columns if c.endswith("_Cosine")],
                    "Accuracy": [c for c in res_df.columns if c.endswith("_Accuracy")]
                }
                for metric, cols in metric_sets.items():
                    if not cols:
                        continue
                    plt.figure(figsize=(12, len(res_df)*0.3 + 2))
                    sns.heatmap(res_df[cols], annot=True, fmt=".2f", cmap="YlGnBu",
                                cbar_kws={'label': f'{metric} Score'})
                    plt.ylabel("Sentence Index")
                    plt.xlabel("Translation / Student")
                    plt.title(f"{metric} per Sentence")
                    st.pyplot(plt)

            # ---------------------------
            # Dashboard Metrics
            # ---------------------------
            st.subheader("📊 Dashboard Summary")
            if mode == "Reference-based":
                metrics = ["BLEU", "Cosine", "Accuracy", "Fluency"]
            elif mode == "Pairwise Comparison":
                metrics = ["BLEU", "Cosine", "Accuracy"]
            elif mode == "Standalone Student Assessment":
                metrics = ["Fluency"]

            for metric in metrics:
                plt.figure(figsize=(10,4))
                metric_cols = [c for c in res_df.columns if c.endswith(metric)]
                if metric_cols:
                    sns.boxplot(data=res_df[metric_cols])
                    plt.ylabel(metric)
                    plt.title(f"{metric} Distribution Across Students / Translations")
                    st.pyplot(plt)

            # ---------------------------
            # Clean CSV Export
            # ---------------------------
            st.subheader("📥 Export Cleaned Results")
            preferred_order = []
            for base in translation_cols:
                for metric in ["Accuracy", "BLEU", "Cosine", "Fluency", "Errors"]:
                    matches = [c for c in res_df.columns if c.startswith(base) and c.endswith(metric)]
                    preferred_order.extend(matches)
            flag_cols = [c for c in res_df.columns if "Flag" in c]
            preferred_order.extend(flag_cols)
            other_cols = [c for c in res_df.columns if c not in preferred_order]
            ordered_cols = preferred_order + other_cols
            res_df = res_df[ordered_cols]

            # Human-readable column names
            res_df.columns = (
                res_df.columns
                .str.replace("_", " ")
                .str.replace(" BLEU", " (BLEU)")
                .str.replace(" Accuracy", " (Hybrid Accuracy)")
                .str.replace(" Cosine", " (Semantic Cosine)")
                .str.replace(" Fluency", " (Fluency)")
                .str.replace(" Errors", " (Error Categories)")
                .str.replace(" Flag", " ⚠️")
            )
            res_df = res_df.applymap(lambda x: round(x,3) if isinstance(x,(float,int)) else x)
            st.dataframe(res_df.head(20))
            csv = res_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button("Download Full Analysis Results (Clean CSV)", csv,
                               "translation_analysis_clean.csv", "text/csv")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload CSV, Excel, or Word file to begin analysis.")
