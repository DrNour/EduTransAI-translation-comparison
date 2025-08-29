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
    """Simple tokenizer avoiding NLTK punkt."""
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

def fluency_score(text):
    """Heuristic fluency score 1-5"""
    if not text.strip():
        return 1
    penalties = len(re.findall(r'\s{2,}', text)) + len(re.findall(r'[^\w\s.,;:!?]', text))
    length = len(text.split())
    score = max(1, min(5, 5 - penalties*0.5, length/20))
    return round(score, 2)

def enhanced_error_detection(text, reference=None, other_texts=None, embed_cache=None):
    fluency = fluency_score(text)
    style_score = 0
    if embed_cache and embed_cache.values():
        emb = embed_cache[text]
        all_embeddings = np.array(list(embed_cache.values()))
        style_score = np.mean([cosine_similarity(emb, e) for e in all_embeddings])
        style_score = round(style_score, 3)
    errors = []
    if fluency < 3:
        errors.append("Fluency/Grammar")
    words = text.split()
    if len(words) < 3:
        errors.append("Too Short")
    elif len(words) > 50:
        errors.append("Too Long / Style Issue")
    if reference:
        ref_tokens = reference.split()
        diff = list(difflib.ndiff(ref_tokens, words))
        additions = sum(1 for d in diff if d.startswith('+ '))
        deletions = sum(1 for d in diff if d.startswith('- '))
        if additions > 0:
            errors.append("Addition")
        if deletions > 0:
            errors.append("Deletion")
        if cosine_similarity(embed_cache[reference], embed_cache[text]) < 0.7:
            errors.append("Semantic / Meaning")
    if other_texts:
        for other in other_texts:
            sim = cosine_similarity(embed_cache[text], embed_cache[other])
            if sim < 0.6:
                errors.append("Style / Idiomaticity")
    return fluency, style_score, ", ".join(sorted(set(errors))) if errors else "None"

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
            smoothie = SmoothingFunction().method4
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
                        bleu = sentence_bleu(
                            [simple_tokenize(source_text)],
                            simple_tokenize(trans_text),
                            smoothing_function=smoothie
                        )
                        fluency, style_score, error_str = enhanced_error_detection(
                            trans_text,
                            reference=source_text,
                            embed_cache=embed_cache
                        )
                        seq = difflib.ndiff(source_text.split(), trans_text.split())
                        diff = ' '.join([f"[{s}]" if s.startswith('-') or s.startswith('+') else s for s in seq])
                        row_result.update({
                            f"{t_col}_BLEU": round(bleu,3),
                            f"{t_col}_Fluency": fluency,
                            f"{t_col}_Style": style_score,
                            f"{t_col}_Diff": diff,
                            f"{t_col}_Errors": error_str
                        })
                    elif mode == "Pairwise Comparison":
                        for other_col in translation_cols:
                            if other_col == t_col:
                                continue
                            other_text = str(row[other_col])
                            bleu = sentence_bleu([simple_tokenize(other_text)], simple_tokenize(trans_text), smoothing_function=smoothie)
                            fluency, style_score, error_str = enhanced_error_detection(
                                trans_text,
                                other_texts=[other_text],
                                embed_cache=embed_cache
                            )
                            seq = difflib.ndiff(other_text.split(), trans_text.split())
                            diff = ' '.join([f"[{s}]" if s.startswith('-') or s.startswith('+') else s for s in seq])
                            row_result.update({
                                f"{t_col}_vs_{other_col}_BLEU": round(bleu,3),
                                f"{t_col}_vs_{other_col}_Style": style_score,
                                f"{t_col}_vs_{other_col}_Diff": diff,
                                f"{t_col}_vs_{other_col}_Errors": error_str
                            })
                    elif mode == "Standalone Student Assessment":
                        fluency, style_score, error_str = enhanced_error_detection(
                            trans_text,
                            embed_cache=embed_cache
                        )
                        row_result.update({
                            f"{t_col}_Fluency": fluency,
                            f"{t_col}_Style": style_score,
                            f"{t_col}_Errors": error_str
                        })
                results.append(row_result)

            res_df = pd.DataFrame(results)
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
                elif col.endswith("_Style"):
                    flag_col = col.replace("_Style", "_Low_Style_Flag")
                    res_df[flag_col] = res_df[col].apply(lambda x: "⚠️" if x < 0.5 else "")
                    flagged_cols.append(flag_col)
                elif col.endswith("_BLEU"):
                    flag_col = col.replace("_BLEU", "_Low_BLEU_Flag")
                    res_df[flag_col] = res_df[col].apply(lambda x: "⚠️" if x < 0.5 else "")
                    flagged_cols.append(flag_col)
            if flagged_cols:
                st.dataframe(res_df[flagged_cols].head(20))

            # ---------------------------
            # Dashboard
            # ---------------------------
            st.subheader("📊 Dashboard")
            metrics = []
            if mode == "Reference-based":
                metrics = ["BLEU", "Fluency", "Style"]
            elif mode == "Pairwise Comparison":
                metrics = ["BLEU", "Style"]
            elif mode == "Standalone Student Assessment":
                metrics = ["Fluency", "Style"]
            for metric in metrics:
                plt.figure(figsize=(10, 4))
                metric_cols = [c for c in res_df.columns if c.endswith(metric)]
                if metric_cols:
                    sns.boxplot(data=res_df[metric_cols])
                    plt.ylabel(metric)
                    plt.title(f"{metric} per Translation / Student")
                    st.pyplot(plt)

            # ---------------------------
            # Error Categories Summary
            # ---------------------------
            st.subheader("📌 Error Categories Summary")
            error_cols = [c for c in res_df.columns if c.endswith("Errors")]
            if error_cols:
                error_counts = {}
                for col in error_cols:
                    col_errors = res_df[col].apply(lambda x: str(x).split(", ") if x != "None" else [])
                    counts = {}
                    for errors_list in col_errors:
                        for e in errors_list:
                            counts[e] = counts.get(e, 0) + 1
                    error_counts[col] = counts

                all_errors = set(e for counts in error_counts.values() for e in counts)
                plot_df = pd.DataFrame(0, index=error_cols, columns=sorted(all_errors))
                for col in error_cols:
                    for e, cnt in error_counts[col].items():
                        plot_df.loc[col, e] = cnt
                plot_df.plot(kind="bar", stacked=True, figsize=(12,6), colormap="tab20")
                plt.ylabel("Number of Sentences")
                plt.xlabel("Translation / Student")
                plt.title("Distribution of Error Categories per Translation / Student")
                plt.xticks(rotation=45)
                plt.legend(title="Error Category", bbox_to_anchor=(1.05, 1), loc='upper left')
                st.pyplot(plt)

            # ---------------------------
            # Download CSV
            # ---------------------------
            csv = res_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button("Download Full Analysis Results", csv, "translation_analysis_results.csv", "text/csv")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload CSV, Excel, or Word file to begin analysis.")










































