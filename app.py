# app.py
import streamlit as st
import pandas as pd
import numpy as np
import difflib
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
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
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

def fluency_score(text):
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

def generate_explanation(row, t_col):
    """Convert scores and errors into plain-language explanation"""
    explanations = []
    fluency = row.get(f"{t_col}_Fluency", 0)
    if fluency <= 2:
        explanations.append("Very hard to read; many grammar issues.")
    elif fluency == 3:
        explanations.append("Understandable but could be more fluent.")
    else:
        explanations.append("Reads naturally and fluent.")
    style = row.get(f"{t_col}_Style", 0)
    if style < 0.5:
        explanations.append("Style differs significantly from other translations or reference.")
    elif style < 0.7:
        explanations.append("Style is acceptable but not very natural.")
    else:
        explanations.append("Style is natural and consistent.")
    bleu = row.get(f"{t_col}_BLEU", None)
    if bleu is not None:
        if bleu < 0.5:
            explanations.append("Low similarity to reference; meaning may not be fully preserved.")
        elif bleu < 0.7:
            explanations.append("Moderate similarity; most meaning preserved.")
        else:
            explanations.append("High similarity to reference; meaning preserved accurately.")
    errors = row.get(f"{t_col}_Errors", "")
    if errors and errors != "None":
        explanations.append(f"Detected issues: {errors}")
    return " ".join(explanations)

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
        mode = st.radio("Select assessment mode:",
                        ["Reference-based", "Pairwise Comparison", "Standalone Student Assessment"])

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
            all_texts = list(set(all_texts))  # remove duplicates
            embeddings = model.encode(all_texts, batch_size=64, show_progress_bar=True)
            embed_cache = {text: emb for text, emb in zip(all_texts, embeddings)}

            # ---------------------------
            # Analyze each row
            # ---------------------------
            for idx, row in df.iterrows():
                row_result = {}
                if source_col:
                    source_text = str(row[source_col])
                    row_result["Source"] = source_text
                for t_col in translation_cols:
                    trans_text = str(row[t_col])
                    if mode == "Reference-based":
                        bleu = sentence_bleu([word_tokenize(str(row[source_col]))],
                                             word_tokenize(trans_text),
                                             smoothing_function=smoothie)
                        fluency, style_score, error_str = enhanced_error_detection(
                            trans_text, reference=str(row[source_col]), embed_cache=embed_cache)
                        seq = difflib.ndiff(str(row[source_col]).split(), trans_text.split())
                        diff = ' '.join([f"[{s}]" if s.startswith('-') or s.startswith('+') else s for s in seq])
                        row_result[f"{t_col}_BLEU"] = round(bleu,3)
                        row_result[f"{t_col}_Fluency"] = fluency
                        row_result[f"{t_col}_Style"] = style_score
                        row_result[f"{t_col}_Diff"] = diff
                        row_result[f"{t_col}_Errors"] = error_str
                        # Plain-language explanation
                        row_result[f"{t_col}_Explanation"] = generate_explanation(row_result, t_col)
                    elif mode == "Pairwise Comparison":
                        for other_col in translation_cols:
                            if other_col == t_col:
                                continue
                            other_text = str(row[other_col])
                            bleu = sentence_bleu([other_text.split()], trans_text.split(), smoothing_function=smoothie)
                            fluency, style_score, error_str = enhanced_error_detection(
                                trans_text, other_texts=[other_text], embed_cache=embed_cache)
                            seq = difflib.ndiff(other_text.split(), trans_text.split())
                            diff = ' '.join([f"[{s}]" if s.startswith('-') or s.startswith('+') else s for s in seq])
                            row_result[f"{t_col}_vs_{other_col}_BLEU"] = round(bleu,3)
                            row_result[f"{t_col}_vs_{other_col}_Style"] = style_score
                            row_result[f"{t_col}_vs_{other_col}_Diff"] = diff
                            row_result[f"{t_col}_vs_{other_col}_Errors"] = error_str
                            # Plain-language explanation
                            row_result[f"{t_col}_vs_{other_col}_Explanation"] = generate_explanation(row_result, t_col)
                    elif mode == "Standalone Student Assessment":
                        fluency, style_score, error_str = enhanced_error_detection(
                            trans_text, embed_cache=embed_cache)
                        row_result[f"{t_col}_Fluency"] = fluency
                        row_result[f"{t_col}_Style"] = style_score
                        row_result[f"{t_col}_Errors"] = error_str
                        # Plain-language explanation
                        row_result[f"{t_col}_Explanation"] = generate_explanation(row_result, t_col)
                results.append(row_result)

            res_df = pd.DataFrame(results)
            st.dataframe(res_df.head(20))

            # ---------------------------
            # Download CSV
            # ---------------------------
            csv = res_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button("Download Full Analysis Results", csv, "translation_analysis_results.csv", "text/csv")

            # ---------------------------
            # Show explanations interactively
            # ---------------------------
            st.subheader("📝 Plain-Language Explanations")
            for t_col in translation_cols:
                st.markdown(f"**{t_col} Explanation**")
                for idx, row in res_df.iterrows():
                    st.write(f"Row {idx+1}: {row.get(f'{t_col}_Explanation','')}")
                    st.markdown("---")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload CSV, Excel, or Word file to begin analysis.")
