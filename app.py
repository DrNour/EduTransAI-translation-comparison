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
import nltk
from io import BytesIO
from docx import Document
import matplotlib

# ===========================
# Fix NLTK punkt issue
# ===========================
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ===========================
# Fix font for Arabic/Unicode
# ===========================
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ===========================
# Streamlit page config
# ===========================
st.set_page_config(page_title="EduTransAI - Translation Assessment", layout="wide")
st.title("📊 EduTransAI - Translation Comparison & Student Assessment")

# ===========================
# Load Sentence Transformer model
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
        emb = model.encode(text)
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
        if cosine_similarity(model.encode(reference), model.encode(text)) < 0.7:
            errors.append("Semantic / Meaning")
    
    if other_texts:
        for other in other_texts:
            sim = cosine_similarity(model.encode(text), model.encode(other))
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

df = None
if uploaded_file:
    try:
        # ---------- CSV ----------
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding="utf-8", na_filter=False)
        # ---------- Excel ----------
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            xls = pd.ExcelFile(uploaded_file)
            sheet_dfs = []
            for sheet in xls.sheet_names:
                sheet_df = pd.read_excel(xls, sheet_name=sheet, na_filter=False, engine="openpyxl")
                sheet_dfs.append(sheet_df)
            df = pd.concat(sheet_dfs, ignore_index=True)
        # ---------- Word ----------
        elif uploaded_file.name.endswith(".docx"):
            doc = Document(uploaded_file)
            table_dfs = []
            for table in doc.tables:
                table_data = []
                for i, row in enumerate(table.rows):
                    row_data = [cell.text.strip() for cell in row.cells]
                    table_data.append(row_data)
                if table_data:
                    table_dfs.append(pd.DataFrame(table_data[1:], columns=table_data[0]))
            if table_dfs:
                df = pd.concat(table_dfs, ignore_index=True)
            else:
                st.error("No tables found in the Word document.")
        else:
            st.error("Unsupported file type.")

        # ---------- Prepare Data ----------
        if df is not None:
            df = df.applymap(lambda x: str(x) if x is not None else "")
            df.columns = df.columns.str.strip().fillna("Unnamed")
            df = df.fillna("")
            st.subheader("Preview of Uploaded Data")
            st.dataframe(df.head())

            # ---------- Automatic column detection with override ----------
            keywords_ref = ['source', 'reference', 'original', 'text']
            ref_candidates = [c for c in df.columns if any(k in c.lower() for k in keywords_ref)]

            if ref_candidates:
                auto_source_col = ref_candidates[0]
            else:
                col_lengths = df.applymap(lambda x: len(str(x))).sum()
                auto_source_col = col_lengths.idxmin()

            auto_translation_cols = [c for c in df.columns if c != auto_source_col]

            st.subheader("Column Selection")
            st.info(f"Auto-detected reference/source column: **{auto_source_col}**")
            st.info(f"Auto-detected student/translation columns: **{', '.join(auto_translation_cols)}**")

            # Allow user to override
            source_col = st.selectbox(
                "Select reference/source column (you can override detection):",
                df.columns,
                index=df.columns.get_loc(auto_source_col)
            )

            translation_cols = st.multiselect(
                "Select translation/student columns (you can override detection):",
                [c for c in df.columns if c != source_col],
                default=[c for c in df.columns if c in auto_translation_cols]
            )

            # ---------- Assessment mode ----------
            mode = st.radio("Select assessment mode:",
                            ["Reference-based", "Pairwise Comparison", "Standalone Student Assessment"])

            # ---------- Analysis ----------
            if translation_cols and st.button("Run Analysis"):
                st.subheader("✅ Analysis Results")
                smoothie = SmoothingFunction().method4
                results = []

                # Precompute embeddings
                all_texts = []
                for t_col in translation_cols:
                    all_texts.extend(df[t_col].astype(str))
                embeddings = model.encode(all_texts, show_progress_bar=True)
                embed_cache = {text: emb for text, emb in zip(all_texts, embeddings)}

                for idx, row in df.iterrows():
                    row_result = {}
                    if source_col:
                        source_text = str(row[source_col])
                        source_emb = model.encode(source_text)
                        row_result["Source"] = source_text
                    for t_col in translation_cols:
                        trans_text = str(row[t_col])
                        trans_emb = embed_cache[trans_text]

                        if mode == "Reference-based":
                            bleu = sentence_bleu(
                                [word_tokenize(str(row[source_col]))],
                                word_tokenize(trans_text),
                                smoothing_function=smoothie
                            )
                            fluency, style_score, error_str = enhanced_error_detection(
                                trans_text,
                                reference=str(row[source_col]),
                                embed_cache=embed_cache
                            )
                            seq = difflib.ndiff(str(row[source_col]).split(), trans_text.split())
                            diff = ' '.join([f"[{s}]" if s.startswith('-') or s.startswith('+') else s for s in seq])
                            row_result[f"{t_col}_BLEU"] = round(bleu,3)
                            row_result[f"{t_col}_Fluency"] = fluency
                            row_result[f"{t_col}_Style"] = style_score
                            row_result[f"{t_col}_Diff"] = diff
                            row_result[f"{t_col}_Errors"] = error_str

                        elif mode == "Pairwise Comparison":
                            for other_col in translation_cols:
                                if other_col == t_col:
                                    continue
                                other_text = str(row[other_col])
                                other_emb = embed_cache[other_text]
                                bleu = sentence_bleu([other_text.split()], trans_text.split(), smoothing_function=smoothie)
                                fluency, style_score, error_str = enhanced_error_detection(
                                    trans_text,
                                    other_texts=[other_text],
                                    embed_cache=embed_cache
                                )
                                seq = difflib.ndiff(other_text.split(), trans_text.split())
                                diff = ' '.join([f"[{s}]" if s.startswith('-') or s.startswith('+') else s for s in seq])
                                row_result[f"{t_col}_vs_{other_col}_BLEU"] = round(bleu,3)
                                row_result[f"{t_col}_vs_{other_col}_Style"] = style_score
                                row_result[f"{t_col}_vs_{other_col}_Diff"] = diff
                                row_result[f"{t_col}_vs_{other_col}_Errors"] = error_str

                        elif mode == "Standalone Student Assessment":
                            fluency, style_score, error_str = enhanced_error_detection(
                                trans_text,
                                embed_cache=embed_cache
                            )
                            row_result[f"{t_col}_Fluency"] = fluency
                            row_result[f"{t_col}_Style"] = style_score
                            row_result[f"{t_col}_Errors"] = error_str

                    results.append(row_result)

                res_df = pd.DataFrame(results)
                st.dataframe(res_df.head(20))

                # ---------- Low-Quality Flags ----------
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
                    st.subheader("⚠️ Low-Quality Translation Flags")
                    st.dataframe(res_df[flagged_cols].head(20))

                # ---------- Download ----------
                csv = res_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button("Download Full Analysis Results", csv, "translation_analysis_results.csv", "text/csv")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload CSV, Excel, or Word file to begin analysis.")
