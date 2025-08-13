import streamlit as st
import pandas as pd
import numpy as np
import difflib
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import os
import pickle

# ===============================
# Load local sentence-transformers model
# ===============================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ===============================
# Embedding functions with caching
# ===============================
EMBED_CACHE_FILE = "embeddings_cache.pkl"

# Load cache if exists
if os.path.exists(EMBED_CACHE_FILE):
    with open(EMBED_CACHE_FILE, "rb") as f:
        embed_cache = pickle.load(f)
else:
    embed_cache = {}

def get_embedding(text):
    """Get embedding from local sentence-transformers model with caching"""
    if text in embed_cache:
        return embed_cache[text]
    emb = model.encode(text)
    embed_cache[text] = emb
    return emb

def save_cache():
    with open(EMBED_CACHE_FILE, "wb") as f:
        pickle.dump(embed_cache, f)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

def fluency_score(text):
    """Simple heuristic fluency score 1-5"""
    if not text.strip():
        return 1
    penalties = len(re.findall(r'\s{2,}', text)) + len(re.findall(r'[^\w\s.,;:!?]', text))
    length = len(text.split())
    score = max(1, min(5, 5 - penalties*0.5, length/20))
    return round(score, 2)

# ===============================
# Streamlit App
# ===============================
st.title("EduTransAI - Translation Comparison & Scoring (Offline, Large Corpora)")

# --- Upload CSV ---
uploaded_file = st.file_uploader(
    "Upload CSV (columns: Source, Translation1, Translation2, ...)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File loaded successfully!")
    st.write("Sample data:")
    st.dataframe(df.head())

    # --- Select columns ---
    source_col = st.selectbox("Select the source column", df.columns)
    translation_cols = st.multiselect(
        "Select translations to compare",
        [c for c in df.columns if c != source_col]
    )

    if len(translation_cols) >= 2:
        st.write(f"Comparing translations: {translation_cols}")

        # --- Precompute embeddings in batch for speed ---
        all_texts = list(df[source_col].astype(str))
        for col in translation_cols:
            all_texts.extend(list(df[col].astype(str)))

        # Compute embeddings in batch (ignores cache)
        st.info("Computing embeddings... (may take a while for large datasets)")
        batch_embeddings = model.encode(all_texts, show_progress_bar=True)
        # Assign embeddings to cache
        for text, emb in zip(all_texts, batch_embeddings):
            embed_cache[text] = emb

        results = []

        for idx, row in df.iterrows():
            source_text = str(row[source_col])
            trans_texts = [str(row[c]) for c in translation_cols]

            # --- Get embeddings from cache ---
            source_emb = embed_cache[source_text]
            embeddings = [embed_cache[t] for t in trans_texts]

            # --- Accuracy ---
            accuracies = [cosine_similarity(source_emb, e) for e in embeddings]

            # --- Style similarity ---
            style_sims = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    style_sims.append(cosine_similarity(embeddings[i], embeddings[j]))
            style_score = np.mean(style_sims) if style_sims else 0

            # --- Fluency ---
            fluency_scores = [fluency_score(t) for t in trans_texts]

            # --- Diff highlighting ---
            diffs = []
            for t in trans_texts:
                seq = difflib.ndiff(source_text.split(), t.split())
                diff = ' '.join([f"[{s}]" if s.startswith('-') or s.startswith('+') else s for s in seq])
                diffs.append(diff)

            # --- Store results ---
            result_row = {
                "Source": source_text,
                "Style_Score": round(style_score, 2),
                **{f"Translation_{i+1}": t for i, t in enumerate(trans_texts)},
                **{f"Accuracy_{i+1}": round(acc, 2) for i, acc in enumerate(accuracies)},
                **{f"Fluency_{i+1}": f for i, f in enumerate(fluency_scores)},
                **{f"Diff_{i+1}": d for i, d in enumerate(diffs)}
            }
            results.append(result_row)

        # --- Save cache ---
        save_cache()

        # --- Show results ---
        res_df = pd.DataFrame(results)
        st.subheader("Comparison Results with Scoring")
        st.dataframe(res_df)

        # --- Export results ---
        csv = res_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name="translation_comparison_results.csv",
            mime="text/csv"
        )

        # ===============================
        # Visual Dashboard
        # ===============================
        st.subheader("Visual Dashboard")

        # --- Accuracy Plot ---
        acc_cols = [c for c in res_df.columns if c.startswith("Accuracy_")]
        if acc_cols:
            st.write("**Accuracy Scores per Translation**")
            acc_df = res_df[acc_cols]
            plt.figure(figsize=(10, 4))
            sns.boxplot(data=acc_df)
            plt.ylabel("Cosine Similarity")
            plt.xticks(rotation=45)
            st.pyplot(plt)

        # --- Fluency Plot ---
        fluency_cols = [c for c in res_df.columns if c.startswith("Fluency_")]
        if fluency_cols:
            st.write("**Fluency Scores per Translation**")
            flu_df = res_df[fluency_cols]
            plt.figure(figsize=(10, 4))
            sns.boxplot(data=flu_df)
            plt.ylabel("Fluency Score (1-5)")
            plt.xticks(rotation=45)
            st.pyplot(plt)

        # --- Style Score ---
        if "Style_Score" in res_df.columns:
            st.write("**Style Score Across Translations**")
            plt.figure(figsize=(10, 4))
            sns.barplot(x=res_df.index, y=res_df["Style_Score"])
            plt.ylabel("Average Style Similarity")
            plt.xlabel("Sentence Index")
            st.pyplot(plt)

    else:
        st.warning("Please select at least 2 translations to compare.")
