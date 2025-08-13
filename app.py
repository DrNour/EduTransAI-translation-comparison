import streamlit as st
import pandas as pd
import requests
import numpy as np
import difflib

# ===============================
# Hugging Face API Setup
# ===============================
HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
HF_HEADERS = {"Authorization": f"Bearer {st.secrets['HUGGINGFACE_API_TOKEN']}"}

def get_embedding(text):
    """Get sentence embedding from Hugging Face API"""
    try:
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": text})
        response.raise_for_status()
        embedding = response.json()
        # embedding is [tokens][dimensions], average over tokens
        return np.mean(embedding[0], axis=0)
    except Exception as e:
        st.warning(f"Embedding API failed: {e}")
        # fallback to zero vector (384 dims for this model)
        return np.zeros(384)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

# ===============================
# Streamlit App
# ===============================
st.title("EduTransAI - Translation Comparison")

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

        results = []

        for idx, row in df.iterrows():
            source_text = str(row[source_col])
            trans_texts = [str(row[c]) for c in translation_cols]

            # --- Get embeddings ---
            embeddings = [get_embedding(t) for t in trans_texts]

            # --- Compute pairwise cosine similarities ---
            sims = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = cosine_similarity(embeddings[i], embeddings[j])
                    sims.append(sim)

            avg_similarity = np.mean(sims) if sims else 0

            # --- Simple diff highlighting ---
            diffs = []
            for t in trans_texts:
                seq = difflib.ndiff(source_text.split(), t.split())
                diff = ' '.join([f"[{s}]" if s.startswith('-') or s.startswith('+') else s for s in seq])
                diffs.append(diff)

            results.append({
                "Source": source_text,
                **{f"Translation_{i+1}": t for i, t in enumerate(trans_texts)},
                "Avg_Similarity": avg_similarity,
                **{f"Diff_{i+1}": d for i, d in enumerate(diffs)}
            })

        # --- Show results ---
        res_df = pd.DataFrame(results)
        st.subheader("Comparison Results")
        st.dataframe(res_df)

        # --- Export results ---
        csv = res_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name="translation_comparison_results.csv",
            mime="text/csv"
        )

    else:
        st.warning("Please select at least 2 translations to compare.")
