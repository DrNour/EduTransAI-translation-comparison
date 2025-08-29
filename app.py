import streamlit as st
import pandas as pd
from docx import Document
from io import BytesIO

st.title("Translation Alignment Tool (19 IDs)")

# --- Function to read Word file and extract paragraphs ---
def read_word(file):
    doc = Document(file)
    data = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return data

# --- File uploader ---
uploaded_file = st.file_uploader("Upload Word or Excel file", type=["docx", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".docx"):
        texts = read_word(uploaded_file)
        st.write("✅ Extracted paragraphs:", len(texts))

        # Create empty DataFrame with 19 ID columns
        columns = [f"ID{i}" for i in range(1, 20)]
        df = pd.DataFrame(columns=columns)

        # Fill rows sequentially (4 per set: Source, GT, Student, Reference)
        for i in range(0, len(texts), 4):
            row = {}
            row["ID1"] = texts[i] if i < len(texts) else ""
            row["ID2"] = texts[i+1] if i+1 < len(texts) else ""
            row["ID3"] = texts[i+2] if i+2 < len(texts) else ""
            row["ID4"] = texts[i+3] if i+3 < len(texts) else ""
            df.loc[len(df)] = row

        st.dataframe(df, use_container_width=True)

    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
        st.dataframe(df, use_container_width=True)

    # --- Download button ---
    output = BytesIO()
    df.to_excel(output, index=False)
    st.download_button(
        label="Download Aligned Excel",
        data=output.getvalue(),
        file_name="aligned_translations.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
