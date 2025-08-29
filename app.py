# app.py
import streamlit as st
import pandas as pd
from docx import Document
import io

# ===========================
# Streamlit page config
# ===========================
st.set_page_config(page_title="EduTransAI - Word to Excel", layout="wide")
st.title("📑 Word to Excel Converter (Source + GT + Student + Reference)")

# ===========================
# File Upload
# ===========================
uploaded_file = st.file_uploader("Upload a Word file (.docx)", type=["docx"])

if uploaded_file:
    try:
        # Load Word document
        doc = Document(uploaded_file)

        # Prepare lists
        ids, sources, gts, students, refs = [], [], [], [], []
        current_id, source, gt, student, reference = None, None, None, None, None

        # Parse document
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            if text.startswith("ID"):  # Start new record
                if current_id and (source or gt or student or reference):
                    ids.append(current_id)
                    sources.append(source)
                    gts.append(gt)
                    students.append(student)
                    refs.append(reference)

                current_id = text
                source, gt, student, reference = None, None, None, None

            elif text.startswith("Source:"):
                source = text.replace("Source:", "").strip()
            elif text.startswith("GT:"):
                gt = text.replace("GT:", "").strip()
            elif text.startswith("Student:"):
                student = text.replace("Student:", "").strip()
            elif text.startswith("Reference:"):
                reference = text.replace("Reference:", "").strip()

        # Save the last one
        if current_id and (source or gt or student or reference):
            ids.append(current_id)
            sources.append(source)
            gts.append(gt)
            students.append(student)
            refs.append(reference)

        # Create DataFrame
        df = pd.DataFrame({
            "ID": ids,
            "Source": sources,
            "GT": gts,
            "Student": students,
            "Reference": refs
        })

        # Preview
        st.subheader("✅ Extracted Data")
        st.dataframe(df)

        # Export to Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Translations")

        st.download_button(
            label="Download Excel File",
            data=output.getvalue(),
            file_name="translations.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"Error while processing: {e}")
else:
    st.info("Upload a Word file with IDs, Source, GT, Student, and Reference.")
