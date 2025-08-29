# ---------- Automatic column detection ----------
keywords_ref = ['source', 'reference', 'original', 'text']
ref_candidates = [c for c in df.columns if any(k in c.lower() for k in keywords_ref)]

if ref_candidates:
    auto_source_col = ref_candidates[0]  # pick first match
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
