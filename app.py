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

            # stricter (max) for lexical/fluency/style; empirical for Accuracy
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
            # 📌 Examples & Issue Mining (interactive)
            # ---------------------------
            st.subheader("🔍 Examples & Issue Mining")
            top_n = st.slider("How many examples per issue?", 3, 50, 10, 1)
            thr_pack = {
                "LexLow": LEX_LOW,
                "AccLow": ACC_LOW,
                "FluLow": FLU_LOW,
                "StyleLow": STYLE_LOW,
            }
            ex_dict = gather_issue_examples(
                res_df=res_df,
                df_orig=df,
                translation_cols=translation_cols,
                mode=mode,
                source_col=source_col if mode == "Reference-based" else None,
                top_n=top_n,
                thresholds=thr_pack,
                consistency_tolerance=consistency_tolerance,
                paraphrase_sem_hi=paraphrase_sem_hi,
                low_lexical_for_paraphrase=low_lexical_for_paraphrase,
                drift_sem_lo=drift_sem_lo,
            )
            tabs = st.tabs(list(ex_dict.keys()))
            for tab, (label, exdf) in zip(tabs, ex_dict.items()):
                with tab:
                    if exdf.empty:
                        st.info(f"No examples for **{label}** with current thresholds.")
                    else:
                        st.dataframe(exdf)
                        c1, c2 = st.columns(2)
                        with c1:
                            csv = exdf.to_csv(index=False).encode("utf-8-sig")
                            st.download_button(
                                f"Download {label} (CSV)",
                                csv,
                                f"examples_{label.replace(' ','_').replace('/','-').lower()}.csv",
                                "text/csv",
                            )
                        with c2:
                            try:
                                docx_bytes = examples_to_docx(exdf, title=f"{label} Examples")
                                st.download_button(
                                    f"Download {label} (DOCX)",
                                    docx_bytes,
                                    f"examples_{label.replace(' ','_').replace('/','-').lower()}.docx",
                                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                )
                            except Exception as e:
                                st.warning(f"DOCX export skipped: {e}")

                    # Optional: inline diffs per example (when reference available)
                    if mode == "Reference-based" and source_col and not exdf.empty:
                        with st.expander(f"Inline diffs for {label}", expanded=False):
                            for _, r in exdf.iterrows():
                                src = str(r.get("Source",""))
                                hyp = str(r.get("Translation",""))
                                st.markdown(token_diff(src, hyp), unsafe_allow_html=True)
                                st.markdown("<hr/>", unsafe_allow_html=True)

            non_empty = [(k, len(v)) for k, v in ex_dict.items() if isinstance(v, pd.DataFrame) and not v.empty and k != "Combined"]
            if non_empty:
                worst_label = sorted(non_empty, key=lambda x: x[1], reverse=True)[0][0]
                st.caption(f"Most populous bucket right now: **{worst_label}**")

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

            # Only add flag columns that are NOT already in preferred_order
            flag_cols = [c for c in res_df.columns if ("Flag" in c) and (c not in preferred_order)]

            if "Best_Translation" in res_df.columns:
                preferred_order = ["Best_Translation"] + preferred_order

            preferred_order.extend(flag_cols)

            other_cols = [c for c in res_df.columns if c not in preferred_order]
            ordered_cols = preferred_order + other_cols

            export_df = res_df[ordered_cols].copy()
            if not meta_df.empty:
                export_df = pd.concat([meta_df, export_df], axis=1)

            export_df.columns = humanize_export_columns(export_df.columns)
            export_df = export_df.applymap(lambda x: round(x, 3) if isinstance(x, (float, int)) else x)

            st.dataframe(export_df.head(20))
            csv = export_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "Download Full Analysis Results (Clean CSV)",
                csv,
                "translation_analysis_clean.csv",
                "text/csv",
            )

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
