"""
Data Explorer UI module for AgriSense.
Provides user-facing dataset exploration, data quality auditing, feature distribution inspector,
and an expandable Advanced Data Processing section for technical controls.
"""

import streamlit as st
import pandas as pd
from utils.helpers import render_header, render_metric_card
from preprocessing.pipeline import AgriculturalDataPreprocessor
from visualization.charts import (
    plot_box_plots,
    plot_crop_distribution,
    plot_correlation_heatmap,
    plot_feature_histogram
)

def render_data_explorer_page(df_raw: pd.DataFrame):
    """Render AgriSense Data Explorer page."""
    render_header(
        "Data Explorer",
        "Explore agricultural telemetry, audit data quality, examine soil and weather variables, and inspect relationships."
    )

    preprocessor = AgriculturalDataPreprocessor(df_raw)
    raw_meta = preprocessor.get_raw_metadata()

    # -------------------------------------------------------------------------
    # SECTION 1: DATASET OVERVIEW KPI CARDS
    # -------------------------------------------------------------------------
    st.subheader("Dataset Overview")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        render_metric_card("Record Count", f"{raw_meta['total_rows']:,}", "Total dataset rows")
    with m2:
        render_metric_card("Feature Count", f"{raw_meta['total_cols']}", "Attribute columns")
    with m3:
        render_metric_card("Missing Cells", f"{raw_meta['total_missing_cells']}", "Null cell count")
    with m4:
        render_metric_card("Duplicate Rows", f"{raw_meta['duplicate_rows']}", "Exact duplicate copies")
    with m5:
        completeness = ((1 - raw_meta['total_missing_cells'] / (raw_meta['total_rows'] * raw_meta['total_cols'])) * 100)
        render_metric_card("Completeness", f"{completeness:.1f}%", "Valid cell ratio")

    st.markdown("---")

    # Interactive Explorer Tabs
    tab_quality, tab_vars, tab_crop, tab_rel = st.tabs([
        "Data Quality",
        "Agricultural Variables",
        "Crop Distribution",
        "Relationships"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: DATA QUALITY
    # -------------------------------------------------------------------------
    with tab_quality:
        st.subheader("Data Quality & Integrity Audit")

        st.markdown("### Dataset Missing Value Breakdown")
        missing_df = preprocessor.get_missing_analysis(df_raw)
        st.dataframe(missing_df, use_container_width=True)

        q_c1, q_c2 = st.columns(2)
        with q_c1:
            st.markdown("### Duplicate Records Audit")
            dup_rows = preprocessor.get_duplicate_rows()
            if len(dup_rows) > 0:
                st.write(f"Detected **{len(dup_rows)}** duplicate row copies in raw dataset:")
                st.dataframe(dup_rows.head(6), use_container_width=True)
            else:
                st.info("Zero exact duplicate records detected in the dataset.")

        with q_c2:
            st.markdown("### Invalid Values & Outlier Overview")
            outlier_info = preprocessor.detect_outliers_iqr(df_raw, multiplier=1.5)
            st.write(f"Invalid pH Records (<0 or >14): **{raw_meta['invalid_ph_records']}**")
            st.write(f"Negative Telemetry Records: **{raw_meta['negative_value_records']}**")
            st.write(f"Total Rows Flagged with Outliers (1.5× IQR): **{outlier_info['total_outlier_rows']}**")

        st.markdown("### Outliers Box Plot Inspector")
        box_fig = plot_box_plots(df_raw, AgriculturalDataPreprocessor.NUMERIC_FEATURES)
        st.plotly_chart(box_fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: AGRICULTURAL VARIABLES
    # -------------------------------------------------------------------------
    with tab_vars:
        st.subheader("Soil & Microclimate Feature Inspector")

        st.markdown("### Descriptive Statistics")
        desc_stats = preprocessor.get_descriptive_stats(df_raw)
        st.dataframe(desc_stats, use_container_width=True)

        st.markdown("### Feature Distribution Inspector")
        selected_feat = st.selectbox(
            "Select Agricultural Variable for Histogram Distribution",
            AgriculturalDataPreprocessor.NUMERIC_FEATURES,
            index=0
        )
        hist_fig = plot_feature_histogram(df_raw, selected_feat)
        st.plotly_chart(hist_fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: CROP DISTRIBUTION
    # -------------------------------------------------------------------------
    with tab_crop:
        st.subheader("Crop Label Class Distribution")
        crop_fig = plot_crop_distribution(df_raw)
        if crop_fig:
            st.plotly_chart(crop_fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 4: RELATIONSHIPS
    # -------------------------------------------------------------------------
    with tab_rel:
        st.subheader("Nutrient & Climate Relationships")
        corr_fig = plot_correlation_heatmap(df_raw)
        st.plotly_chart(corr_fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # SECONDARY SECTION: ADVANCED DATA PROCESSING (Technical Controls)
    # -------------------------------------------------------------------------
    st.markdown("---")
    with st.expander("Advanced Data Processing & Pipeline Controls"):
        st.subheader("Interactive Preprocessing Pipeline Engine")
        st.caption("Configure technical cleaning rules, scaling strategies, and train/test dataset splits:")

        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1:
            impute_strategy = st.selectbox("Missing Imputation Strategy", ["median", "mean", "zero"])
        with p_col2:
            remove_duplicates = st.checkbox("Remove Duplicate Rows", value=True)
        with p_col3:
            handle_invalid = st.checkbox("Fix Invalid Values", value=True)
        with p_col4:
            remove_outliers = st.checkbox("Filter IQR Outliers", value=False)
            iqr_mult = st.slider("IQR Multiplier", 1.0, 3.0, 1.5, step=0.1) if remove_outliers else 1.5

        # Execute Pipeline dynamically
        df_clean, summary = preprocessor.execute_pipeline(
            impute_strategy=impute_strategy,
            remove_duplicates=remove_duplicates,
            handle_invalid=handle_invalid,
            remove_outliers=remove_outliers,
            iqr_multiplier=iqr_mult
        )

        st.markdown("### Preprocessing Pipeline Summary Results")
        ps1, ps2, ps3, ps4 = st.columns(4)
        with ps1:
            st.write(f"Cleaned Rows: **{summary['final_rows']:,}**")
        with ps2:
            st.write(f"Duplicates Removed: **{summary['duplicates_removed']}**")
        with ps3:
            st.write(f"Missing Cells Imputed: **{summary['missing_cells_imputed']}**")
        with ps4:
            st.write(f"Outliers Filtered: **{summary['outliers_removed']}**")

        st.markdown("---")
        sc_col1, sc_col2 = st.columns(2)
        with sc_col1:
            st.markdown("### Feature Scaling")
            scale_method = st.selectbox("Scaling Method", ["StandardScaler (Z-Score)", "MinMaxScaler (0-1 Range)"])
            scaled_df, _ = preprocessor.scale_features(df_clean, method=scale_method.split()[0])
            scaled_cols = [f"{c}_scaled" for c in AgriculturalDataPreprocessor.NUMERIC_FEATURES]
            st.dataframe(scaled_df[scaled_cols].head(5), use_container_width=True)

        with sc_col2:
            st.markdown("### Train/Test Data Split")
            test_ratio = st.slider("Test Set Ratio (%)", 10, 40, 20, 5)
            split_res = preprocessor.split_data(df_clean, test_size=test_ratio/100.0)
            st.write(f"Train Set: **{split_res['X_train_shape'][0]}** rows")
            st.write(f"Test Set: **{split_res['X_test_shape'][0]}** rows")

        st.markdown("---")
        st.subheader("Export Cleaned Dataset")
        csv_bytes = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cleaned Dataset (CSV)",
            data=csv_bytes,
            file_name="Crop_recommendation_cleaned.csv",
            mime="text/csv"
        )
