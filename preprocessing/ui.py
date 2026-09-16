"""
Data Preprocessing page for AgriSense.
Integrates AgriculturalDataPreprocessor pipeline with real data diagnostics,
quality metrics, interactive Plotly charts, and clean dataset export.
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

def render_preprocessing_page(df_raw: pd.DataFrame):
    """Render interactive Data Preprocessing dashboard using actual dataset."""
    render_header(
        "Agricultural Data Loading & Preprocessing Engine",
        "Inspect, validate, clean, scale, and partition soil nutrient and climate data using a reproducible preprocessing pipeline."
    )

    # Initialize preprocessor pipeline instance
    preprocessor = AgriculturalDataPreprocessor(df_raw)
    raw_meta = preprocessor.get_raw_metadata()

    # Preprocessing Pipeline Controls in Sidebar / Top
    st.markdown("### Preprocessing Pipeline Settings")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)

    with p_col1:
        impute_strategy = st.selectbox("Missing Value Imputation", ["median", "mean", "zero"])
    with p_col2:
        remove_duplicates = st.checkbox("Remove Exact Duplicate Rows", value=True)
    with p_col3:
        handle_invalid = st.checkbox("Audit & Fix Invalid Values (pH / Negatives)", value=True)
    with p_col4:
        remove_outliers = st.checkbox("Filter IQR Outliers", value=False)
        iqr_mult = st.slider("IQR Multiplier", 1.0, 3.0, 1.5, step=0.1) if remove_outliers else 1.5

    # Execute Preprocessing Pipeline dynamically
    df_clean, summary = preprocessor.execute_pipeline(
        impute_strategy=impute_strategy,
        remove_duplicates=remove_duplicates,
        handle_invalid=handle_invalid,
        remove_outliers=remove_outliers,
        iqr_multiplier=iqr_mult
    )

    st.markdown("---")

    # High Level Summary Metrics (100% computed from real dataset)
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        render_metric_card("Raw Dimensions", f"{raw_meta['total_rows']:,} × {raw_meta['total_cols']}", "Before processing")
    with m2:
        render_metric_card("Cleaned Dimensions", f"{summary['final_rows']:,} × {len(df_clean.columns)}", "After processing")
    with m3:
        render_metric_card("Duplicates Removed", f"{summary['duplicates_removed']}", "Exact row copies")
    with m4:
        render_metric_card("Missing Imputed", f"{summary['missing_cells_imputed']}", f"{impute_strategy} strategy")
    with m5:
        render_metric_card("Outliers Filtered", f"{summary['outliers_removed']}", f"{iqr_mult}× IQR rule")

    st.markdown("---")

    # Interactive Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Overview & Quality Audit",
        "Outlier & Boxplot Analysis",
        "Descriptive Statistics & Distributions",
        "Correlation & Crop Distribution",
        "Scaling, Split & Export"
    ])

    with tab1:
        st.subheader("Raw & Cleaned Dataset Inspection")

        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            st.markdown("**Raw Dataset Preview**")
            st.dataframe(df_raw.head(8), use_container_width=True)
        with sub_col2:
            st.markdown("**Cleaned Dataset Preview**")
            st.dataframe(df_clean.head(8), use_container_width=True)

        st.markdown("### Missing Value & Data Type Analysis")
        missing_df = preprocessor.get_missing_analysis(df_raw)
        st.dataframe(missing_df, use_container_width=True)

        st.markdown("### Duplicate Rows Audit")
        dup_rows = preprocessor.get_duplicate_rows()
        if len(dup_rows) > 0:
            st.write(f"Found {len(dup_rows)} duplicate row records in dataset:")
            st.dataframe(dup_rows, use_container_width=True)
        else:
            st.info("No exact duplicate records detected in the dataset.")

    with tab2:
        st.subheader("Outlier Detection (IQR Method)")
        outlier_info = preprocessor.detect_outliers_iqr(df_raw, multiplier=iqr_mult)

        st.write(f"Total rows flagged with at least one outlier: **{outlier_info['total_outlier_rows']}**")

        # Outlier count per feature table
        outlier_list = []
        for feat, details in outlier_info["by_feature"].items():
            outlier_list.append({
                "Feature": feat,
                "Q1 (25%)": details["q1"],
                "Q3 (75%)": details["q3"],
                "IQR": details["iqr"],
                "Lower Bound": details["lower_bound"],
                "Upper Bound": details["upper_bound"],
                "Outlier Count": details["count"],
                "Outlier Pct (%)": details["percentage"]
            })
        outlier_df = pd.DataFrame(outlier_list)
        st.dataframe(outlier_df, use_container_width=True)

        st.markdown("### Interactive Boxplots")
        box_fig = plot_box_plots(df_clean, AgriculturalDataPreprocessor.NUMERIC_FEATURES)
        st.plotly_chart(box_fig, use_container_width=True)

    with tab3:
        st.subheader("Descriptive Statistics")
        desc_stats = preprocessor.get_descriptive_stats(df_clean)
        st.dataframe(desc_stats, use_container_width=True)

        st.markdown("### Feature Distribution Inspector")
        selected_feature = st.selectbox("Select Feature for Distribution Plot", AgriculturalDataPreprocessor.NUMERIC_FEATURES)
        hist_fig = plot_feature_histogram(df_clean, selected_feature)
        st.plotly_chart(hist_fig, use_container_width=True)

    with tab4:
        st.subheader("Feature Correlation & Target Class Balance")

        c_col1, c_col2 = st.columns(2)
        with c_col1:
            st.markdown("### Pearson Correlation Heatmap")
            corr_fig = plot_correlation_heatmap(df_clean)
            st.plotly_chart(corr_fig, use_container_width=True)
        with c_col2:
            st.markdown("### Crop Label Class Distribution")
            crop_fig = plot_crop_distribution(df_clean)
            if crop_fig:
                st.plotly_chart(crop_fig, use_container_width=True)

    with tab5:
        st.subheader("Feature Scaling, Train-Test Partitioning & Export")

        sc_col1, sc_col2 = st.columns([1, 1])

        with sc_col1:
            st.markdown("### Feature Scaling")
            scale_method = st.selectbox("Scaling Method", ["StandardScaler (Z-Score)", "MinMaxScaler (0-1 Range)"])
            scaled_df, _ = preprocessor.scale_features(df_clean, method=scale_method.split()[0])
            st.markdown("**Scaled Dataset Sample**")
            scaled_cols = [f"{c}_scaled" for c in AgriculturalDataPreprocessor.NUMERIC_FEATURES]
            st.dataframe(scaled_df[scaled_cols].head(6), use_container_width=True)

        with sc_col2:
            st.markdown("### Train/Test Data Split")
            test_ratio = st.slider("Test Set Ratio (%)", min_value=10, max_value=40, value=20, step=5)
            split_res = preprocessor.split_data(df_clean, test_size=test_ratio/100.0)

            st.write(f"Training Set Samples (X_train): **{split_res['X_train_shape'][0]}** rows")
            st.write(f"Testing Set Samples (X_test): **{split_res['X_test_shape'][0]}** rows")
            st.write(f"Features (X): **{split_res['X_train_shape'][1]}** columns")

        st.markdown("---")
        st.subheader("Export Cleaned Dataset")
        csv_bytes = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cleaned Crop Dataset (CSV)",
            data=csv_bytes,
            file_name="Crop_recommendation_cleaned.csv",
            mime="text/csv"
        )
