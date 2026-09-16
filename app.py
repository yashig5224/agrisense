"""
AgriSense — Smart Agriculture Decision Support System Using Data Mining
Main Application Entry Point & Executive Analytics Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np

from utils.theme import apply_custom_theme
from utils.helpers import render_header, render_metric_card, display_dataset_summary
from data.loader import load_crop_dataset
from visualization.charts import (
    plot_feature_histogram,
    plot_scatter,
    plot_correlation_heatmap,
    plot_crop_distribution
)

from preprocessing.ui import render_preprocessing_page
from association.ui import render_association_page
from classification.ui import render_classification_page
from classification.weka_j48 import WekaJ48Classifier
from classification.naive_bayes import AgriculturalNaiveBayesClassifier
from regression.ui import render_regression_page
from regression.engine import AgriculturalRegressionEngine
from clustering.ui import render_clustering_page
from clustering.engine import AgriculturalClusteringEngine
from decision_support.ui import render_decision_support_page

# Set Page Config
st.set_page_config(
    page_title="AgriSense — Smart Agriculture Analytics SaaS",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Strict Light-Mode Custom Theme
apply_custom_theme()

# Load Primary Dataset
@st.cache_data
def load_primary_data():
    return load_crop_dataset()

df_raw = load_primary_data()

# Sidebar Navigation
st.sidebar.markdown("### AGRI SENSE")
st.sidebar.markdown("**Agricultural Analytics Platform**")
st.sidebar.markdown("---")

navigation_selection = st.sidebar.selectbox(
    "Navigation Menu",
    [
        "Overview",
        "Data Preprocessing",
        "Association Rules",
        "Classification",
        "Regression",
        "Clustering",
        "Decision Support",
        "Model Evaluation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Metadata")
st.sidebar.write(f"Total Records: **{len(df_raw):,}**")
st.sidebar.write(f"Attributes: **{len(df_raw.columns)}**")
st.sidebar.write(f"Crop Varieties: **{df_raw['label'].nunique() if 'label' in df_raw else 0}**")

st.sidebar.markdown("---")
st.sidebar.markdown("")


# -----------------------------------------------------------------------------
# NAVIGATION ROUTER
# -----------------------------------------------------------------------------

if navigation_selection == "Overview":
    render_header(
        "AgriSense Executive Analytics Dashboard",
        "Smart Agriculture Decision Support System providing crop recommendation, soil nutrient analysis, and microclimate telemetry insights."
    )

    # Dataset Summary Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Total Records", f"{len(df_raw):,}", "Primary telemetry dataset")
    with m2:
        render_metric_card("Number of Crops", f"{df_raw['label'].nunique()}", "Distinct target categories")
    with m3:
        render_metric_card("Numerical Features", "7", "N, P, K, Temp, Humidity, pH, Rainfall")
    with m4:
        render_metric_card("Data Completeness", f"{((1 - df_raw.isna().sum().sum() / (len(df_raw)*len(df_raw.columns)))*100):.1f}%", "Valid cell ratio")

    st.markdown("---")

    # Section 1: Crop Class Balance & Distribution
    st.subheader("Crop Label Class Distribution")
    crop_fig = plot_crop_distribution(df_raw)
    if crop_fig:
        st.plotly_chart(crop_fig, use_container_width=True)

    st.markdown("---")

    # Section 2: Soil Nutrient Analysis (N, P, K)
    st.subheader("Soil Nutrient Analysis (Nitrogen, Phosphorus, Potassium)")
    n_col1, n_col2, n_col3 = st.columns(3)
    with n_col1:
        render_metric_card("Avg Nitrogen (N)", f"{df_raw['N'].mean():.1f} kg/ha", f"Min: {df_raw['N'].min():.1f} | Max: {df_raw['N'].max():.1f}")
        fig_n = plot_feature_histogram(df_raw, "N")
        st.plotly_chart(fig_n, use_container_width=True)
    with n_col2:
        render_metric_card("Avg Phosphorus (P)", f"{df_raw['P'].mean():.1f} kg/ha", f"Min: {df_raw['P'].min():.1f} | Max: {df_raw['P'].max():.1f}")
        fig_p = plot_feature_histogram(df_raw, "P")
        st.plotly_chart(fig_p, use_container_width=True)
    with n_col3:
        render_metric_card("Avg Potassium (K)", f"{df_raw['K'].mean():.1f} kg/ha", f"Min: {df_raw['K'].min():.1f} | Max: {df_raw['K'].max():.1f}")
        fig_k = plot_feature_histogram(df_raw, "K")
        st.plotly_chart(fig_k, use_container_width=True)

    st.markdown("---")

    # Section 3: Weather & Environment Telemetry Analysis
    st.subheader("Weather & Environmental Microclimate Analysis")
    e_col1, e_col2, e_col3 = st.columns(3)
    with e_col1:
        render_metric_card("Avg Temperature", f"{df_raw['temperature'].mean():.1f} °C", f"Range: {df_raw['temperature'].min():.1f} - {df_raw['temperature'].max():.1f}")
        fig_temp = plot_feature_histogram(df_raw, "temperature")
        st.plotly_chart(fig_temp, use_container_width=True)
    with e_col2:
        render_metric_card("Avg Humidity", f"{df_raw['humidity'].mean():.1f} %", f"Range: {df_raw['humidity'].min():.1f} - {df_raw['humidity'].max():.1f}")
        fig_hum = plot_feature_histogram(df_raw, "humidity")
        st.plotly_chart(fig_hum, use_container_width=True)
    with e_col3:
        render_metric_card("Avg Annual Rainfall", f"{df_raw['rainfall'].mean():.1f} mm", f"Range: {df_raw['rainfall'].min():.1f} - {df_raw['rainfall'].max():.1f}")
        fig_rain = plot_feature_histogram(df_raw, "rainfall")
        st.plotly_chart(fig_rain, use_container_width=True)

    st.markdown("---")

    # Section 4: Correlation Matrix & Dataset Records
    st.subheader("Nutrient & Climate Pearson Correlation Heatmap")
    heatmap_fig = plot_correlation_heatmap(df_raw)
    st.plotly_chart(heatmap_fig, use_container_width=True)

    st.markdown("### Key Dataset Statistics")
    num_df = df_raw.select_dtypes(include=[np.number])
    st.dataframe(num_df.describe().T.round(2), use_container_width=True)

    st.markdown("---")

    # Final Computed Insights Box
    st.subheader("Final Data Insights & Agronomic Summary")
    high_rain_crop = df_raw.groupby("label")["rainfall"].mean().idxmax()
    high_n_crop = df_raw.groupby("label")["N"].mean().idxmax()

    st.markdown(f"""
    - **High Water Requirement Crop**: Historical dataset telemetry shows **{high_rain_crop.upper()}** requires the highest average rainfall ({df_raw[df_raw['label']==high_rain_crop]['rainfall'].mean():.1f} mm).
    - **Nitrogen Intensive Crop**: **{high_n_crop.upper()}** exhibits the highest average Nitrogen demand ({df_raw[df_raw['label']==high_n_crop]['N'].mean():.1f} kg/ha).
    - **Soil pH Distribution**: Mean soil pH across dataset records is **{df_raw['ph'].mean():.2f}**, representing slightly acidic to neutral agronomic conditions.
    """)

elif navigation_selection == "Data Preprocessing":
    render_preprocessing_page(df_raw)

elif navigation_selection == "Association Rules":
    render_association_page(df_raw)

elif navigation_selection == "Classification":
    render_classification_page(df_raw)

elif navigation_selection == "Regression":
    render_regression_page(df_raw)

elif navigation_selection == "Clustering":
    render_clustering_page(df_raw)

elif navigation_selection == "Decision Support":
    render_decision_support_page(df_raw)

elif navigation_selection == "Model Evaluation":
    render_header(
        "Unified Model Evaluation Area",
        "Empirical evaluation metrics calculated dynamically across classification, regression, and clustering algorithms."
    )

    clean_df = df_raw.dropna().copy()

    # Dynamic Computation of Classification Benchmarks
    j48_eval = WekaJ48Classifier.run_j48(clean_df, test_size=0.20)
    nb_eval = AgriculturalNaiveBayesClassifier.train_and_evaluate(clean_df, test_size=0.20)

    # Dynamic Computation of Regression Benchmarks
    reg1_df, reg1_target, reg1_feats = AgriculturalRegressionEngine.load_regression_dataset("Dataset 1: Crop Yield Prediction")
    reg1_res = AgriculturalRegressionEngine.train_and_evaluate(reg1_df, reg1_target, reg1_feats, model_type="Random Forest Regressor")

    reg2_df, reg2_target, reg2_feats = AgriculturalRegressionEngine.load_regression_dataset("Dataset 2: Soil Organic Carbon Content")
    reg2_res = AgriculturalRegressionEngine.train_and_evaluate(reg2_df, reg2_target, reg2_feats, model_type="Random Forest Regressor")

    # Dynamic Computation of Clustering Benchmarks
    clust1_df, clust1_feats = AgriculturalClusteringEngine.load_clustering_dataset("Dataset 1: Crop & Microclimate Telemetry")
    clust1_res = AgriculturalClusteringEngine.execute_kmeans(clust1_df, clust1_feats, n_clusters=4)

    clust2_df, clust2_feats = AgriculturalClusteringEngine.load_clustering_dataset("Dataset 2: Soil & Water Quality")
    clust2_res = AgriculturalClusteringEngine.execute_kmeans(clust2_df, clust2_feats, n_clusters=3)

    tab1, tab2, tab3 = st.tabs([
        "Classification Benchmarks (J48 vs. Naive Bayes)",
        "Regression Model Benchmarks (Dataset 1 & 2)",
        "Clustering Model Benchmarks (Dataset 1 & 2)"
    ])

    with tab1:
        st.subheader("Classification Algorithm Benchmarks (Dynamic Computation)")
        clf_comp_df = pd.DataFrame([
            {
                "Model Algorithm": "WEKA J48 Decision Tree",
                "Accuracy (%)": f"{j48_eval['accuracy']:.2f}%",
                "Precision": f"{j48_eval['precision_weighted']:.3f}",
                "Recall": f"{j48_eval['recall_weighted']:.3f}",
                "F1-Score": f"{j48_eval['f1_weighted']:.3f}",
                "Correct Samples": f"{j48_eval['correct_count']} / {j48_eval['total_test_instances']}"
            },
            {
                "Model Algorithm": "Gaussian Naive Bayes",
                "Accuracy (%)": f"{nb_eval['accuracy']:.2f}%",
                "Precision": f"{nb_eval['precision']:.3f}",
                "Recall": f"{nb_eval['recall']:.3f}",
                "F1-Score": f"{nb_eval['f1']:.3f}",
                "Correct Samples": f"{nb_eval['correct_count']} / {nb_eval['total_test_instances']}"
            }
        ])
        st.dataframe(clf_comp_df, use_container_width=True)

    with tab2:
        st.subheader("Regression Model Benchmarks (Dynamic Computation)")
        reg_comp_df = pd.DataFrame([
            {
                "Dataset": "Dataset 1: Crop Yield Prediction",
                "Model Algorithm": reg1_res["model_name"],
                "Target Variable": reg1_res["target_col"],
                "R² Score": f"{reg1_res['r2']:.4f}",
                "RMSE": f"{reg1_res['rmse']:.4f}",
                "MAE": f"{reg1_res['mae']:.4f}",
                "MSE": f"{reg1_res['mse']:.4f}"
            },
            {
                "Dataset": "Dataset 2: Soil Organic Carbon",
                "Model Algorithm": reg2_res["model_name"],
                "Target Variable": reg2_res["target_col"],
                "R² Score": f"{reg2_res['r2']:.4f}",
                "RMSE": f"{reg2_res['rmse']:.4f}",
                "MAE": f"{reg2_res['mae']:.4f}",
                "MSE": f"{reg2_res['mse']:.4f}"
            }
        ])
        st.dataframe(reg_comp_df, use_container_width=True)

    with tab3:
        st.subheader("Clustering Model Benchmarks (Dynamic Computation)")
        clust_comp_df = pd.DataFrame([
            {
                "Dataset": "Dataset 1: Crop Microclimate Telemetry",
                "Clusters (k)": clust1_res["n_clusters"],
                "Silhouette Score": f"{clust1_res['silhouette_score']:.3f}",
                "Inertia (SSE)": f"{clust1_res['inertia']:,}",
                "Sample Count": len(clust1_res["clustered_df"])
            },
            {
                "Dataset": "Dataset 2: Soil & Water Quality",
                "Clusters (k)": clust2_res["n_clusters"],
                "Silhouette Score": f"{clust2_res['silhouette_score']:.3f}",
                "Inertia (SSE)": f"{clust2_res['inertia']:,}",
                "Sample Count": len(clust2_res["clustered_df"])
            }
        ])
        st.dataframe(clust_comp_df, use_container_width=True)
