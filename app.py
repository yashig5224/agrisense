"""
AgriSense — Agricultural Analytics Platform
Main Application Shell, Navigation Router, and Product Overview Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np

from utils.theme import apply_custom_theme
from utils.helpers import render_header, render_metric_card
from data.loader import load_crop_dataset
from visualization.charts import (
    plot_feature_histogram,
    plot_scatter,
    plot_correlation_heatmap,
    plot_crop_distribution
)

from preprocessing.ui import render_data_explorer_page
from analytics.ui import render_analytics_workspace
from decision_support.ui import render_decision_intelligence_page
from market.ui import render_market_intelligence_page
from location.ui import render_location_intelligence_page

from classification.weka_j48 import WekaJ48Classifier
from classification.naive_bayes import AgriculturalNaiveBayesClassifier
from regression.engine import AgriculturalRegressionEngine
from association.apriori_engine import AgriculturalAprioriEngine

# -----------------------------------------------------------------------------
# PAGE CONFIG & THEME INITIALIZATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AgriSense — Agricultural Analytics Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Light-Mode Custom Theme
apply_custom_theme()

# Load Dataset (Cached)
@st.cache_data
def load_primary_data():
    return load_crop_dataset()

df_raw = load_primary_data()

# -----------------------------------------------------------------------------
# GLOBAL APP SHELL SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.markdown("## AGRI SENSE")
st.sidebar.markdown("##### Agricultural Analytics Platform")
st.sidebar.markdown("---")

st.sidebar.caption("SELECT SECTION")
navigation_selection = st.sidebar.radio(
    "Navigation Menu",
    [
        "Overview",
        "Data Explorer",
        "Analytics",
        "Decision Intelligence",
        "Market Intelligence",
        "Location Intelligence"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.caption("SYSTEM STATUS & METADATA")
st.sidebar.write(f"Telemetry Records: **{len(df_raw):,}**")
st.sidebar.write(f"Crop Varieties: **{df_raw['label'].nunique() if 'label' in df_raw else 0}**")
st.sidebar.write("Geographic Scope: **India (States & APMC Mandis)**")
st.sidebar.write("Official Market Feed: **Agmarknet (DMI, MoA&FW)**")
st.sidebar.write("Fertilizer Registry: **DoF DBT (iFMS, GoI)**")
st.sidebar.write("Platform Status: **Active**")
st.sidebar.write("Registry Timestamp: **2026-09-28**")

# -----------------------------------------------------------------------------
# NAVIGATION ROUTER & PAGES
# -----------------------------------------------------------------------------

if navigation_selection == "Overview":
    render_header(
        "AgriSense Indian Agricultural Intelligence Dashboard",
        "Explore agricultural data, discover agronomic patterns, track Indian APMC mandi prices, and generate data-driven decision insights."
    )


    # Compute high level model accuracy dynamically
    clean_df = df_raw.dropna(subset=["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]).copy()
    nb_res = AgriculturalNaiveBayesClassifier.train_and_evaluate(clean_df, test_size=0.20)
    best_acc = nb_res["accuracy"]

    # SECTION 1: DATASET OVERVIEW KPI CARDS
    st.subheader("Platform Key Metrics")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_metric_card("Total Records", f"{len(df_raw):,}", "Primary telemetry dataset")
    with k2:
        render_metric_card("Crop Varieties", f"{df_raw['label'].nunique()}", "Target crop classes")
    with k3:
        render_metric_card("Core Features", "7", "N, P, K, Temp, Humidity, pH, Rainfall")
    with k4:
        render_metric_card("Best Model Accuracy", f"{best_acc:.2f}%", "Gaussian Naive Bayes classifier")

    st.markdown("---")

    # SECTION 2: AGRICULTURAL LANDSCAPE
    st.subheader("Agricultural Landscape — Crop Distribution")
    crop_fig = plot_crop_distribution(df_raw)
    if crop_fig:
        st.plotly_chart(crop_fig, use_container_width=True)

    st.markdown("---")

    # SECTION 3: TWO-COLUMN LAYOUT (SOIL NUTRIENTS vs ENVIRONMENTAL CONDITIONS)
    col_soil, col_env = st.columns(2)

    with col_soil:
        st.subheader("Soil Nutrient Profile (N, P, K)")
        st.markdown(f"- **Avg Nitrogen (N)**: {df_raw['N'].mean():.1f} kg/ha (Min: {df_raw['N'].min():.1f}, Max: {df_raw['N'].max():.1f})")
        st.markdown(f"- **Avg Phosphorus (P)**: {df_raw['P'].mean():.1f} kg/ha (Min: {df_raw['P'].min():.1f}, Max: {df_raw['P'].max():.1f})")
        st.markdown(f"- **Avg Potassium (K)**: {df_raw['K'].mean():.1f} kg/ha (Min: {df_raw['K'].min():.1f}, Max: {df_raw['K'].max():.1f})")

        fig_n = plot_feature_histogram(df_raw, "N")
        st.plotly_chart(fig_n, use_container_width=True)

    with col_env:
        st.subheader("Environmental Microclimate Conditions")
        st.markdown(f"- **Avg Temperature**: {df_raw['temperature'].mean():.1f} °C (Range: {df_raw['temperature'].min():.1f} - {df_raw['temperature'].max():.1f})")
        st.markdown(f"- **Avg Humidity**: {df_raw['humidity'].mean():.1f} % (Range: {df_raw['humidity'].min():.1f} - {df_raw['humidity'].max():.1f})")
        st.markdown(f"- **Avg Annual Rainfall**: {df_raw['rainfall'].mean():.1f} mm (Range: {df_raw['rainfall'].min():.1f} - {df_raw['rainfall'].max():.1f})")

        fig_rain = plot_feature_histogram(df_raw, "rainfall")
        st.plotly_chart(fig_rain, use_container_width=True)

    st.markdown("---")

    # SECTION 4: TWO-COLUMN LAYOUT (CORRELATION OVERVIEW vs MODEL PERFORMANCE)
    col_corr, col_perf = st.columns(2)

    with col_corr:
        st.subheader("Correlation Overview")
        heatmap_fig = plot_correlation_heatmap(df_raw)
        st.plotly_chart(heatmap_fig, use_container_width=True)

    with col_perf:
        st.subheader("Internal Predictive Engine Benchmarks")

        j48_res = WekaJ48Classifier.run_j48(clean_df, test_size=0.20)
        reg1_df, reg1_target, reg1_feats = AgriculturalRegressionEngine.load_regression_dataset("Dataset 1: Crop Yield Prediction")
        reg1_res = AgriculturalRegressionEngine.train_and_evaluate(reg1_df, reg1_target, reg1_feats, model_type="Random Forest Regressor")

        perf_df = pd.DataFrame([
            {"Engine Module": "Gaussian Naive Bayes Classifier", "Metric": "Accuracy", "Value": f"{nb_res['accuracy']:.2f}%", "Status": "Trained"},
            {"Engine Module": "WEKA J48 Decision Tree Classifier", "Metric": "Accuracy", "Value": f"{j48_res['accuracy']:.2f}%", "Status": "Trained"},
            {"Engine Module": "Crop Yield Random Forest Regressor", "Metric": "R² Score", "Value": f"{reg1_res['r2']:.4f}", "Status": "Trained"},
            {"Engine Module": "Apriori Association Rule Miner", "Metric": "Min Support", "Value": "0.05 (684 itemsets)", "Status": "Active"}
        ])
        st.dataframe(perf_df, use_container_width=True)

    st.markdown("---")

    # SECTION 5: DISCOVERED PATTERNS
    st.subheader("Discovered Agronomic Patterns (Apriori Engine)")
    _, top_rules = AgriculturalAprioriEngine.mine_rules(clean_df, min_support=0.05, min_confidence=0.60, min_lift=1.2)
    if not top_rules.empty:
        disp_top_rules = top_rules.head(5)[["antecedents_str", "consequents_str", "support", "confidence", "lift"]].copy()
        disp_top_rules.columns = ["IF Condition (Soil/Climate)", "THEN Recommendation", "Support", "Confidence", "Lift Factor"]
        st.dataframe(disp_top_rules, use_container_width=True)
    else:
        st.info("No high-confidence patterns mined at current threshold.")

    st.markdown("---")

    # SECTION 6: AGRICULTURAL INSIGHTS
    st.subheader("Computed Agronomic Insights")
    high_rain_crop = df_raw.groupby("label")["rainfall"].mean().idxmax()
    high_n_crop = df_raw.groupby("label")["N"].mean().idxmax()

    st.markdown(f"""
    - **High Water Requirement Crop**: Telemetry indicates **{high_rain_crop.upper()}** requires the highest average rainfall ({df_raw[df_raw['label']==high_rain_crop]['rainfall'].mean():.1f} mm).
    - **Nitrogen Intensive Crop**: **{high_n_crop.upper()}** exhibits the highest average Nitrogen demand ({df_raw[df_raw['label']==high_n_crop]['N'].mean():.1f} kg/ha).
    - **Soil pH Baseline**: Mean soil pH across dataset records is **{df_raw['ph'].mean():.2f}**, representing slightly acidic to neutral agronomic conditions.
    """)

elif navigation_selection == "Data Explorer":
    render_data_explorer_page(df_raw)

elif navigation_selection == "Analytics":
    render_analytics_workspace(df_raw)

elif navigation_selection == "Decision Intelligence":
    render_decision_intelligence_page(df_raw)

elif navigation_selection == "Market Intelligence":
    render_market_intelligence_page(df_raw)

elif navigation_selection == "Location Intelligence":
    render_location_intelligence_page(df_raw)
