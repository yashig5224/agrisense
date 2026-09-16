"""
Clustering UI Module for AgriSense (Phase 7).
Supports Dataset 1 (Crop & Microclimate Telemetry) and Dataset 2 (Soil & Water Quality)
with K-Means clustering, configurable K (2-8), Inertia, Silhouette scores,
2D PCA visual projections, and un-scaled cluster profile tables.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import render_header, render_metric_card
from clustering.engine import AgriculturalClusteringEngine

def render_clustering_page(df_raw: pd.DataFrame):
    """Render Phase 7 Dual-Dataset K-Means Clustering dashboard."""
    render_header(
        "Agricultural Land & Microclimate Clustering Engine",
        "Segment agricultural land and soil telemetry into distinct agronomic clusters using unsupervised K-Means learning and PCA visualization."
    )

    st.markdown("### Dataset & Clustering Parameters")
    c1, c2, c3 = st.columns(3)

    with c1:
        dataset_choice = st.selectbox(
            "Select Agricultural Dataset",
            ["Dataset 1: Crop & Microclimate Telemetry", "Dataset 2: Soil & Water Quality"]
        )
    with c2:
        n_clusters = st.slider("Target Clusters (k)", min_value=2, max_value=8, value=4, step=1)
    with c3:
        random_state = st.number_input("Random State Seed", value=42, step=1)

    # Load chosen dataset
    clust_df, available_features = AgriculturalClusteringEngine.load_clustering_dataset(dataset_choice)

    st.markdown(f"**Selected Features**: {', '.join(available_features)} | **Total Dataset Records**: {len(clust_df):,}")

    # Execute K-Means clustering
    results = AgriculturalClusteringEngine.execute_kmeans(
        clust_df, feature_cols=available_features, n_clusters=n_clusters, random_state=random_state
    )

    st.markdown("---")

    # Metrics Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Silhouette Score", f"{results['silhouette_score']:.3f}", "Cluster separation quality")
    with m2:
        render_metric_card("Inertia (SSE)", f"{results['inertia']:,}", "Sum of squared errors")
    with m3:
        render_metric_card("Cluster Count (k)", f"{results['n_clusters']}", "Configured segments")
    with m4:
        render_metric_card("Dataset Samples", f"{len(results['clustered_df']):,}", "Records analyzed")

    st.markdown("---")

    # Tabs Layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "2D PCA Cluster Projection",
        "Cluster Profile Summary",
        "Cluster Interpretation Advisory",
        "Raw Clustered Dataset"
    ])

    with tab1:
        st.subheader("2D PCA Visualization Projection")
        st.info("Note: Principal Component Analysis (PCA) is utilized strictly for 2D visual representation of high-dimensional feature clusters.")

        var1, var2 = results["pca_variance_explained"]
        pca_df = results["pca_df"]

        pca_fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="Cluster_ID",
            title=f"K-Means Clusters (k={n_clusters}) — 2D PCA Space (PC1: {var1}%, PC2: {var2}% variance)",
            color_discrete_sequence=["#2D5A27", "#739072", "#4A5568", "#8B9D83", "#A2B29F", "#5C7658", "#3A4D39", "#A8BBA2"]
        )

        pca_fig.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(family="Inter, sans-serif", color="#1A202C"),
            xaxis=dict(showgrid=True, gridcolor="#E2E8F0", title=f"Principal Component 1 ({var1}% Variance)"),
            yaxis=dict(showgrid=True, gridcolor="#E2E8F0", title=f"Principal Component 2 ({var2}% Variance)"),
            height=450
        )
        st.plotly_chart(pca_fig, use_container_width=True)

    with tab2:
        st.subheader("Cluster Profile Means (Un-scaled)")
        st.caption("Average feature values for every cluster calculated from actual dataset records:")
        profile_df = results["cluster_profiles"]
        st.dataframe(profile_df, use_container_width=True)

    with tab3:
        st.subheader("Agronomic Cluster Interpretations")
        st.caption("Post-hoc interpretations calculated strictly from actual cluster centroid averages:")
        interpretations = AgriculturalClusteringEngine.generate_cluster_interpretations(profile_df, available_features)
        for interp in interpretations:
            st.markdown(f"- {interp}")

    with tab4:
        st.subheader("Dataset Records with Assigned Cluster IDs")
        st.dataframe(results["clustered_df"], use_container_width=True)
