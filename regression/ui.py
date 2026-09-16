"""
Regression UI Module for AgriSense (Phase 6).
Supports Dataset 1 (Crop Yield Prediction) and Dataset 2 (Soil Organic Carbon Content)
with interactive evaluation metrics (MAE, MSE, RMSE, R²), actual vs predicted plots,
residual analysis, feature relationship scatter plots, and error distribution charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import render_header, render_metric_card
from regression.engine import AgriculturalRegressionEngine

def render_regression_page(df_raw: pd.DataFrame):
    """Render Phase 6 Dual-Dataset Regression dashboard."""
    render_header(
        "Agricultural Regression Modeling & Prediction",
        "Predict continuous agricultural outcomes (Crop Yield and Soil Organic Carbon) using ensemble and linear regression models."
    )

    st.markdown("### Dataset & Model Selection")
    c1, c2, c3 = st.columns(3)

    with c1:
        dataset_choice = st.selectbox(
            "Select Agricultural Dataset",
            ["Dataset 1: Crop Yield Prediction", "Dataset 2: Soil Organic Carbon Content"]
        )
    with c2:
        model_choice = st.selectbox(
            "Select Regressor Model",
            ["Random Forest Regressor", "Gradient Boosting Regressor", "Linear Regression", "Ridge Regression"]
        )
    with c3:
        test_ratio = st.slider("Test Split Ratio (%)", min_value=10, max_value=40, value=20, step=5)

    # Load chosen dataset
    reg_df, target_col, available_features = AgriculturalRegressionEngine.load_regression_dataset(dataset_choice)

    st.markdown(f"**Target Variable**: `{target_col}` | **Dataset Rows**: {len(reg_df):,} | **Features**: {', '.join(available_features)}")

    # Execute regression engine
    results = AgriculturalRegressionEngine.train_and_evaluate(
        reg_df, target_col=target_col, feature_cols=available_features,
        model_type=model_choice, test_size=test_ratio/100.0
    )

    st.markdown("---")

    # Metrics Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("R-Squared (R²)", f"{results['r2']:.4f}", "Variance explained")
    with m2:
        render_metric_card("RMSE", f"{results['rmse']:.4f}", f"Units of {target_col}")
    with m3:
        render_metric_card("MAE", f"{results['mae']:.4f}", "Mean abs error")
    with m4:
        render_metric_card("MSE", f"{results['mse']:.4f}", "Mean squared error")

    st.markdown("---")

    # Tabs Layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "Actual vs. Predicted Plot",
        "Residual Analysis",
        "Feature Relationships",
        "Predictions & Error Table"
    ])

    eval_df = results["eval_df"]

    with tab1:
        st.subheader("Actual vs. Predicted Outcomes")
        fig_actual_pred = px.scatter(
            eval_df,
            x="Actual",
            y="Predicted",
            hover_data=["Residual"],
            title=f"{target_col} — Actual vs. Predicted Target",
            color_discrete_sequence=["#2D5A27"]
        )

        # Add y=x ideal prediction reference line
        min_val = min(eval_df["Actual"].min(), eval_df["Predicted"].min())
        max_val = max(eval_df["Actual"].max(), eval_df["Predicted"].max())

        fig_actual_pred.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Ideal Prediction (y=x)",
            line=dict(color="#739072", dash="dash", width=2)
        ))

        fig_actual_pred.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(family="Inter, sans-serif", color="#1A202C"),
            xaxis=dict(showgrid=True, gridcolor="#E2E8F0", title=f"Actual {target_col}"),
            yaxis=dict(showgrid=True, gridcolor="#E2E8F0", title=f"Predicted {target_col}"),
            height=450
        )
        st.plotly_chart(fig_actual_pred, use_container_width=True)

    with tab2:
        st.subheader("Residual Error Analysis")
        r_col1, r_col2 = st.columns(2)

        with r_col1:
            st.markdown("### Residuals vs. Predicted Values")
            fig_residuals = px.scatter(
                eval_df,
                x="Predicted",
                y="Residual",
                title="Residual Errors Scatter Plot",
                color_discrete_sequence=["#4A5568"]
            )
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="#2D5A27")
            fig_residuals.update_layout(
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(family="Inter, sans-serif", color="#1A202C"),
                xaxis=dict(showgrid=True, gridcolor="#E2E8F0", title="Predicted Value"),
                yaxis=dict(showgrid=True, gridcolor="#E2E8F0", title="Residual (Actual - Predicted)"),
                height=400
            )
            st.plotly_chart(fig_residuals, use_container_width=True)

        with r_col2:
            st.markdown("### Residual Error Distribution Histogram")
            fig_err_hist = px.histogram(
                eval_df,
                x="Residual",
                nbins=25,
                title="Residual Error Distribution",
                color_discrete_sequence=["#739072"]
            )
            fig_err_hist.update_layout(
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(family="Inter, sans-serif", color="#1A202C"),
                xaxis=dict(showgrid=True, gridcolor="#E2E8F0", title="Residual Error"),
                yaxis=dict(showgrid=True, gridcolor="#E2E8F0", title="Frequency"),
                height=400
            )
            st.plotly_chart(fig_err_hist, use_container_width=True)

    with tab3:
        st.subheader("Feature Relationship Scatter Inspector")
        sel_feat = st.selectbox("Select Feature to Plot Against Target", available_features)
        fig_feat = px.scatter(
            reg_df,
            x=sel_feat,
            y=target_col,
            title=f"{target_col} vs. {sel_feat}",
            color_discrete_sequence=["#2D5A27"]
        )
        fig_feat.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(family="Inter, sans-serif", color="#1A202C"),
            xaxis=dict(showgrid=True, gridcolor="#E2E8F0", title=sel_feat),
            yaxis=dict(showgrid=True, gridcolor="#E2E8F0", title=target_col),
            height=400
        )
        st.plotly_chart(fig_feat, use_container_width=True)

    with tab4:
        st.subheader("Test Dataset Predictions Table")
        st.dataframe(eval_df, use_container_width=True)
