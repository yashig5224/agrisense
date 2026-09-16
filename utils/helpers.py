"""
Helper utilities for UI rendering, dataset stats, and metric displays.
"""

import streamlit as st
import pandas as pd

def render_header(title: str, subtitle: str):
    """Render standardized section header box."""
    html = f"""
    <div class="agri-header-box">
        <h2>{title}</h2>
        <p>{subtitle}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_metric_card(title: str, value: str, subtitle: str = ""):
    """Render clean metric card strictly without emojis or badges."""
    html = f"""
    <div class="agri-card">
        <div class="agri-metric-title">{title}</div>
        <div class="agri-metric-value">{value}</div>
        <div class="agri-metric-subtitle">{subtitle}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def display_dataset_summary(df: pd.DataFrame):
    """Render high-level dataset summary metrics."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Samples", f"{df.shape[0]:,}", "Record count")
    with col2:
        render_metric_card("Attributes", f"{df.shape[1]}", "Feature columns")
    with col3:
        render_metric_card("Missing Values", f"{df.isna().sum().sum():,}", "Null cell count")
    with col4:
        render_metric_card("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB", "RAM memory size")
