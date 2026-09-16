"""
Chart generator functions using Plotly & Matplotlib styled in cohesive agricultural greens and earthy neutrals.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Agricultural Palette definition (Strict Light Mode, No Neons, No Purple, No Dark Mode)
COLOR_PRIMARY = "#2D5A27"     # Forest Green
COLOR_SECONDARY = "#739072"   # Sage Green
COLOR_ACCENT = "#4A5568"      # Slate / Earthy
COLOR_MUTED = "#8B9D83"       # Soft Muted Green
COLOR_LIGHT_BG = "#F4F6F4"    # Light Sage Tint
COLOR_BG = "#FFFFFF"          # Pure White
COLOR_GRID = "#E2E8F0"        # Subtle Border Gray
COLOR_PALETTE = ["#2D5A27", "#739072", "#4A5568", "#8B9D83", "#A2B29F", "#5C7658"]

def plot_feature_histogram(df: pd.DataFrame, feature: str):
    """Create styled Plotly distribution histogram for numeric feature."""
    fig = px.histogram(
        df,
        x=feature,
        nbins=30,
        marginal="rug",
        title=f"Distribution of {feature}",
        color_discrete_sequence=[COLOR_PRIMARY]
    )
    fig.update_layout(
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font=dict(family="Inter, sans-serif", color="#1A202C", size=12),
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(showgrid=True, gridcolor=COLOR_GRID, title=feature),
        yaxis=dict(showgrid=True, gridcolor=COLOR_GRID, title="Count"),
        height=380
    )
    return fig

def plot_box_plots(df: pd.DataFrame, numeric_cols: list):
    """Create box plots for outlier visualization across numeric features."""
    fig = go.Figure()
    for col in numeric_cols:
        if col in df.columns:
            fig.add_trace(go.Box(
                y=df[col].dropna(),
                name=col,
                marker_color=COLOR_PRIMARY,
                line_color=COLOR_SECONDARY,
                boxpoints="outliers"
            ))

    fig.update_layout(
        title="Numerical Features Outlier Analysis (Box Plots)",
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font=dict(family="Inter, sans-serif", color="#1A202C", size=12),
        margin=dict(l=40, r=40, t=50, b=40),
        yaxis=dict(showgrid=True, gridcolor=COLOR_GRID),
        height=400,
        showlegend=False
    )
    return fig

def plot_crop_distribution(df: pd.DataFrame):
    """Bar chart showing crop label counts in dataset."""
    if "label" not in df.columns:
        return None

    counts = df["label"].value_counts().reset_index()
    counts.columns = ["Crop", "Sample Count"]

    fig = px.bar(
        counts,
        x="Crop",
        y="Sample Count",
        color_discrete_sequence=[COLOR_PRIMARY],
        title="Crop Label Class Distribution"
    )

    fig.update_layout(
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font=dict(family="Inter, sans-serif", color="#1A202C", size=12),
        margin=dict(l=40, r=40, t=50, b=60),
        xaxis=dict(showgrid=False, title="Crop Label", tickangle=-45),
        yaxis=dict(showgrid=True, gridcolor=COLOR_GRID, title="Count"),
        height=420
    )
    return fig

def plot_correlation_heatmap(df: pd.DataFrame):
    """Correlation matrix heatmap using Plotly."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr().round(2)

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale=[
            [0.0, "#F4F6F4"],
            [0.5, "#8B9D83"],
            [1.0, "#1B3B18"]
        ],
        text=corr.values,
        texttemplate="%{text}",
        textfont={"size": 11, "family": "Inter, sans-serif"},
        showscale=True
    ))

    fig.update_layout(
        title="Pearson Correlation Heatmap (Numerical Features)",
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font=dict(family="Inter, sans-serif", color="#1A202C", size=12),
        margin=dict(l=60, r=40, t=50, b=60),
        height=450
    )
    return fig

def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None):
    """Scatter plot."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        color_discrete_sequence=COLOR_PALETTE,
        title=f"{y_col} vs {x_col}"
    )
    fig.update_layout(
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font=dict(family="Inter, sans-serif", color="#1A202C", size=12),
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(showgrid=True, gridcolor=COLOR_GRID, title=x_col),
        yaxis=dict(showgrid=True, gridcolor=COLOR_GRID, title=y_col),
        height=400
    )
    return fig

def plot_bar_chart(df: pd.DataFrame, category_col: str, value_col: str = None, title: str = ""):
    """Bar chart."""
    if value_col:
        grouped = df.groupby(category_col)[value_col].mean().reset_index()
        fig = px.bar(
            grouped,
            x=category_col,
            y=value_col,
            color_discrete_sequence=[COLOR_PRIMARY],
            title=title or f"Average {value_col} by {category_col}"
        )
    else:
        counts = df[category_col].value_counts().reset_index()
        counts.columns = [category_col, "Count"]
        fig = px.bar(
            counts,
            x=category_col,
            y="Count",
            color_discrete_sequence=[COLOR_SECONDARY],
            title=title or f"Count by {category_col}"
        )

    fig.update_layout(
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font=dict(family="Inter, sans-serif", color="#1A202C", size=12),
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(showgrid=False, title=category_col),
        yaxis=dict(showgrid=True, gridcolor=COLOR_GRID),
        height=380
    )
    return fig
