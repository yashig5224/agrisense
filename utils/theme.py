"""
Theme and Styling Module for AgriSense
Enforces strict light mode design system with agricultural green, sage, and earthy neutrals.
"""

import streamlit as st

def apply_custom_theme():
    """Inject custom CSS rules for a clean, professional agricultural analytics interface."""
    css = """
    <style>
    /* Global Reset & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Merriweather:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #1A202C;
        background-color: #FFFFFF;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F4F6F4 !important;
        border-right: 1px solid #E2E8F0;
        padding-top: 1rem;
    }

    section[data-testid="stSidebar"] .stSelectbox label {
        color: #2D5A27 !important;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Titles and Headings */
    h1 {
        font-family: 'Merriweather', Georgia, serif;
        color: #1B3B18 !important;
        font-weight: 700 !important;
        font-size: 2.1rem !important;
        margin-bottom: 0.5rem !important;
        border-bottom: 2px solid #2D5A27;
        padding-bottom: 0.5rem;
    }

    h2 {
        font-family: 'Inter', sans-serif;
        color: #2D5A27 !important;
        font-weight: 600 !important;
        font-size: 1.4rem !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.75rem !important;
    }

    h3 {
        font-family: 'Inter', sans-serif;
        color: #3A4D39 !important;
        font-weight: 600 !important;
        font-size: 1.15rem !important;
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
    }

    p, li, label {
        color: #2D3748;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* AgriSense Metric Cards */
    .agri-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 4px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }

    .agri-metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #5A6B5C;
        margin-bottom: 0.35rem;
    }

    .agri-metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1B3B18;
        line-height: 1.2;
    }

    .agri-metric-subtitle {
        font-size: 0.8rem;
        color: #718096;
        margin-top: 0.35rem;
    }

    /* Section Header Block */
    .agri-header-box {
        background-color: #F8FAF8;
        border-left: 4px solid #2D5A27;
        border-right: 1px solid #E2E8F0;
        border-top: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        padding: 1rem 1.25rem;
        margin-bottom: 1.5rem;
        border-radius: 2px;
    }

    .agri-header-box h2 {
        margin: 0 !important;
        color: #1B3B18 !important;
    }

    .agri-header-box p {
        margin: 0.35rem 0 0 0;
        color: #4A5568;
        font-size: 0.9rem;
    }

    /* Buttons */
    .stButton > button {
        background-color: #2D5A27 !important;
        color: #FFFFFF !important;
        border: 1px solid #2D5A27 !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        padding: 0.4rem 1.2rem !important;
        transition: all 0.15s ease-in-out;
    }

    .stButton > button:hover {
        background-color: #1B3B18 !important;
        border-color: #1B3B18 !important;
        color: #FFFFFF !important;
    }

    /* Inputs, Selectboxes, Sliders */
    .stSelectbox > div > div, .stMultiSelect > div > div, .stTextInput > div > div {
        border-radius: 4px !important;
        border: 1px solid #CBD5E0 !important;
        background-color: #FFFFFF !important;
    }

    /* Dataframe Table styling */
    .stDataFrame {
        border: 1px solid #E2E8F0;
        border-radius: 4px;
    }

    /* Radio buttons & Checkboxes */
    .stRadio label, .stCheckbox label {
        font-size: 0.9rem !important;
        color: #2D3748 !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        border-bottom: 1px solid #E2E8F0;
    }

    .stTabs [data-baseweb="tab"] {
        height: 2.5rem;
        white-space: pre;
        border-radius: 0;
        color: #4A5568;
        font-weight: 500;
        font-size: 0.9rem;
        padding: 0 0.5rem;
        border-bottom: 2px solid transparent;
    }

    .stTabs [aria-selected="true"] {
        color: #2D5A27 !important;
        border-bottom: 2px solid #2D5A27 !important;
        font-weight: 600;
        background-color: transparent !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
