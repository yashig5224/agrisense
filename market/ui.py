"""
Market Intelligence UI Module for AgriSense (Version 2).
Provides official Indian APMC Mandi commodity market prices (Agmarknet / DMI),
certified seed variety directory (NSC / ICAR), statutory fertilizer prices (DoF),
authorized district fertilizer dealers (DBT portal), and agricultural input-cost budgeting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import render_header, render_metric_card
from market.provider import MarketDataProvider
from market.engine import AgriculturalMarketEngine
from data.india.provider import IndianAgriculturalDataProvider

def render_market_intelligence_page(df_crop_raw: pd.DataFrame):
    """Render Version 2 Agricultural Market Intelligence dashboard backed by Indian official datasets."""
    render_header(
        "Indian Agricultural Market Intelligence & Input-Cost Budgeting",
        "Track official APMC mandi prices (Agmarknet), certified seed varieties (NSC/ICAR), statutory fertilizer pricing (DoF), and district retailer networks across India."
    )

    # Central Data Provenance Banner
    st.info("""
    **OFFICIAL INDIAN MARKET DATA REGISTRY & PROVENANCE**:
    - **APMC Mandi Commodity Prices**: Directorate of Marketing & Inspection (DMI) / Agmarknet, Ministry of Agriculture & Farmers Welfare, GoI.
    - **Fertilizer Statutory Pricing**: Department of Fertilizers, Ministry of Chemicals & Fertilizers (Uniform Statutory MRP for Urea; NBS subsidized complex fertilizers).
    - **Retail Dealers**: Authorized Point-of-Sale (POS) District Fertilizer Retailers & Primary Agricultural Credit Societies (PACS), DBT Portal (iFMS).
    - **Certified Seeds**: National Seeds Corporation (NSC) & ICAR Crop Science Division.
    - **Currency & Units**: Indian Rupee (INR) only | Modal Prices in ₹/quintal | Fertilizer in ₹/50kg bag.
    """)

    # Main Market Intelligence Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Indian Mandi Prices (Agmarknet)",
        "Certified Seed Directory (NSC / ICAR)",
        "Statutory Fertilizer Prices (DoF)",
        "District Fertilizer Dealers (DBT)",
        "Input-Cost Budgeting Calculator",
        "Central Data Provenance Registry"
    ])

    with tab1:
        st.subheader("Official APMC Mandi Commodity Prices (Agmarknet)")
        st.caption("Price observations from regulated Agricultural Produce Market Committees (APMCs) across Indian agricultural states:")

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            states_list = IndianAgriculturalDataProvider.get_states()
            sel_state = st.selectbox("Select State", ["All States"] + states_list, index=1, key="mkt_state")
        with m_col2:
            if sel_state != "All States":
                dist_list = IndianAgriculturalDataProvider.get_districts(sel_state)
                sel_dist = st.selectbox("Select District", ["All Districts"] + dist_list, index=1, key="mkt_dist")
            else:
                sel_dist = "All Districts"
                st.selectbox("Select District", ["All Districts"], disabled=True, key="mkt_dist_dis")
        with m_col3:
            crops = sorted(df_crop_raw["label"].dropna().unique().tolist()) if "label" in df_crop_raw else ["rice", "maize", "cotton", "wheat"]
            sel_crop = st.selectbox("Select Commodity Crop", crops, index=0, key="mkt_crop")

        state_query = None if sel_state == "All States" else sel_state
        dist_query = None if sel_dist == "All Districts" else sel_dist
        crop_info = IndianAgriculturalDataProvider.get_crop_mandi_info(sel_crop, state=state_query, district=dist_query)

        if crop_info.get("status") == "SUCCESS":
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                render_metric_card("Modal Mandi Price", f"₹ {crop_info['latest_price']:,.2f}", f"per Quintal ({crop_info['market_apmc']})")
            with m2:
                render_metric_card("Price Trend", f"{crop_info['price_trend']}", "Short-term mandi trajectory")
            with m3:
                render_metric_card("Minimum Mandi Price", f"₹ {crop_info['historical_min']:,.2f}", "Lowest recorded lot price")
            with m4:
                render_metric_card("Maximum Mandi Price", f"₹ {crop_info['historical_max']:,.2f}", "Peak recorded lot price")

            st.markdown("---")
            p_c1, p_c2 = st.columns([1, 1])

            with p_c1:
                st.markdown("### Mandi Transaction Details")
                st.write(f"- **Commodity & Variety**: **{crop_info['commodity']}** ({crop_info['variety']})")
                st.write(f"- **Regulated Market / APMC**: **{crop_info['market_apmc']}**")
                st.write(f"- **Geography**: **{crop_info['district']}, {crop_info['state']}**")
                st.write(f"- **Arrival / Reference Date**: **{crop_info['arrival_date']}**")
                st.write(f"- **Official Data Source**: `{crop_info['data_source']}`")
                st.caption(f"Provenance: {crop_info['provenance']}")

            with p_c2:
                st.markdown("### Price Range Visualization (INR / Quintal)")
                fig_range = go.Figure()
                fig_range.add_trace(go.Bar(
                    x=["Minimum Price", "Modal Price (Benchmark)", "Maximum Price"],
                    y=[crop_info['historical_min'], crop_info['latest_price'], crop_info['historical_max']],
                    marker_color=["#8B9D83", "#2D5A27", "#739072"],
                    text=[f"₹{crop_info['historical_min']:,.0f}", f"₹{crop_info['latest_price']:,.0f}", f"₹{crop_info['historical_max']:,.0f}"],
                    textposition="auto"
                ))
                fig_range.update_layout(
                    paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                    font=dict(family="Inter, sans-serif", color="#1A202C"),
                    yaxis=dict(showgrid=True, gridcolor="#E2E8F0", title="Price (₹ / Quintal)"),
                    height=280
                )
                st.plotly_chart(fig_range, use_container_width=True)
        else:
            st.warning(crop_info.get("message", "Indian market data unavailable for this selection."))

    with tab2:
        st.subheader("Indian Certified Seed Varieties Directory (NSC / ICAR)")
        st.caption("Notified and released seed varieties recognized under the Indian Seeds Act and National Seed Programme:")

        sel_seed_crop = st.selectbox("Select Crop for Seed Varieties", crops, key="seed_crop_sel")
        seed_df = IndianAgriculturalDataProvider.get_seed_varieties(sel_seed_crop)

        if not seed_df.empty:
            st.dataframe(seed_df, use_container_width=True)
            st.caption("Source: National Seeds Corporation (NSC) & ICAR Crop Science Division.")
        else:
            st.info(f"Indian certified seed variety records for '{sel_seed_crop}' currently unlisted in registry.")

    with tab3:
        st.subheader("Official Indian Fertilizer Statutory Pricing & NBS Subsidy (DoF)")
        st.caption("Maximum Retail Prices (MRP) administered by the Government of India under the Nutrient Based Subsidy (NBS) Scheme:")

        fert_df = IndianAgriculturalDataProvider.get_fertilizers_df()
        if not fert_df.empty:
            st.dataframe(fert_df, use_container_width=True)
            st.caption("Source: Department of Fertilizers, Ministry of Chemicals & Fertilizers, Government of India.")
        else:
            st.info("Fertilizer pricing registry currently unavailable.")

    with tab4:
        st.subheader("District Retail Fertilizer Dealers & PACS Directory (DoF / DBT)")
        st.caption("Authorized Point-of-Sale (POS) retailers and Primary Agricultural Credit Societies licensed under Fertilizer Control Order (FCO):")

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            dealer_state = st.selectbox("Select State", states_list, index=0, key="dlr_state")
        with d_col2:
            dealer_dist_list = IndianAgriculturalDataProvider.get_districts(dealer_state)
            dealer_dist = st.selectbox("Select District", dealer_dist_list, index=0, key="dlr_dist")

        dealers_df, count = IndianAgriculturalDataProvider.get_retail_dealers(state=dealer_state, district=dealer_dist)

        m_c1, m_c2 = st.columns(2)
        with m_c1:
            render_metric_card("Licensed Dealers in District", f"{count}", f"District: {dealer_dist}, {dealer_state}")
        with m_c2:
            render_metric_card("Data Transparency", "District-level dealer information", "Source: DoF iFMS DBT Portal")

        st.markdown("### Authorized Retail Network")
        if not dealers_df.empty:
            st.dataframe(dealers_df, use_container_width=True)
        else:
            st.info(f"No retailer records registered for {dealer_dist} in current directory.")

    with tab5:
        st.subheader("Agricultural Input-Cost Budgeting Calculator (INR)")
        st.caption("Estimate farm production input costs based on Indian statutory fertilizer rates and certified seed prices:")

        calc_c1, calc_c2, calc_c3 = st.columns(3)
        with calc_c1:
            land_area = st.slider("Farm Land Area (Hectares)", min_value=0.5, max_value=50.0, value=2.5, step=0.5)
            sel_calc_crop = st.selectbox("Target Crop", crops, key="calc_crop")
        with calc_c2:
            seed_qty = st.number_input("Seed Rate (kg / Hectare)", min_value=1.0, max_value=100.0, value=25.0, step=1.0)
            seed_price = st.number_input("Certified Seed Price (₹ / kg)", min_value=10.0, max_value=1000.0, value=85.0, step=5.0)
        with calc_c3:
            fert_qty = st.number_input("Fertilizer Rate (kg / Hectare)", min_value=10.0, max_value=500.0, value=150.0, step=10.0)
            fert_bag_price = st.number_input("Subsidized Fertilizer Price (₹ / 50kg bag)", min_value=100.0, max_value=3000.0, value=1350.0, step=50.0)

        ops_cost_rate = st.number_input("Operational Expenses (₹ / Hectare - Labor, Land Preparation, Irrigation, Fuel)", min_value=0.0, max_value=25000.0, value=4000.0, step=500.0)

        cost_results = AgriculturalMarketEngine.calculate_input_costs(
            land_area_ha=land_area,
            seed_qty_kg_ha=seed_qty,
            seed_price_per_kg=seed_price,
            fert_qty_kg_ha=fert_qty,
            fert_price_per_bag_50kg=fert_bag_price,
            additional_ops_cost_per_ha=ops_cost_rate
        )

        st.markdown("---")
        st.markdown("### Estimated Cost Breakdown (INR)")

        b_c1, b_c2, b_c3, b_c4 = st.columns(4)
        with b_c1:
            render_metric_card("Total Seed Cost", f"₹ {cost_results['seed_cost']:,.2f}", f"for {land_area} Ha")
        with b_c2:
            render_metric_card("Total Fertilizer Cost", f"₹ {cost_results['fertilizer_cost']:,.2f}", f"for {land_area} Ha")
        with b_c3:
            render_metric_card("Operational Expenses", f"₹ {cost_results['operational_cost']:,.2f}", f"at ₹ {ops_cost_rate:,.0f}/Ha")
        with b_c4:
            render_metric_card("Total Estimated Input Cost", f"₹ {cost_results['total_estimated_input_cost']:,.2f}", f"₹ {cost_results['estimated_cost_per_hectare']:,.2f} / Ha")

        st.info(f"**BUDGETING DISCLAIMER**: {cost_results['disclaimer']}")

    with tab6:
        st.subheader("Central Data Provenance & Official Source Traceability")
        st.caption("Every agricultural data point in AgriSense is traced to its official Indian government source and geographic jurisdiction:")

        prov_table = IndianAgriculturalDataProvider.get_provenance_summary()
        st.dataframe(prov_table, use_container_width=True)
