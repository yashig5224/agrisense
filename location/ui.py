"""
Location Intelligence UI Module for AgriSense (Version 3).
Renders India-specific Location Intelligence dashboard with hierarchical workflow:
INDIA -> STATE -> DISTRICT -> LOCAL AGRICULTURAL CONDITIONS -> CROP CANDIDATES ->
CERTIFIED SEED -> FERTILIZER INFORMATION -> LOCAL DEALERS -> APMC MANDI PRICES -> INPUT COST & GROSS RETURN ESTIMATION.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.helpers import render_header, render_metric_card
from location.engine import LocationIntelligenceEngine
from data.india.provider import IndianAgriculturalDataProvider

def render_location_intelligence_page(df_crop_raw: pd.DataFrame):
    """Render Version 3 Indian Location Intelligence Dashboard."""
    render_header(
        "Indian Location Intelligence & Agricultural Decision Framework",
        "Synthesize Indian state/district geography, Soil Health Card data, ML crop recommendations, certified seeds, statutory fertilizers, and Agmarknet mandi returns."
    )

    # Provenance Disclaimer Banner
    st.info("""
    **DATA INTEGRITY & INDIAN PROVENANCE NOTICE**:
    - **Geographic Hierarchy**: India → State → District → Regulated Market / APMC.
    - **Soil & Telemetry Profiles**: Linked to Indian District Soil Health Card averages and ICAR Agro-Climatic Zones.
    - **Prices & Dealers**: Derived from Agmarknet (DMI) and Department of Fertilizers (DoF) DBT POS network.
    - **Economic Metrics**: Formatted strictly in INR (₹/quintal, ₹/ha) with full cost-breakdown transparency.
    """)

    st.markdown("### 1. Indian Geographic Location Hierarchy")

    loc_c1, loc_c2, loc_c3 = st.columns(3)

    with loc_c1:
        states = IndianAgriculturalDataProvider.get_states()
        sel_state = st.selectbox("Select Indian State", states, index=0, key="loc_state")
    with loc_c2:
        districts = IndianAgriculturalDataProvider.get_districts(sel_state)
        sel_district = st.selectbox("Select District", districts, index=0, key="loc_district")
    with loc_c3:
        season = st.selectbox("Agricultural Season", ["Kharif / Monsoon (June - Oct)", "Rabi / Winter (Oct - March)", "Zaid / Summer (March - June)"])

    # Load district agro-climatic context
    dist_info = IndianAgriculturalDataProvider.get_district_info(sel_state, sel_district)
    soil_profile = IndianAgriculturalDataProvider.get_soil_profile(sel_state, sel_district)

    # Context Card
    with st.container():
        st.markdown(f"**District Agro-Climatic Context**: `{dist_info['Agro_Climatic_Zone']}` | **Major Soil**: `{dist_info['Major_Soil']}`")
        if soil_profile:
            st.caption(f"Soil Health Card Benchmark ({sel_district}): Avg Available N: {soil_profile['Avg_Available_N_kg_ha']} kg/ha ({soil_profile['N_Status']}) | P: {soil_profile['Avg_Available_P_kg_ha']} kg/ha | K: {soil_profile['Avg_Available_K_kg_ha']} kg/ha | Mean pH: {soil_profile['Mean_pH']} ({soil_profile['pH_Reaction']})")

    st.markdown("---")
    st.markdown("### 2. Field Telemetry & Soil Nutrient Conditions")

    # Set default values based on Soil Health Card profile if available
    def_n = float(soil_profile['Avg_Available_N_kg_ha']) if soil_profile else 90.0
    def_p = float(soil_profile['Avg_Available_P_kg_ha']) if soil_profile else 42.0
    def_k = float(soil_profile['Avg_Available_K_kg_ha']) if soil_profile else 43.0
    def_ph = float(soil_profile['Mean_pH']) if soil_profile else 6.5

    tel_c1, tel_c2, tel_c3 = st.columns(3)

    with tel_c1:
        in_n = st.number_input("Nitrogen (N kg/ha)", 0.0, 300.0, def_n, 5.0)
        in_p = st.number_input("Phosphorus (P kg/ha)", 0.0, 300.0, def_p, 5.0)
        in_k = st.number_input("Potassium (K kg/ha)", 0.0, 400.0, def_k, 5.0)

    with tel_c2:
        in_temp = st.number_input("Average Temperature (°C)", 0.0, 50.0, 24.5, 0.5)
        in_hum = st.number_input("Relative Humidity (%)", 0.0, 100.0, 78.0, 1.0)
        in_ph = st.number_input("Soil pH Level", 0.0, 14.0, def_ph, 0.1)

    with tel_c3:
        in_rain = st.number_input("Annual / Seasonal Rainfall (mm)", 0.0, 1200.0, 195.0, 5.0)
        water_avail = st.selectbox("Irrigation / Water Availability", ["Canal / Well Irrigated", "Rainfed / Moderate Water", "Arid / Water Stressed"])

    st.markdown("---")
    st.markdown("### 3. Farm Management & Economic Parameters")
    f_c1, f_c2, f_c3, f_c4 = st.columns(4)

    with f_c1:
        land_area = st.slider("Land Area (Hectares)", 0.5, 50.0, 2.5, 0.5)
    with f_c2:
        seed_rate = st.number_input("Seed Rate (kg/Ha)", 1.0, 100.0, 25.0, 1.0)
    with f_c3:
        fert_rate = st.number_input("Fertilizer Rate (kg/Ha)", 10.0, 500.0, 150.0, 10.0)
    with f_c4:
        ops_rate = st.number_input("Operational Cost (₹ / Ha - Labor, Fuel, Tillage)", 0.0, 25000.0, 4000.0, 500.0)

    soil_inputs = {
        "N": in_n, "P": in_p, "K": in_k,
        "temperature": in_temp, "humidity": in_hum,
        "ph": in_ph, "rainfall": in_rain
    }

    farm_inputs = {
        "land_area_ha": land_area,
        "seed_qty_kg_ha": seed_rate,
        "fert_qty_kg_ha": fert_rate,
        "operational_rate_per_ha": ops_rate,
        "season": season,
        "water_availability": water_avail
    }

    # Execute Indian Location Intelligence Report
    report = LocationIntelligenceEngine.generate_location_report(
        df_crop_raw, state=sel_state, district=sel_district, soil_inputs=soil_inputs, farm_inputs=farm_inputs
    )

    st.markdown("---")

    # Workflow Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1. Crop Candidates (ML Inference)",
        "2. Certified Seeds (NSC / ICAR)",
        "3. Recommended Fertilizer (DoF NBS)",
        "4. District Fertilizer Retailers (DBT)",
        "5. Agmarknet APMC Mandi Prices",
        "6. Input Cost & Gross Return Model"
    ])

    with tab1:
        st.subheader("1. Suitable Crop Candidates & Dual-Model Predictions")
        st.caption(f"Location: **{sel_district}, {sel_state}** | Model Training: 22 Major Crops Cultivated Across India")

        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            render_metric_card("Gaussian Naive Bayes Recommendation", f"{report['recommended_crop']}", f"Confidence: {report['crop_confidence_pct']}%")
        with c_m2:
            render_metric_card("WEKA J48 Candidate", f"{report['j48_crop']}", "Decision tree inference")
        with c_m3:
            render_metric_card("Estimated Yield Potential", f"{report['economic_model']['yield_tons_ha']} Tonnes / Ha", "Random Forest Regressor")

        st.markdown("### Indian Agricultural Context & Official District Statistics")
        dist_stats = report["district_crop_stats"]
        if not dist_stats.empty:
            st.markdown(f"**Official Historical Statistics ({sel_district}, {sel_state})**: Directorate of Economics & Statistics (DES)")
            st.dataframe(dist_stats[["Crop", "Crop_Category", "Season", "Area_Hectares", "Production_Tonnes", "Yield_Tonnes_Ha", "Reference_Year"]], use_container_width=True)
        else:
            st.info(f"Official DES district historical crop database record for '{report['recommended_crop']}' is currently unlisted for {sel_district}; model prediction is based on agro-climatic telemetry.")

        st.caption("Provenance: Source: AgriSense Dual Classification Engine (Gaussian Naive Bayes & WEKA J48) | Training Data: Multi-Crop Agricultural Benchmark")

    with tab2:
        st.subheader(f"2. Indian Certified Seed Varieties for {report['recommended_crop']}")
        st.caption("Data Provenance: `Source: National Seeds Corporation (NSC) & ICAR Crop Science Division`")
        seed_info = report["seed_info"]

        s_m1, s_m2, s_m3 = st.columns(3)
        with s_m1:
            render_metric_card("Certified Seed Variety", f"{seed_info['name']}", f"Agency: {seed_info['agency']}")
        with s_m2:
            render_metric_card("Nominal Seed Rate", f"₹ {seed_info['price_per_kg']:,.2f} / kg", "Certified seed benchmark")
        with s_m3:
            render_metric_card("Availability Status", f"{seed_info['availability']}", "Government Seed Distribution")

        st.write(f"- **Agronomic Seed Notes**: {seed_info['notes']}")
        st.caption(seed_info["provenance"])

    with tab3:
        st.subheader("3. Fertilizer Recommendation & Statutory Price (DoF)")
        st.caption("Data Provenance: `Source: Department of Fertilizers, Ministry of Chemicals & Fertilizers, Government of India`")
        fert_info = report["fertilizer_info"]
        fert_rec = fert_info["recommendation"]

        if fert_rec.get("status") == "SUCCESS":
            f_m1, f_m2, f_m3 = st.columns(3)
            with f_m1:
                render_metric_card("Fertilizer Product", f"{fert_rec['recommended_fertilizer']}", f"Primary Deficit: {fert_rec['primary_deficiency']}")
            with f_m2:
                render_metric_card("Recommended Dosage", f"{fert_rec['recommended_dose_kg_ha']} kg/ha", "Nutrient deficit balancing")
            with f_m3:
                render_metric_card("Statutory MRP (DoF)", f"₹ {fert_info['bag_price_50kg']:,.2f}", f"per 50 kg bag ({fert_info['subsidy_note']})")

            st.markdown("### Supporting Data Mining Pattern & Agronomic Rationale")
            st.code(fert_rec["supporting_pattern"], language="text")
            st.markdown(fert_rec["explanation"])
            st.caption(fert_info["provenance"])
        else:
            st.warning("Insufficient historical fertilizer pattern data for this selection.")

    with tab4:
        st.subheader(f"4. Authorized Fertilizer Retailers & PACS in {sel_district} ({sel_state})")
        st.caption("Data Provenance: `Source: Department of Fertilizers (DoF), DBT in Fertilizers Portal (iFMS)`")
        dealers_info = report["dealers_info"]
        dealers_df = dealers_info["df"]

        render_metric_card("Authorized District Retailers / PACS", f"{dealers_info['dealer_count']}", f"District: {sel_district}, {sel_state}")

        st.markdown("### District Authorized Retail Dealer Registry")
        st.info("Note: Fertilizer dealer records represent authorized Point-of-Sale (POS) cooperative societies and licensed retailers in the district. GPS-level proximity is not implied.")

        if not dealers_df.empty:
            st.dataframe(dealers_df, use_container_width=True)
        else:
            st.info(f"No retailer records registered for {sel_district} in current directory.")

        st.caption(dealers_info["provenance"])

    with tab5:
        st.subheader(f"5. Agmarknet APMC Mandi Market Prices for {report['recommended_crop']}")
        st.caption("Data Provenance: `Source: Directorate of Marketing & Inspection (DMI) / Agmarknet, Ministry of Agriculture & Farmers Welfare, GoI`")
        mkt_info = report["market_price_info"]

        if mkt_info.get("status") == "SUCCESS":
            p_m1, p_m2, p_m3 = st.columns(3)
            with p_m1:
                render_metric_card("Modal Mandi Price", f"₹ {mkt_info['latest_price']:,.2f}", f"per Quintal ({mkt_info['market_apmc']})")
            with p_m2:
                render_metric_card("Market Trend", f"{mkt_info['price_trend']}", "Short-term mandi trajectory")
            with p_m3:
                render_metric_card("Recorded Range", f"₹ {mkt_info['historical_min']} - {mkt_info['historical_max']}", "Min to Max (₹/quintal)")

            st.write(f"- **Regulated Market / APMC**: **{mkt_info['market_apmc']}** ({mkt_info['district']}, {mkt_info['state']})")
            st.write(f"- **Arrival / Reference Date**: **{mkt_info['arrival_date']}** | **Variety**: **{mkt_info['variety']}**")
            st.caption(mkt_info["provenance"])
        else:
            st.warning("Indian market data unavailable: No active Agmarknet APMC mandi price record found for this crop in the district.")

    with tab6:
        st.subheader("6. Input Cost & Estimated Gross Return Budgeting Model (INR)")
        st.caption("Data Provenance: `Source: Yield Regressor & Agmarknet APMC Mandi Registry`")
        econ = report["economic_model"]

        r_m1, r_m2, r_m3 = st.columns(3)
        with r_m1:
            render_metric_card("Total Estimated Input Cost", f"₹ {econ['total_input_cost']:,.2f}", f"for {land_area} Hectares")
        with r_m2:
            render_metric_card("Estimated Gross Revenue", f"₹ {econ['estimated_gross_revenue']:,.2f}" if econ['return_status'] == "SUCCESS" else "Data unavailable", f"Yield: {econ['total_yield_quintals']} Quintals")
        with r_m3:
            render_metric_card("Estimated Gross Return", f"₹ {econ['estimated_gross_return']:,.2f}" if econ['return_status'] == "SUCCESS" else "Data unavailable", "Gross Revenue - Input Cost")

        st.markdown("---")
        st.markdown("### Farm Budget Breakdown Summary (INR)")

        breakdown_df = pd.DataFrame([
            {"Budget Item": "Estimated Seed Cost", "Amount (INR)": f"₹ {econ['seed_cost']:,.2f}", "Basis": f"{seed_rate} kg/ha @ ₹ {seed_info['price_per_kg']:.2f}/kg ({land_area} Ha)"},
            {"Budget Item": "Estimated Fertilizer Cost", "Amount (INR)": f"₹ {econ['fertilizer_cost']:,.2f}", "Basis": f"{fert_rate} kg/ha @ ₹ {fert_info['bag_price_50kg']:.2f}/50kg bag ({land_area} Ha)"},
            {"Budget Item": "Operational Expenses", "Amount (INR)": f"₹ {econ['operational_cost']:,.2f}", "Basis": f"Labor, tillage & irrigation @ ₹ {ops_rate:,.0f}/ha ({land_area} Ha)"},
            {"Budget Item": "TOTAL ESTIMATED INPUT COST", "Amount (INR)": f"₹ {econ['total_input_cost']:,.2f}", "Basis": "Sum of seed, fertilizer, and operational costs"},
            {"Budget Item": "ESTIMATED GROSS REVENUE", "Amount (INR)": f"₹ {econ['estimated_gross_revenue']:,.2f}" if econ['return_status'] == "SUCCESS" else "Data unavailable", "Basis": f"{econ['total_yield_quintals']} Quintals @ Modal Mandi Price ₹ {econ['mkt_price_per_quintal']:,.2f}/quintal"},
            {"Budget Item": "ESTIMATED GROSS RETURN", "Amount (INR)": f"₹ {econ['estimated_gross_return']:,.2f}" if econ['return_status'] == "SUCCESS" else "Data unavailable", "Basis": "Estimated Gross Revenue - Total Input Cost"}
        ])
        st.dataframe(breakdown_df, use_container_width=True)

        st.info("""
        **ESTIMATED GROSS RETURN DISCLAIMER**:
        - **Gross Return** = Estimated Gross Revenue − Total Estimated Input Cost (Seeds + Subsidized Fertilizer + Operational Expenses).
        - Excludes land lease rentals, depreciation on machinery, long-term capital interest, and post-harvest transport expenses.
        - Not a guaranteed farmer income predictor. Mandi prices fluctuate daily based on APMC supply-demand dynamics.
        """)
