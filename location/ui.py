"""
Location Intelligence UI Module for AgriSense (Version 3).
Renders integrated Location Intelligence dashboard with logical workflow:
LOCATION -> AGRICULTURAL CONDITIONS -> CROP OPTIONS -> SEED -> FERTILIZER -> LOCAL PROVIDERS -> MARKET PRICES -> COST & RETURN ESTIMATION.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.helpers import render_header, render_metric_card
from location.engine import LocationIntelligenceEngine

def render_location_intelligence_page(df_crop_raw: pd.DataFrame):
    """Render Version 3 Location Intelligence Dashboard."""
    render_header(
        "Agricultural Location Intelligence & Decision Framework",
        "Integrated location-based analytics synthesizing ML crop recommendations, seed varieties, fertilizer inputs, supplier directories, and gross return budgeting."
    )

    # Provenance Disclaimer Banner
    st.info("""
    **DATA INTEGRITY & PROVENANCE DISCLAIMER**:
    Every metric displayed in Location Intelligence indicates its exact data provenance. 
    Machine learning predictions and gross return models are data-driven estimations based on historical training datasets for decision support.
    """)

    st.markdown("### Location & Agricultural Setup")

    loc_c1, loc_c2, loc_c3 = st.columns(3)

    with loc_c1:
        sel_zone = st.selectbox(
            "Select District Zone Location",
            ["North Zone", "South Zone", "Central Zone", "East Zone", "West Zone"],
            index=0
        )
        season = st.selectbox("Agricultural Season", ["Kharif / Monsoon", "Rabi / Winter", "Zaid / Summer"])
        water_avail = st.selectbox("Water Availability", ["Adequate / Irrigated", "Moderate / Rainfed", "Limited / Arid"])

    with loc_c2:
        in_n = st.number_input("Nitrogen (N kg/ha)", 0.0, 250.0, 90.0, 5.0)
        in_p = st.number_input("Phosphorus (P kg/ha)", 0.0, 250.0, 42.0, 5.0)
        in_k = st.number_input("Potassium (K kg/ha)", 0.0, 350.0, 43.0, 5.0)

    with loc_c3:
        in_temp = st.number_input("Temperature (°C)", 0.0, 50.0, 23.5, 0.5)
        in_hum = st.number_input("Humidity (%)", 0.0, 100.0, 82.0, 1.0)
        in_ph = st.number_input("Soil pH Level", 0.0, 14.0, 6.5, 0.1)
        in_rain = st.number_input("Annual Rainfall (mm)", 0.0, 600.0, 202.0, 5.0)

    st.markdown("---")
    st.markdown("### Farm Management Parameters")
    f_c1, f_c2, f_c3, f_c4 = st.columns(4)

    with f_c1:
        land_area = st.slider("Land Area (Hectares)", 0.5, 50.0, 2.5, 0.5)
    with f_c2:
        seed_rate = st.number_input("Seed Rate (kg/Ha)", 1.0, 100.0, 25.0, 1.0)
    with f_c3:
        fert_rate = st.number_input("Fertilizer Rate (kg/Ha)", 10.0, 500.0, 150.0, 10.0)
    with f_c4:
        ops_rate = st.number_input("Ops Costs (INR/Ha)", 0.0, 20000.0, 3500.0, 500.0)

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

    # Generate Location Report
    report = LocationIntelligenceEngine.generate_location_report(
        df_crop_raw, location_zone=sel_zone, soil_inputs=soil_inputs, farm_inputs=farm_inputs
    )

    st.markdown("---")

    # Workflow Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1. Crop Options & ML Predictions",
        "2. Seed Varieties",
        "3. Recommended Fertilizer",
        "4. Local Suppliers & Availability",
        "5. Market Commodity Prices",
        "6. Input Cost & Gross Return Model"
    ])

    with tab1:
        st.subheader("1. Suitable Crop Candidates & ML Model Predictions")
        st.caption("Data Provenance: `Source: ML Classification Models & Historical Dataset`")

        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            render_metric_card("Naive Bayes Recommendation", f"{report['recommended_crop']}", f"Confidence: {report['crop_confidence_pct']}%")
        with c_m2:
            render_metric_card("WEKA J48 Candidate", f"{report['j48_crop']}", "Decision tree inference")
        with c_m3:
            render_metric_card("Estimated Yield", f"{report['economic_model']['yield_tons_ha']} Tons / Ha", "Random Forest Regressor")

        st.markdown("### Underlying Data Conditions")
        st.write(f"- **Entered Nutrient Profile**: Nitrogen: {in_n} kg/ha | Phosphorus: {in_p} kg/ha | Potassium: {in_k} kg/ha | pH: {in_ph}")
        st.write(f"- **Microclimate Telemetry**: Temp: {in_temp}°C | Humidity: {in_hum}% | Rainfall: {in_rain} mm | Water: {water_avail}")
        st.write(f"- **District Zone**: {sel_zone} | **Season**: {season}")
        st.caption("Note: Crop predictions represent probabilistic data-driven estimations based on historical training data.")

    with tab2:
        st.subheader("2. Relevant Seed Varieties & Pricing")
        st.caption("Data Provenance: `Source: Retail Dealer Directory`")
        seed_info = report["seed_info"]

        s_m1, s_m2, s_m3 = st.columns(3)
        with s_m1:
            render_metric_card("Seed Variety", f"{seed_info['name']}", f"Target Crop: {report['recommended_crop']}")
        with s_m2:
            render_metric_card("Estimated Unit Price", f"₹ {seed_info['price_per_kg'] * 10:,.2f}", "per 10 kg package")
        with s_m3:
            render_metric_card("Availability Status", f"{seed_info['availability']}", f"Supplier: {seed_info['provider']}")

    with tab3:
        st.subheader("3. Recommended Fertilizer & Nutrient Management")
        st.caption("Data Provenance: `Source: Fertilizer Engine & Apriori Rules`")
        fert_info = report["fertilizer_info"]
        fert_rec = fert_info["recommendation"]

        if fert_rec.get("status") == "SUCCESS":
            f_m1, f_m2, f_m3 = st.columns(3)
            with f_m1:
                render_metric_card("Fertilizer Product", f"{fert_rec['recommended_fertilizer']}", f"Deficiency: {fert_rec['primary_deficiency']}")
            with f_m2:
                render_metric_card("Application Rate", f"{fert_rec['recommended_dose_kg_ha']} kg/ha", "Recommended dosage")
            with f_m3:
                render_metric_card("Market Price", f"₹ {fert_info['bag_price_50kg']:,.2f}", "per 50 kg bag")

            st.markdown("### Mined Co-Occurrence Pattern & Explanation")
            st.code(fert_rec["supporting_pattern"], language="text")
            st.markdown(fert_rec["explanation"])
        else:
            st.warning(fert_rec.get("message", "Insufficient historical data for a reliable recommendation."))

    with tab4:
        st.subheader("4. Nearby Fertilizer Providers & Local Availability")
        st.caption("Data Provenance: `Source: District Retail Directory`")
        prov_info = report["providers_info"]
        prov_df = prov_info["df"]

        render_metric_card("Number of Fertilizer Providers Found", f"{prov_info['provider_count']}", f"Location: {sel_zone}")

        st.markdown("### Local Supplier Directory")
        if not prov_df.empty:
            disp_prov = prov_df[["Provider_Name", "Provider_Address", "District_Zone", "Distance_km", "Name", "Price_Per_Unit", "Availability_Status"]].copy()
            disp_prov.columns = ["Provider Name", "Physical Address", "District Zone", "Distance (km)", "Available Product", "Unit Price (INR)", "Stock Status"]
            st.dataframe(disp_prov, use_container_width=True)
        else:
            st.info(f"No local suppliers currently registered in directory for '{sel_zone}'.")

    with tab5:
        st.subheader("5. Market Commodity Prices & Trend")
        st.caption("Data Provenance: `Source: Public Commodity Price Registry`")
        mkt_info = report["market_price_info"]

        if mkt_info.get("status") == "SUCCESS":
            p_m1, p_m2, p_m3 = st.columns(3)
            with p_m1:
                render_metric_card("Commodity Price", f"₹ {mkt_info['latest_price']:,.2f}", f"per Quintal ({mkt_info['market_location']})")
            with p_m2:
                render_metric_card("Market Price Trend", f"{mkt_info['price_trend']}", "Short-term trajectory")
            with p_m3:
                render_metric_card("Historical Price Range", f"₹ {mkt_info['historical_min']} - {mkt_info['historical_max']}", "Min to Max")
        else:
            st.warning("Data unavailable: Commodity market price registry currently unlisted for this crop.")

    with tab6:
        st.subheader("6. Input Cost & Estimated Gross Return Budgeting Model")
        st.caption("Data Provenance: `Source: Yield Regressor & Commodity Registry`")
        econ = report["economic_model"]

        r_m1, r_m2, r_m3 = st.columns(3)
        with r_m1:
            render_metric_card("Total Estimated Input Cost", f"₹ {econ['total_input_cost']:,.2f}", f"for {land_area} Hectares")
        with r_m2:
            render_metric_card("Estimated Gross Revenue", f"₹ {econ['estimated_gross_revenue']:,.2f}" if econ['return_status'] == "SUCCESS" else "Data unavailable", f"Yield: {econ['total_yield_quintals']} Quintals")
        with r_m3:
            render_metric_card("Estimated Gross Return", f"₹ {econ['estimated_gross_return']:,.2f}" if econ['return_status'] == "SUCCESS" else "Data unavailable", "Gross Revenue - Input Cost")

        st.markdown("---")
        st.markdown("### Budget Breakdown Summary")

        breakdown_df = pd.DataFrame([
            {"Expense / Revenue Stream": "Estimated Seed Cost", "Amount (INR)": f"₹ {econ['seed_cost']:,.2f}", "Notes": f"{seed_rate} kg/ha @ ₹ {seed_info['price_per_kg']:.2f}/kg"},
            {"Expense / Revenue Stream": "Estimated Fertilizer Cost", "Amount (INR)": f"₹ {econ['fertilizer_cost']:,.2f}", "Notes": f"{fert_rate} kg/ha @ ₹ {fert_info['bag_price_50kg']:.2f}/50kg bag"},
            {"Expense / Revenue Stream": "Operational Expenses (Labor/Irrigation)", "Amount (INR)": f"₹ {econ['operational_cost']:,.2f}", "Notes": f"Rate: ₹ {ops_rate}/ha"},
            {"Expense / Revenue Stream": "TOTAL ESTIMATED INPUT COST", "Amount (INR)": f"₹ {econ['total_input_cost']:,.2f}", "Notes": "Sum of all input costs"},
            {"Expense / Revenue Stream": "ESTIMATED GROSS REVENUE", "Amount (INR)": f"₹ {econ['estimated_gross_revenue']:,.2f}" if econ['return_status'] == "SUCCESS" else "Data unavailable", "Notes": "Yield (Quintals) × Commodity Price"},
            {"Expense / Revenue Stream": "ESTIMATED GROSS RETURN", "Amount (INR)": f"₹ {econ['estimated_gross_return']:,.2f}" if econ['return_status'] == "SUCCESS" else "Data unavailable", "Notes": "Gross Revenue - Total Input Cost"}
        ])
        st.dataframe(breakdown_df, use_container_width=True)

        st.info("""
        **BUDGETING & GROSS RETURN DISCLAIMER**:
        Gross revenue and return metrics are hypothetical estimations calculated from predicted crop yields and regional commodity prices. 
        Actual farm returns vary based on market fluctuations, weather events, and management practices. Treat as decision support budgeting.
        """)
