"""
Association Rule Mining UI module for AgriSense (Phase 3).
Integrates AgriculturalAprioriEngine with interactive sliders, Plotly rule charts,
frequent itemsets tables, rule filtering, and agronomic interpretation.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import render_header, render_metric_card
from association.apriori_engine import AgriculturalAprioriEngine

def render_association_page(df_raw: pd.DataFrame):
    """Render Phase 3 Association Rule Mining interface."""
    render_header(
        "Association Rule Mining Engine (Apriori Algorithm)",
        "Discover frequent co-occurrence patterns, environmental conditions, and crop associations using mlxtend Apriori rule mining."
    )

    # Binning Logic Documentation Expander
    with st.expander("Categorical Binning Logic Documentation (Numerical -> Transactional Items)"):
        st.markdown("""
        To apply the **Apriori Algorithm**, continuous agricultural variables are transformed into discrete categorical items:
        - **Nitrogen (N)**: `N_Low` (<50 kg/ha), `N_Medium` (50–90 kg/ha), `N_High` (>90 kg/ha)
        - **Phosphorus (P)**: `P_Low` (<35 kg/ha), `P_Medium` (35–70 kg/ha), `P_High` (>70 kg/ha)
        - **Potassium (K)**: `K_Low` (<35 kg/ha), `K_Medium` (35–80 kg/ha), `K_High` (>80 kg/ha)
        - **Temperature**: `Temp_Cool` (<20°C), `Temp_Moderate` (20–30°C), `Temp_Warm` (>30°C)
        - **Humidity**: `Hum_Low` (<50%), `Hum_Moderate` (50–75%), `Hum_High` (>75%)
        - **Soil pH**: `pH_Acidic` (<6.0), `pH_Neutral` (6.0–7.5), `pH_Alkaline` (>7.5)
        - **Rainfall**: `Rain_Low` (<75 mm), `Rain_Moderate` (75–150 mm), `Rain_Heavy` (>150 mm)
        - **Crop Label**: `Crop_<label>` (e.g. `Crop_rice`, `Crop_maize`, `Crop_cotton`)
        """)

    st.markdown("### Rule Mining Control Panel")
    c1, c2, c3 = st.columns(3)

    with c1:
        min_support = st.slider("Minimum Support Threshold", min_value=0.01, max_value=0.30, value=0.05, step=0.01)
    with c2:
        min_confidence = st.slider("Minimum Confidence Threshold", min_value=0.10, max_value=1.00, value=0.50, step=0.05)
    with c3:
        min_lift = st.slider("Minimum Lift Threshold", min_value=1.0, max_value=5.0, value=1.2, step=0.1)

    # Run Apriori Mining
    frequent_itemsets, rules = AgriculturalAprioriEngine.mine_rules(
        df_raw,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift
    )

    st.markdown("---")

    # High-level Metrics Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Frequent Itemsets", f"{len(frequent_itemsets):,}", f"Min Support: {min_support:.2f}")
    with m2:
        render_metric_card("Mined Rules", f"{len(rules):,}", f"Min Conf: {min_confidence:.2f}")
    with m3:
        max_lift = f"{rules['lift'].max():.2f}" if not rules.empty else "N/A"
        render_metric_card("Maximum Lift", max_lift, "Rule correlation strength")
    with m4:
        max_conf = f"{rules['confidence'].max()*100:.1f}%" if not rules.empty else "N/A"
        render_metric_card("Maximum Confidence", max_conf, "Rule certainty score")

    st.markdown("---")

    # Tabs Layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "Discovered Association Rules",
        "Frequent Itemsets Table",
        "Rule Visualizations",
        "Agronomic Insights & Metric Guide"
    ])

    with tab1:
        st.subheader("Mined Agricultural Association Rules")

        if rules.empty:
            st.warning("No association rules found for the selected thresholds. Try lowering Minimum Support or Minimum Confidence.")
        else:
            # Rule Filtering Controls
            st.markdown("### Rule Filter")
            f_col1, f_col2 = st.columns([2, 1])
            with f_col1:
                search_query = st.text_input("Search Antecedents / Consequents (e.g., 'rice', 'N_High', 'pH_Acidic')", value="")
            with f_col2:
                sort_by = st.selectbox("Sort Rules By", ["lift", "confidence", "support"])

            display_rules = rules.copy()

            if search_query.strip():
                query = search_query.strip().lower()
                display_rules = display_rules[
                    display_rules["antecedents_str"].str.lower().str.contains(query) |
                    display_rules["consequents_str"].str.lower().str.contains(query)
                ]

            display_rules = display_rules.sort_values(by=sort_by, ascending=False).reset_index(drop=True)

            table_df = display_rules[["antecedents_str", "consequents_str", "support", "confidence", "lift", "leverage", "conviction"]].copy()
            table_df.columns = ["Antecedents (IF)", "Consequents (THEN)", "Support", "Confidence", "Lift", "Leverage", "Conviction"]

            st.dataframe(table_df, use_container_width=True)
            st.caption(f"Displaying {len(display_rules)} rules matching criteria.")

    with tab2:
        st.subheader("Frequent Itemsets (Apriori)")
        if frequent_itemsets.empty:
            st.info("No frequent itemsets found for the current minimum support threshold.")
        else:
            itemsets_df = frequent_itemsets[["itemsets_str", "length", "support"]].copy()
            itemsets_df.columns = ["Itemset Combination", "Set Length", "Support Score"]
            itemsets_df = itemsets_df.sort_values(by="Support Score", ascending=False).reset_index(drop=True)
            st.dataframe(itemsets_df, use_container_width=True)

    with tab3:
        st.subheader("Interactive Rule Scatter & Bar Charts")

        if rules.empty:
            st.info("No rules available to visualize.")
        else:
            r_col1, r_col2 = st.columns(2)

            with r_col1:
                st.markdown("### Support vs. Confidence (Colored by Lift)")
                scatter_fig = px.scatter(
                    rules,
                    x="support",
                    y="confidence",
                    color="lift",
                    size="lift",
                    hover_data=["antecedents_str", "consequents_str"],
                    title="Rules Metric Scatter Distribution",
                    color_continuous_scale=[[0, "#8B9D83"], [0.5, "#739072"], [1.0, "#2D5A27"]]
                )
                scatter_fig.update_layout(
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    font=dict(family="Inter, sans-serif", color="#1A202C"),
                    xaxis=dict(showgrid=True, gridcolor="#E2E8F0", title="Support"),
                    yaxis=dict(showgrid=True, gridcolor="#E2E8F0", title="Confidence"),
                    height=400
                )
                st.plotly_chart(scatter_fig, use_container_width=True)

            with r_col2:
                st.markdown("### Top 10 Association Rules by Lift")
                top_10 = rules.head(10).copy()
                top_10["Rule_Label"] = top_10["antecedents_str"] + " => " + top_10["consequents_str"]

                bar_fig = px.bar(
                    top_10,
                    x="lift",
                    y="Rule_Label",
                    orientation="h",
                    color_discrete_sequence=["#2D5A27"],
                    title="Top Rules Correlation Strength (Lift)"
                )
                bar_fig.update_layout(
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    font=dict(family="Inter, sans-serif", color="#1A202C"),
                    xaxis=dict(showgrid=True, gridcolor="#E2E8F0", title="Lift Score"),
                    yaxis=dict(autorange="reversed", title="Mined Rule"),
                    height=400
                )
                st.plotly_chart(bar_fig, use_container_width=True)

    with tab4:
        st.subheader("Agronomic Interpretation of Top Rules")
        insights = AgriculturalAprioriEngine.generate_agronomic_interpretations(rules)
        for insight in insights:
            st.markdown(f"- {insight}")

        st.markdown("---")
        st.subheader("Data Mining Metrics Reference Guide")
        st.markdown("""
        - **Support**: The proportion of total farm samples containing both the antecedent and consequent items:
          $$\\text{Support}(A \\rightarrow B) = \\frac{\\text{Count}(A \\cup B)}{N}$$
        - **Confidence**: The conditional probability that consequent $B$ occurs given that antecedent $A$ is present:
          $$\\text{Confidence}(A \\rightarrow B) = \\frac{\\text{Support}(A \\cup B)}{\\text{Support}(A)}$$
        - **Lift**: The ratio of observed confidence to expected confidence assuming independence. A lift $> 1.0$ indicates positive correlation:
          $$\\text{Lift}(A \\rightarrow B) = \\frac{\\text{Confidence}(A \\rightarrow B)}{\\text{Support}(B)}$$
        """)
