"""
Classification UI Module for AgriSense (Phase 4 & Phase 5).
Integrates WEKA J48 decision tree classifier, Gaussian Naive Bayes classifier,
factual model comparative analysis, and interactive crop prediction widget.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import render_header, render_metric_card
from classification.weka_j48 import WekaJ48Classifier
from classification.naive_bayes import AgriculturalNaiveBayesClassifier

def render_classification_page(df_raw: pd.DataFrame):
    """Render Classification dashboard with J48, Naive Bayes, comparison & live predictor."""
    render_header(
        "Crop & Soil Classification Engine",
        "Train, evaluate, compare, and deploy WEKA J48 decision tree and Gaussian Naive Bayes classifiers for crop recommendation."
    )

    # Global Parameters
    st.markdown("### Model Configuration Panel")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        test_ratio = st.slider("Test Partition Ratio (%)", min_value=10, max_value=40, value=20, step=5)
    with c2:
        unpruned = st.checkbox("J48 Unpruned Tree (-U)", value=False)
    with c3:
        confidence_factor = st.slider("J48 Confidence Factor (-C)", 0.05, 0.50, 0.25, 0.05) if not unpruned else 0.25
    with c4:
        min_num_obj = st.slider("J48 Min NumObj (-M)", 1, 10, 2, 1)

    # Train both classifiers dynamically on the dataset
    j48_result = WekaJ48Classifier.run_j48(
        df_raw, test_size=test_ratio/100.0, unpruned=unpruned,
        confidence_factor=confidence_factor, min_num_obj=min_num_obj
    )

    nb_result = AgriculturalNaiveBayesClassifier.train_and_evaluate(
        df_raw, test_size=test_ratio/100.0
    )

    st.markdown("---")

    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "WEKA J48 Decision Tree",
        "Gaussian Naive Bayes",
        "Factual Model Comparison (J48 vs. NB)",
        "Interactive Crop Predictor"
    ])

    with tab1:
        st.subheader("WEKA J48 Classification Results")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_metric_card("J48 Accuracy", f"{j48_result['accuracy']:.2f}%", "Test set score")
        with m2:
            render_metric_card("Correctly Classified", f"{j48_result['correct_count']}", f"out of {j48_result['total_test_instances']}")
        with m3:
            render_metric_card("Incorrectly Classified", f"{j48_result['incorrect_count']}", "error instances")
        with m4:
            render_metric_card("Weighted F1-Score", f"{j48_result['f1_weighted']:.3f}", "Harmonic mean")

        st.markdown("### Generated J48 Tree Rules")
        st.code(j48_result["tree_structure"], language="text")

        st.markdown("### J48 Confusion Matrix")
        if isinstance(j48_result["confusion_matrix"], np.ndarray):
            cm_fig = px.imshow(
                j48_result["confusion_matrix"],
                x=j48_result["classes"],
                y=j48_result["classes"],
                labels=dict(x="Predicted Crop", y="Actual Crop", color="Count"),
                color_continuous_scale=[[0, "#F4F6F4"], [0.5, "#8B9D83"], [1.0, "#1B3B18"]],
                title="J48 Confusion Matrix"
            )
            cm_fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", height=500, xaxis=dict(tickangle=-45))
            st.plotly_chart(cm_fig, use_container_width=True)

        st.markdown("### J48 Per-Class Performance")
        st.dataframe(j48_result["class_metrics"], use_container_width=True)

    with tab2:
        st.subheader("Gaussian Naive Bayes Classification Results")

        nb_m1, nb_m2, nb_m3, nb_m4 = st.columns(4)
        with nb_m1:
            render_metric_card("NB Accuracy", f"{nb_result['accuracy']:.2f}%", "Test set score")
        with nb_m2:
            render_metric_card("Correctly Classified", f"{nb_result['correct_count']}", f"out of {nb_result['total_test_instances']}")
        with nb_m3:
            render_metric_card("Incorrectly Classified", f"{nb_result['incorrect_count']}", "error instances")
        with nb_m4:
            render_metric_card("Weighted F1-Score", f"{nb_result['f1_weighted']:.3f}", "Harmonic mean")

        st.markdown("### Naive Bayes Confusion Matrix")
        nb_cm_fig = px.imshow(
            nb_result["confusion_matrix"],
            x=nb_result["classes"],
            y=nb_result["classes"],
            labels=dict(x="Predicted Crop", y="Actual Crop", color="Count"),
            color_continuous_scale=[[0, "#F4F6F4"], [0.5, "#8B9D83"], [1.0, "#1B3B18"]],
            title="Gaussian Naive Bayes Confusion Matrix"
        )
        nb_cm_fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", height=500, xaxis=dict(tickangle=-45))
        st.plotly_chart(nb_cm_fig, use_container_width=True)

        st.markdown("### Naive Bayes Classification Report")
        st.dataframe(nb_result["class_metrics"], use_container_width=True)

    with tab3:
        st.subheader("Factual Classification Model Comparison")
        st.caption("Empirical benchmark derived from identical train/test data splits without subjective bias:")

        comp_data = [
            {
                "Evaluation Metric": "Classification Accuracy (%)",
                "WEKA J48 Decision Tree": f"{j48_result['accuracy']:.2f}%",
                "Gaussian Naive Bayes": f"{nb_result['accuracy']:.2f}%",
                "Difference": f"{(j48_result['accuracy'] - nb_result['accuracy']):+.2f}%"
            },
            {
                "Evaluation Metric": "Weighted Precision",
                "WEKA J48 Decision Tree": f"{j48_result['precision_weighted']:.3f}",
                "Gaussian Naive Bayes": f"{nb_result['precision']:.3f}",
                "Difference": f"{(j48_result['precision_weighted'] - nb_result['precision']):+.3f}"
            },
            {
                "Evaluation Metric": "Weighted Recall",
                "WEKA J48 Decision Tree": f"{j48_result['recall_weighted']:.3f}",
                "Gaussian Naive Bayes": f"{nb_result['recall']:.3f}",
                "Difference": f"{(j48_result['recall_weighted'] - nb_result['recall']):+.3f}"
            },
            {
                "Evaluation Metric": "Weighted F1-Score",
                "WEKA J48 Decision Tree": f"{j48_result['f1_weighted']:.3f}",
                "Gaussian Naive Bayes": f"{nb_result['f1']:.3f}",
                "Difference": f"{(j48_result['f1_weighted'] - nb_result['f1']):+.3f}"
            },
            {
                "Evaluation Metric": "Correctly Classified Samples",
                "WEKA J48 Decision Tree": f"{j48_result['correct_count']} / {j48_result['total_test_instances']}",
                "Gaussian Naive Bayes": f"{nb_result['correct_count']} / {nb_result['total_test_instances']}",
                "Difference": f"{(j48_result['correct_count'] - nb_result['correct_count']):+d}"
            },
            {
                "Evaluation Metric": "Incorrectly Classified Samples",
                "WEKA J48 Decision Tree": f"{j48_result['incorrect_count']}",
                "Gaussian Naive Bayes": f"{nb_result['incorrect_count']}",
                "Difference": f"{(j48_result['incorrect_count'] - nb_result['incorrect_count']):+d}"
            }
        ]
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True)

        # Comparative Bar Chart
        comp_chart_df = pd.DataFrame([
            {"Metric": "Accuracy (%)", "Model": "WEKA J48", "Score": j48_result["accuracy"]},
            {"Metric": "Accuracy (%)", "Model": "Gaussian Naive Bayes", "Score": nb_result["accuracy"]},
            {"Metric": "F1-Score (×100)", "Model": "WEKA J48", "Score": j48_result["f1_weighted"] * 100},
            {"Metric": "F1-Score (×100)", "Model": "Gaussian Naive Bayes", "Score": nb_result["f1"] * 100},
        ])
        comp_fig = px.bar(
            comp_chart_df,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            color_discrete_sequence=["#2D5A27", "#739072"],
            title="Model Benchmark Comparison"
        )
        comp_fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", height=380)
        st.plotly_chart(comp_fig, use_container_width=True)

    with tab4:
        st.subheader("Interactive Single-Sample Crop Predictor")
        st.caption("Input real-time soil nutrient and environmental parameters to receive crop recommendations from the trained Gaussian Naive Bayes model:")

        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1:
            in_n = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=90.0, step=5.0)
            in_p = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=42.0, step=5.0)
        with p_col2:
            in_k = st.number_input("Potassium (K)", min_value=0.0, max_value=300.0, value=43.0, step=5.0)
            in_temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=23.5, step=0.5)
        with p_col3:
            in_hum = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0, step=1.0)
            in_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        with p_col4:
            in_rain = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=202.0, step=5.0)
            predict_btn = st.button("Predict Optimal Crop")

        inputs_dict = {
            "N": in_n, "P": in_p, "K": in_k,
            "temperature": in_temp, "humidity": in_hum,
            "ph": in_ph, "rainfall": in_rain
        }

        pred_res = AgriculturalNaiveBayesClassifier.predict_single_sample(nb_result["model"], inputs_dict)

        st.markdown("---")
        res_c1, res_c2 = st.columns([1, 2])

        with res_c1:
            render_metric_card("Recommended Crop", f"{pred_res['predicted_crop'].upper()}", f"Confidence: {pred_res['confidence_pct']}%")

        with res_c2:
            st.markdown("### Top Crop Probability Distribution")
            st.dataframe(pred_res["probabilities_table"].head(5), use_container_width=True)
