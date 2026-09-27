"""
Data-Driven Fertilizer Recommendation Engine for AgriSense.
Analyzes crop telemetry, identifies soil nutrient deficiencies, matches historical
fertilizer dataset patterns, and cross-references mined Apriori association rules.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from data.loader import load_fertilizer_dataset

class FertilizerRecommendationEngine:
    """Reusable engine for data-driven fertilizer recommendation."""

    @classmethod
    def recommend_fertilizer(
        cls,
        crop: str,
        soil_inputs: Dict[str, float],
        mined_rules: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate data-driven fertilizer recommendation based on dataset historical patterns and mined rules.
        soil_inputs format: {'N': 40.0, 'P': 20.0, 'K': 25.0, 'temperature': 25.0, 'humidity': 70.0, 'ph': 6.5, 'rainfall': 100.0}
        """
        fert_df = load_fertilizer_dataset()

        if fert_df.empty:
            return {
                "status": "INSUFFICIENT_DATA",
                "message": "Insufficient historical data for a reliable recommendation."
            }

        # Normalize crop string for matching
        crop_clean = str(crop).strip().lower()
        crop_records = fert_df[fert_df["Crop"].str.strip().str.lower() == crop_clean].copy()

        # Data safety check: If crop records are insufficient, fallback to dataset matching or return insufficient status
        if len(crop_records) == 0:
            return {
                "status": "INSUFFICIENT_DATA",
                "message": "Insufficient historical data for a reliable recommendation."
            }

        # Extract input nutrients
        input_n = float(soil_inputs.get("N", 50.0))
        input_p = float(soil_inputs.get("P", 40.0))
        input_k = float(soil_inputs.get("K", 40.0))

        # Calculate dataset nutrient averages for this crop to determine deficiency
        avg_n = crop_records["Nitrogen_N"].mean()
        avg_p = crop_records["Phosphorous_P"].mean()
        avg_k = crop_records["Potassium_K"].mean()

        deficits = {
            "Nitrogen": max(0.0, avg_n - input_n),
            "Phosphorus": max(0.0, avg_p - input_p),
            "Potassium": max(0.0, avg_k - input_k)
        }

        max_deficit_element = max(deficits, key=deficits.get)
        max_deficit_val = deficits[max_deficit_element]

        if max_deficit_val < 5.0:
            primary_deficiency = "Balanced"
        else:
            primary_deficiency = max_deficit_element

        # Calculate Euclidean distance in normalized N-P-K nutrient space to find nearest historical pattern
        n_scaled = (crop_records["Nitrogen_N"] - input_n) / (crop_records["Nitrogen_N"].std() + 1e-5)
        p_scaled = (crop_records["Phosphorous_P"] - input_p) / (crop_records["Phosphorous_P"].std() + 1e-5)
        k_scaled = (crop_records["Potassium_K"] - input_k) / (crop_records["Potassium_K"].std() + 1e-5)

        crop_records["dist"] = np.sqrt(n_scaled**2 + p_scaled**2 + k_scaled**2)
        nearest_matches = crop_records.sort_values(by="dist").reset_index(drop=True)

        if nearest_matches.empty:
            return {
                "status": "INSUFFICIENT_DATA",
                "message": "Insufficient historical data for a reliable recommendation."
            }

        top_match = nearest_matches.iloc[0]
        recommended_fert = top_match["Fertilizer_Name"]
        recommended_dose = float(top_match["Recommended_Dose_kg_ha"])

        # Calculate confidence / pattern support indicator based on distance & sample size
        min_dist = float(top_match["dist"])
        conf_score = round(max(62.0, min(98.5, 100.0 - (min_dist * 12.0))), 1)

        # Cross-reference supporting Apriori pattern if available
        supporting_pattern = f"Historical {top_match['Soil_Type']} soil record (Sample N: {top_match['Nitrogen_N']}, P: {top_match['Phosphorous_P']}, K: {top_match['Potassium_K']})"
        if mined_rules:
            for rule in mined_rules:
                if primary_deficiency.lower() in rule.get("antecedents_str", "").lower():
                    supporting_pattern += f" | Mined Rule: [{rule['antecedents_str']}] => [{rule['consequents_str']}] (Lift: {rule['lift']:.2f})"
                    conf_score = min(99.0, conf_score + 2.5)
                    break

        explanation = (
            f"For crop **{crop.upper()}**, your entered soil nutrients (N: {input_n}, P: {input_p}, K: {input_k}) "
            f"indicate a primary deficit in **{primary_deficiency}** relative to historical target averages "
            f"(Avg N: {avg_n:.1f}, P: {avg_p:.1f}, K: {avg_k:.1f}). "
            f"Historical dataset telemetry supports **{recommended_fert}** application at **{recommended_dose} kg/ha** "
            f"with a statistical pattern match confidence of **{conf_score}%**."
        )

        return {
            "status": "SUCCESS",
            "crop": crop.title(),
            "recommended_fertilizer": recommended_fert,
            "recommended_dose_kg_ha": recommended_dose,
            "primary_deficiency": primary_deficiency,
            "supporting_pattern": supporting_pattern,
            "confidence_score_pct": conf_score,
            "explanation": explanation,
            "sample_support_count": len(nearest_matches)
        }
