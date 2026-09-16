"""
Agricultural Apriori Association Rule Mining Engine.
Uses mlxtend to convert continuous numerical features into agronomic categorical bins
and mine frequent itemsets and association rules.
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from typing import Dict, Any, Tuple

class AgriculturalAprioriEngine:
    """Apriori Rule Mining engine tailored for agricultural crop & soil telemetry."""

    @staticmethod
    def bin_agricultural_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert continuous numerical soil & microclimate attributes into meaningful agronomic categories.
        
        Binning Logic:
        - Nitrogen (N): Low (<50 kg/ha), Medium (50-90 kg/ha), High (>90 kg/ha)
        - Phosphorus (P): Low (<35 kg/ha), Medium (35-70 kg/ha), High (>70 kg/ha)
        - Potassium (K): Low (<35 kg/ha), Medium (35-80 kg/ha), High (>80 kg/ha)
        - Temperature: Cool (<20°C), Moderate (20-30°C), Warm (>30°C)
        - Humidity: Low (<50%), Moderate (50-75%), High (>75%)
        - Soil pH: Acidic (<6.0), Neutral (6.0-7.5), Alkaline (>7.5)
        - Rainfall: Low (<75 mm), Moderate (75-150 mm), Heavy (>150 mm)
        - Crop Label: Crop_<label_name>
        """
        binned = pd.DataFrame()

        # 1. Nitrogen (N)
        binned["N_Bin"] = pd.cut(
            df["N"],
            bins=[-np.inf, 50, 90, np.inf],
            labels=["N_Low", "N_Medium", "N_High"]
        )

        # 2. Phosphorus (P)
        binned["P_Bin"] = pd.cut(
            df["P"],
            bins=[-np.inf, 35, 70, np.inf],
            labels=["P_Low", "P_Medium", "P_High"]
        )

        # 3. Potassium (K)
        binned["K_Bin"] = pd.cut(
            df["K"],
            bins=[-np.inf, 35, 80, np.inf],
            labels=["K_Low", "K_Medium", "K_High"]
        )

        # 4. Temperature (°C)
        binned["Temp_Bin"] = pd.cut(
            df["temperature"],
            bins=[-np.inf, 20, 30, np.inf],
            labels=["Temp_Cool", "Temp_Moderate", "Temp_Warm"]
        )

        # 5. Humidity (%)
        binned["Hum_Bin"] = pd.cut(
            df["humidity"],
            bins=[-np.inf, 50, 75, np.inf],
            labels=["Hum_Low", "Hum_Moderate", "Hum_High"]
        )

        # 6. Soil pH
        binned["pH_Bin"] = pd.cut(
            df["ph"],
            bins=[-np.inf, 6.0, 7.5, np.inf],
            labels=["pH_Acidic", "pH_Neutral", "pH_Alkaline"]
        )

        # 7. Rainfall (mm)
        binned["Rain_Bin"] = pd.cut(
            df["rainfall"],
            bins=[-np.inf, 75, 150, np.inf],
            labels=["Rain_Low", "Rain_Moderate", "Rain_Heavy"]
        )

        # 8. Target Crop Label
        if "label" in df.columns:
            binned["Crop"] = "Crop_" + df["label"].astype(str)

        return binned

    @classmethod
    def one_hot_encode(cls, binned_df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical transaction items into a boolean DataFrame."""
        dummies = pd.get_dummies(binned_df, prefix="", prefix_sep="")
        return dummies.astype(bool)

    @classmethod
    def mine_rules(
        cls,
        df: pd.DataFrame,
        min_support: float = 0.05,
        min_confidence: float = 0.60,
        min_lift: float = 1.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute Apriori algorithm and generate association rules.
        Returns (frequent_itemsets, association_rules_df)
        """
        binned_df = cls.bin_agricultural_data(df)
        onehot_df = cls.one_hot_encode(binned_df)

        # 1. Mining Frequent Itemsets via Apriori
        frequent_itemsets = apriori(
            onehot_df,
            min_support=min_support,
            use_colnames=True
        )

        if frequent_itemsets.empty:
            return pd.DataFrame(), pd.DataFrame()

        frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))
        frequent_itemsets["itemsets_str"] = frequent_itemsets["itemsets"].apply(lambda x: ", ".join(sorted(list(x))))

        # 2. Mining Association Rules
        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence
        )

        if rules.empty:
            return frequent_itemsets, pd.DataFrame()

        # Filter rules by minimum lift
        rules = rules[rules["lift"] >= min_lift].copy()

        # Format string representations of antecedents & consequents
        rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
        rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))

        # Round metric values for display
        for col in ["support", "confidence", "lift", "leverage", "conviction"]:
            if col in rules.columns:
                rules[col] = rules[col].round(3)

        # Sort by Lift descending
        rules = rules.sort_values(by="lift", ascending=False).reset_index(drop=True)
        return frequent_itemsets, rules

    @staticmethod
    def generate_agronomic_interpretations(rules_df: pd.DataFrame) -> list:
        """Generate human-readable agronomic insights from top mined association rules."""
        if rules_df.empty:
            return ["No association rules met the specified support, confidence, and lift thresholds."]

        insights = []
        for idx, row in rules_df.head(5).iterrows():
            ant = row['antecedents_str']
            con = row['consequents_str']
            conf_pct = row['confidence'] * 100
            lift_val = row['lift']

            # If rule points to a crop
            if "Crop_" in con:
                crop_name = con.replace("Crop_", "")
                insight = (
                    f"Rule #{idx+1}: When soil & weather condition [{ant}] occurs, "
                    f"the recommendation for **{crop_name}** holds with **{conf_pct:.1f}% confidence** "
                    f"(Lift: **{lift_val:.2f}**x higher likelihood than random expectation)."
                )
            else:
                insight = (
                    f"Rule #{idx+1}: Agricultural attribute [{ant}] strongly co-occurs with [{con}] "
                    f"with **{conf_pct:.1f}% confidence** and **{lift_val:.2f}** lift factor."
                )
            insights.append(insight)

        return insights
