"""
Data loading module for AgriSense.
Loads the Crop Recommendation dataset from data/Crop_recommendation.csv.
"""

import pandas as pd
import os
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), "Crop_recommendation.csv")

def load_crop_dataset() -> pd.DataFrame:
    """Load the Crop Recommendation agricultural dataset."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        return df
    else:
        # Fallback generator if CSV is missing
        return generate_fallback_dataset()

def generate_fallback_dataset() -> pd.DataFrame:
    """Fallback generator in case dataset file is missing."""
    np.random.seed(42)
    crops = ["rice", "maize", "chickpea", "cotton", "jute", "coffee"]
    data = []
    for _ in range(500):
        data.append({
            "N": round(float(np.random.uniform(10, 140)), 2),
            "P": round(float(np.random.uniform(10, 90)), 2),
            "K": round(float(np.random.uniform(15, 85)), 2),
            "temperature": round(float(np.random.uniform(15, 38)), 2),
            "humidity": round(float(np.random.uniform(30, 95)), 2),
            "ph": round(float(np.random.uniform(4.5, 8.5)), 2),
            "rainfall": round(float(np.random.uniform(40, 300)), 2),
            "label": np.random.choice(crops)
        })
    return pd.DataFrame(data)

def get_transaction_dataset() -> list:
    """Generate transactional data for Association Rule Mining."""
    transactions = [
        ["Rice", "High Nitrogen", "Clay Soil", "Urea Fertilizer", "Heavy Irrigation"],
        ["Wheat", "Medium Nitrogen", "Loam Soil", "DAP Fertilizer", "Moderate Irrigation"],
        ["Maize", "High Phosphorus", "Sandy Soil", "NPK Fertilizer", "Moderate Irrigation"],
        ["Cotton", "High Potassium", "Black Soil", "Potash Fertilizer", "Low Irrigation"],
        ["Rice", "High Nitrogen", "Urea Fertilizer", "Heavy Irrigation"],
        ["Sugarcane", "High Nitrogen", "Clay Soil", "Urea Fertilizer", "Heavy Irrigation"],
        ["Pulses", "Low Nitrogen", "High Phosphorus", "Bio-Fertilizer", "Low Irrigation"],
        ["Wheat", "Medium Nitrogen", "Loam Soil", "Urea Fertilizer", "Moderate Irrigation"],
        ["Maize", "High Nitrogen", "NPK Fertilizer", "Moderate Irrigation"],
        ["Rice", "Heavy Irrigation", "Clay Soil", "Urea Fertilizer"],
        ["Cotton", "Black Soil", "High Potassium", "Potash Fertilizer"],
        ["Pulses", "Low Nitrogen", "Bio-Fertilizer", "Low Irrigation"],
        ["Wheat", "Loam Soil", "DAP Fertilizer", "Moderate Irrigation"],
        ["Sugarcane", "High Nitrogen", "Heavy Irrigation", "Urea Fertilizer"],
    ]
    return transactions
