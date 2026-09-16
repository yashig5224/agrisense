"""
Generate data/Soil_Organic_Carbon.csv dataset for Phase 6 Regression.
"""
import pandas as pd
import numpy as np

def generate_soc_dataset():
    np.random.seed(42)
    n_samples = 600

    clay_pct = np.random.uniform(10.0, 60.0, size=n_samples).round(2)
    moisture_pct = np.random.uniform(15.0, 85.0, size=n_samples).round(2)
    nitrogen_n = np.random.uniform(20.0, 140.0, size=n_samples).round(2)
    bulk_density = np.random.uniform(1.0, 1.7, size=n_samples).round(2)
    soil_ph = np.random.uniform(5.0, 8.5, size=n_samples).round(2)
    temperature_c = np.random.uniform(15.0, 35.0, size=n_samples).round(2)

    # Compute target Organic Carbon Pct (0.2% to 2.5%)
    soc = (
        0.3
        + (clay_pct * 0.015)
        + (nitrogen_n * 0.005)
        + (moisture_pct * 0.008)
        - (bulk_density * 0.35)
        - (temperature_c * 0.01)
        + np.random.normal(0, 0.05, size=n_samples)
    ).round(2)

    soc = np.clip(soc, 0.25, 2.85)

    df = pd.DataFrame({
        "Clay_Pct": clay_pct,
        "Moisture_Pct": moisture_pct,
        "Nitrogen_N": nitrogen_n,
        "Bulk_Density": bulk_density,
        "Soil_pH": soil_ph,
        "Temperature_C": temperature_c,
        "Organic_Carbon_Pct": soc
    })

    df.to_csv("c:/Users/Harshal/OneDrive/Desktop/agrisense/data/Soil_Organic_Carbon.csv", index=False)
    print("Soil_Organic_Carbon.csv generated successfully. Shape:", df.shape)

if __name__ == "__main__":
    generate_soc_dataset()
