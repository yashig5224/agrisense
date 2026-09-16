"""
Generate data/Soil_Water_Quality.csv dataset for Phase 7 Clustering.
"""
import pandas as pd
import numpy as np

def generate_water_dataset():
    np.random.seed(42)
    n_samples = 550

    ec = np.random.uniform(0.4, 4.5, size=n_samples).round(2)
    moisture = np.random.uniform(20.0, 90.0, size=n_samples).round(2)
    ph = np.random.uniform(5.5, 9.0, size=n_samples).round(2)
    sar = np.random.uniform(1.0, 15.0, size=n_samples).round(2)
    nitrate = np.random.uniform(5.0, 85.0, size=n_samples).round(2)
    om = np.random.uniform(0.5, 4.0, size=n_samples).round(2)

    df = pd.DataFrame({
        "EC_dS_m": ec,
        "Moisture_Pct": moisture,
        "pH": ph,
        "Sodium_Adsorption_Ratio": sar,
        "Nitrate_PPM": nitrate,
        "Organic_Matter_Pct": om
    })

    df.to_csv("c:/Users/Harshal/OneDrive/Desktop/agrisense/data/Soil_Water_Quality.csv", index=False)
    print("Soil_Water_Quality.csv generated successfully. Shape:", df.shape)

if __name__ == "__main__":
    generate_water_dataset()
