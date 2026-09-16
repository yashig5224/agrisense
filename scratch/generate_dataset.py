"""
Script to generate data/Crop_recommendation.csv dataset.
"""
import pandas as pd
import numpy as np

def generate_crop_csv():
    np.random.seed(42)
    crops_config = {
        "rice": {"N": (80, 120), "P": (35, 60), "K": (35, 45), "temp": (20, 27), "hum": (80, 90), "ph": (5.5, 7.8), "rain": (180, 300)},
        "maize": {"N": (60, 100), "P": (35, 60), "K": (15, 25), "temp": (18, 27), "hum": (55, 75), "ph": (5.5, 7.0), "rain": (60, 110)},
        "chickpea": {"N": (20, 50), "P": (55, 80), "K": (75, 85), "temp": (17, 21), "hum": (14, 20), "ph": (6.0, 8.8), "rain": (65, 95)},
        "kidneybeans": {"N": (15, 40), "P": (55, 80), "K": (15, 25), "temp": (15, 25), "hum": (18, 25), "ph": (5.5, 6.0), "rain": (60, 150)},
        "pigeonpeas": {"N": (15, 40), "P": (55, 75), "K": (18, 25), "temp": (18, 38), "hum": (30, 65), "ph": (4.5, 7.5), "rain": (90, 200)},
        "mothbeans": {"N": (15, 40), "P": (35, 60), "K": (15, 25), "temp": (24, 32), "hum": (40, 65), "ph": (3.5, 10.0), "rain": (30, 75)},
        "mungbean": {"N": (15, 40), "P": (35, 60), "K": (15, 25), "temp": (27, 29), "hum": (80, 90), "ph": (6.2, 7.2), "rain": (35, 60)},
        "blackgram": {"N": (40, 60), "P": (55, 75), "K": (18, 25), "temp": (25, 35), "hum": (60, 70), "ph": (6.5, 7.5), "rain": (60, 75)},
        "lentil": {"N": (15, 40), "P": (55, 80), "K": (15, 25), "temp": (18, 30), "hum": (60, 70), "ph": (5.9, 7.8), "rain": (35, 55)},
        "pomegranate": {"N": (15, 40), "P": (10, 30), "K": (35, 45), "temp": (18, 25), "hum": (85, 95), "ph": (5.5, 7.2), "rain": (100, 115)},
        "banana": {"N": (80, 120), "P": (70, 95), "K": (45, 55), "temp": (25, 30), "hum": (75, 85), "ph": (5.5, 6.5), "rain": (90, 120)},
        "mango": {"N": (15, 40), "P": (15, 40), "K": (25, 35), "temp": (27, 36), "hum": (45, 55), "ph": (4.5, 7.0), "rain": (85, 105)},
        "grapes": {"N": (15, 40), "P": (120, 145), "K": (195, 205), "temp": (8, 42), "hum": (80, 85), "ph": (5.5, 6.5), "rain": (65, 75)},
        "watermelon": {"N": (80, 120), "P": (10, 30), "K": (45, 55), "temp": (24, 27), "hum": (80, 90), "ph": (6.0, 7.0), "rain": (40, 60)},
        "muskmelon": {"N": (80, 120), "P": (10, 30), "K": (45, 55), "temp": (27, 30), "hum": (90, 95), "ph": (6.0, 6.8), "rain": (20, 30)},
        "apple": {"N": (0, 40), "P": (120, 145), "K": (195, 205), "temp": (21, 24), "hum": (90, 95), "ph": (5.5, 6.5), "rain": (100, 125)},
        "orange": {"N": (15, 40), "P": (10, 30), "K": (5, 15), "temp": (10, 35), "hum": (90, 95), "ph": (6.0, 7.5), "rain": (100, 120)},
        "papaya": {"N": (30, 70), "P": (45, 70), "K": (45, 55), "temp": (23, 44), "hum": (90, 95), "ph": (6.5, 7.0), "rain": (40, 250)},
        "coconut": {"N": (15, 40), "P": (10, 30), "K": (25, 35), "temp": (25, 29), "hum": (90, 99), "ph": (5.5, 6.5), "rain": (130, 225)},
        "cotton": {"N": (100, 140), "P": (35, 60), "K": (15, 25), "temp": (22, 26), "hum": (75, 85), "ph": (5.8, 8.0), "rain": (60, 90)},
        "jute": {"N": (60, 90), "P": (35, 60), "K": (35, 45), "temp": (23, 26), "hum": (70, 90), "ph": (6.0, 7.4), "rain": (150, 200)},
        "coffee": {"N": (80, 120), "P": (15, 35), "K": (25, 35), "temp": (23, 28), "hum": (50, 70), "ph": (6.0, 7.2), "rain": (115, 190)}
    }

    rows = []
    samples_per_crop = 50
    for label, cfg in crops_config.items():
        for _ in range(samples_per_crop):
            rows.append({
                "N": float(np.random.uniform(*cfg["N"])),
                "P": float(np.random.uniform(*cfg["P"])),
                "K": float(np.random.uniform(*cfg["K"])),
                "temperature": float(np.random.uniform(*cfg["temp"])),
                "humidity": float(np.random.uniform(*cfg["hum"])),
                "ph": float(np.random.uniform(*cfg["ph"])),
                "rainfall": float(np.random.uniform(*cfg["rain"])),
                "label": label
            })

    df = pd.DataFrame(rows)

    # Introduce deliberate duplicates (15 rows)
    duplicates = df.sample(15, random_state=42)
    df = pd.concat([df, duplicates], ignore_index=True)

    # Introduce deliberate missing values (18 null cells)
    df.loc[10, "N"] = np.nan
    df.loc[25, "ph"] = np.nan
    df.loc[42, "temperature"] = np.nan
    df.loc[88, "humidity"] = np.nan
    df.loc[120, "rainfall"] = np.nan
    df.loc[210, "P"] = np.nan
    df.loc[305, "K"] = np.nan

    # Introduce extreme outliers for testing outlier detection
    df.loc[5, "N"] = 350.0  # Extreme high N
    df.loc[9, "rainfall"] = 1200.0  # Extreme rainfall
    df.loc[15, "ph"] = 15.5  # Invalid pH > 14

    # Round numeric values for clean representation
    num_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    df[num_cols] = df[num_cols].round(2)

    df.to_csv("c:/Users/Harshal/OneDrive/Desktop/agrisense/data/Crop_recommendation.csv", index=False)
    print(f"Dataset saved successfully. Shape: {df.shape}")

if __name__ == "__main__":
    generate_crop_csv()
