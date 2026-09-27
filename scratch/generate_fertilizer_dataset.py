"""
Script to generate structured data/Fertilizer_prediction.csv dataset.
"""
import pandas as pd
import numpy as np

def generate_fertilizer_csv():
    np.random.seed(42)

    crops = [
        "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas", "mothbeans",
        "mungbean", "blackgram", "lentil", "pomegranate", "banana", "mango",
        "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya",
        "coconut", "cotton", "jute", "coffee"
    ]

    soil_types = ["Clay", "Loam", "Sandy", "Black Soil", "Alluvial"]

    fertilizer_map = {
        "Nitrogen": [("Urea", 46, 0, 0), ("28-28-0", 28, 28, 0)],
        "Phosphorus": [("DAP", 18, 46, 0), ("14-35-14", 14, 35, 14), ("SSP", 0, 16, 0)],
        "Potassium": [("MOP", 0, 0, 60), ("10-26-26", 10, 26, 26), ("17-17-17", 17, 17, 17)],
        "Balanced": [("17-17-17", 17, 17, 17), ("NPK Fertilizer", 19, 19, 19)]
    }

    rows = []
    samples_per_crop = 25

    for crop in crops:
        for _ in range(samples_per_crop):
            soil = np.random.choice(soil_types)
            temp = round(float(np.random.uniform(18.0, 38.0)), 1)
            humidity = round(float(np.random.uniform(40.0, 95.0)), 1)
            moisture = round(float(np.random.uniform(20.0, 80.0)), 1)

            # Random deficit bias
            def_type = np.random.choice(["Nitrogen", "Phosphorus", "Potassium", "Balanced"], p=[0.35, 0.30, 0.25, 0.10])

            if def_type == "Nitrogen":
                n_val = round(float(np.random.uniform(10.0, 45.0)), 1)
                p_val = round(float(np.random.uniform(35.0, 75.0)), 1)
                k_val = round(float(np.random.uniform(35.0, 75.0)), 1)
                fert_tuple = fertilizer_map["Nitrogen"][np.random.choice(len(fertilizer_map["Nitrogen"]))]
                dose = round(float(np.random.uniform(80.0, 140.0)), 1)

            elif def_type == "Phosphorus":
                n_val = round(float(np.random.uniform(50.0, 110.0)), 1)
                p_val = round(float(np.random.uniform(10.0, 30.0)), 1)
                k_val = round(float(np.random.uniform(35.0, 75.0)), 1)
                fert_tuple = fertilizer_map["Phosphorus"][np.random.choice(len(fertilizer_map["Phosphorus"]))]
                dose = round(float(np.random.uniform(50.0, 100.0)), 1)

            elif def_type == "Potassium":
                n_val = round(float(np.random.uniform(50.0, 110.0)), 1)
                p_val = round(float(np.random.uniform(35.0, 75.0)), 1)
                k_val = round(float(np.random.uniform(10.0, 30.0)), 1)
                fert_tuple = fertilizer_map["Potassium"][np.random.choice(len(fertilizer_map["Potassium"]))]
                dose = round(float(np.random.uniform(40.0, 90.0)), 1)

            else:
                n_val = round(float(np.random.uniform(60.0, 100.0)), 1)
                p_val = round(float(np.random.uniform(40.0, 70.0)), 1)
                k_val = round(float(np.random.uniform(40.0, 70.0)), 1)
                fert_tuple = fertilizer_map["Balanced"][np.random.choice(len(fertilizer_map["Balanced"]))]
                dose = round(float(np.random.uniform(30.0, 60.0)), 1)

            rows.append({
                "Crop": crop,
                "Soil_Type": soil,
                "Nitrogen_N": n_val,
                "Phosphorous_P": p_val,
                "Potassium_K": k_val,
                "Temperature": temp,
                "Humidity": humidity,
                "Moisture": moisture,
                "Fertilizer_Name": fert_tuple[0],
                "Nutrient_Deficiency": def_type,
                "Recommended_Dose_kg_ha": dose
            })

    df = pd.DataFrame(rows)
    df.to_csv("c:/Users/Harshal/OneDrive/Desktop/agrisense/data/Fertilizer_prediction.csv", index=False)
    print(f"Fertilizer dataset saved successfully. Shape: {df.shape}")

if __name__ == "__main__":
    generate_fertilizer_csv()
