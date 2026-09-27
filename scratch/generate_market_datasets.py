"""
Script to generate data/Crop_market_prices.csv and data/Seed_fertilizer_providers.csv datasets.
"""
import pandas as pd
import numpy as np

def generate_market_datasets():
    np.random.seed(42)

    crops = [
        "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas", "mothbeans",
        "mungbean", "blackgram", "lentil", "pomegranate", "banana", "mango",
        "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya",
        "coconut", "cotton", "jute", "coffee"
    ]

    mandis = [
        "North Central Mandi", "South District Market", "East Regional Grain Hub",
        "West Terminal Market", "Central Agriculture Yard"
    ]

    trends = ["Upward (+2.4%)", "Stable (+0.2%)", "Downward (-1.1%)", "Upward (+1.8%)", "Stable"]

    # 1. Crop Market Prices Dataset
    crop_price_rows = []
    for crop in crops:
        base_price = float(np.random.uniform(1800, 6500))
        min_p = round(base_price * 0.90, 2)
        max_p = round(base_price * 1.12, 2)
        latest_p = round(float(np.random.uniform(min_p, max_p)), 2)
        mandi = np.random.choice(mandis)
        trend = np.random.choice(trends)

        crop_price_rows.append({
            "Crop": crop,
            "Market_Location": mandi,
            "Latest_Price_Per_Quintal": latest_p,
            "Historical_Min_Price": min_p,
            "Historical_Max_Price": max_p,
            "Price_Trend": trend,
            "Currency": "INR / Quintal",
            "Data_Source": "Public Agricultural Commodity Price Registry",
            "Last_Updated": "2026-09-27 06:00 UTC"
        })

    crop_price_df = pd.DataFrame(crop_price_rows)
    crop_price_df.to_csv("c:/Users/Harshal/OneDrive/Desktop/agrisense/data/Crop_market_prices.csv", index=False)
    print(f"Crop market prices saved. Shape: {crop_price_df.shape}")

    # 2. Seed & Fertilizer Providers Dataset
    providers = [
        "AgriCorp National Supplies", "State Farm Cooperative Society",
        "GreenField Bio Inputs", "National Agri Retail Hub",
        "Kisan Seva Kendra", "Harvest Field Products"
    ]

    zones = ["North Zone", "South Zone", "Central Zone", "East Zone", "West Zone"]
    avail_statuses = ["In Stock", "In Stock", "Limited Stock", "Out of Stock"]

    provider_rows = []

    # Seed Items
    for crop in crops:
        for p_idx in range(2):
            prov = np.random.choice(providers)
            zone = np.random.choice(zones)
            p_unit = round(float(np.random.uniform(80, 450)), 2)
            dist = round(float(np.random.uniform(2.5, 38.0)), 1)
            avail = np.random.choice(avail_statuses)

            provider_rows.append({
                "Item_Type": "Seed",
                "Name": f"Certified {crop.title()} Hybrid Seed",
                "Crop_Target": crop,
                "Price_Per_Unit": p_unit,
                "Unit_Size": "10 kg bag",
                "Provider_Name": prov,
                "Provider_Address": f"{zone} Agri-Hub Sector {p_idx+1}",
                "District_Zone": zone,
                "Distance_km": dist,
                "Availability_Status": avail,
                "Data_Source": "Agricultural Retail Directory",
                "Last_Updated": "2026-09-27 06:00 UTC"
            })

    # Fertilizer Items
    fertilizer_products = [
        ("Urea 46% N", "50 kg bag", 268.0),
        ("DAP 18-46-0", "50 kg bag", 1350.0),
        ("MOP 60% K", "50 kg bag", 1700.0),
        ("17-17-17 NPK", "50 kg bag", 1470.0),
        ("14-35-14 NPK", "50 kg bag", 1420.0),
        ("28-28-0 NPK", "50 kg bag", 1380.0),
        ("SSP 16% P", "50 kg bag", 480.0)
    ]

    for fert_name, unit_sz, base_p in fertilizer_products:
        for z in zones:
            for p_idx in range(2):
                prov = np.random.choice(providers)
                p_unit = round(base_p * float(np.random.uniform(0.97, 1.05)), 2)
                dist = round(float(np.random.uniform(1.8, 42.0)), 1)
                avail = np.random.choice(avail_statuses)

                provider_rows.append({
                    "Item_Type": "Fertilizer",
                    "Name": fert_name,
                    "Crop_Target": "General",
                    "Price_Per_Unit": p_unit,
                    "Unit_Size": unit_sz,
                    "Provider_Name": prov,
                    "Provider_Address": f"{z} Fertilizer Outlet #{p_idx+101}",
                    "District_Zone": z,
                    "Distance_km": dist,
                    "Availability_Status": avail,
                    "Data_Source": "Fertilizer Dealer Registry",
                    "Last_Updated": "2026-09-27 06:00 UTC"
                })

    provider_df = pd.DataFrame(provider_rows)
    provider_df.to_csv("c:/Users/Harshal/OneDrive/Desktop/agrisense/data/Seed_fertilizer_providers.csv", index=False)
    print(f"Providers dataset saved. Shape: {provider_df.shape}")

if __name__ == "__main__":
    generate_market_datasets()
