"""
Decoupled Market Data Provider for AgriSense (Version 2).
Isolates commodity prices, seed varieties, fertilizer rates, and supplier directories.
"""

import pandas as pd
import os
from typing import Dict, Any, List

class MarketDataProvider:
    """Decoupled market data provider interface."""

    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    CROP_PRICES_PATH = os.path.join(DATA_DIR, "Crop_market_prices.csv")
    PROVIDERS_PATH = os.path.join(DATA_DIR, "Seed_fertilizer_providers.csv")

    @classmethod
    def get_crop_prices_df(cls) -> pd.DataFrame:
        """Load crop commodity market prices."""
        if os.path.exists(cls.CROP_PRICES_PATH):
            return pd.read_csv(cls.CROP_PRICES_PATH)
        return pd.DataFrame()

    @classmethod
    def get_providers_df(cls) -> pd.DataFrame:
        """Load seed and fertilizer suppliers directory."""
        if os.path.exists(cls.PROVIDERS_PATH):
            return pd.read_csv(cls.PROVIDERS_PATH)
        return pd.DataFrame()

    @classmethod
    def get_crop_market_info(cls, crop: str) -> Dict[str, Any]:
        """Query commodity market price and trend for a selected crop."""
        df = cls.get_crop_prices_df()
        if df.empty:
            return {"status": "NO_DATA"}

        crop_clean = str(crop).strip().lower()
        records = df[df["Crop"].str.strip().str.lower() == crop_clean]

        if records.empty:
            return {
                "status": "NOT_FOUND",
                "message": f"No historical market commodity data registered for '{crop}'."
            }

        row = records.iloc[0]
        return {
            "status": "SUCCESS",
            "crop": str(row["Crop"]).title(),
            "market_location": row["Market_Location"],
            "latest_price": float(row["Latest_Price_Per_Quintal"]),
            "historical_min": float(row["Historical_Min_Price"]),
            "historical_max": float(row["Historical_Max_Price"]),
            "price_trend": row["Price_Trend"],
            "currency": row["Currency"],
            "data_source": row["Data_Source"],
            "last_updated": row["Last_Updated"]
        }

    @classmethod
    def get_seed_prices(cls, crop: str) -> pd.DataFrame:
        """Query seed varieties and pricing for a crop."""
        df = cls.get_providers_df()
        if df.empty:
            return pd.DataFrame()

        crop_clean = str(crop).strip().lower()
        seed_df = df[(df["Item_Type"] == "Seed") & (df["Crop_Target"].str.strip().str.lower() == crop_clean)].copy()
        return seed_df.reset_index(drop=True)

    @classmethod
    def get_fertilizer_market_data(cls, fertilizer_name: str) -> pd.DataFrame:
        """Query pricing and packaging for a recommended fertilizer."""
        df = cls.get_providers_df()
        if df.empty:
            return pd.DataFrame()

        fert_clean = str(fertilizer_name).strip().lower()
        fert_df = df[(df["Item_Type"] == "Fertilizer") & (df["Name"].str.strip().str.lower().str.contains(fert_clean[:4]))].copy()
        return fert_df.reset_index(drop=True)

    @classmethod
    def get_nearby_providers(cls, district_zone: str) -> Tuple[pd.DataFrame, int]:
        """Search nearby suppliers by District Zone and count matching providers."""
        df = cls.get_providers_df()
        if df.empty:
            return pd.DataFrame(), 0

        zone_clean = str(district_zone).strip().lower()
        if zone_clean == "all":
            matched = df.copy()
        else:
            matched = df[df["District_Zone"].str.strip().str.lower() == zone_clean].copy()

        count = len(matched["Provider_Name"].unique()) if not matched.empty else 0
        return matched.sort_values(by="Distance_km").reset_index(drop=True), count
