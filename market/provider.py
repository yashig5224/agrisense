"""
Decoupled Market Data Provider for AgriSense (Version 2).
Refactored to integrate with official Indian Agricultural Data Provider (Agmarknet, DoF, NSC).
Maintains full backward compatibility while serving genuine Indian market prices and licensed dealers.
"""

import pandas as pd
import os
from typing import Dict, Any, List, Tuple
from data.india.provider import IndianAgriculturalDataProvider

class MarketDataProvider:
    """Decoupled market data provider interface backed by Indian Agricultural Data Provider."""

    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    @classmethod
    def get_crop_prices_df(cls) -> pd.DataFrame:
        """Load crop commodity market prices from Indian Mandi registry."""
        return IndianAgriculturalDataProvider.get_mandi_prices_df()

    @classmethod
    def get_providers_df(cls) -> pd.DataFrame:
        """Load verified Indian district fertilizer dealers and retailers."""
        df_dealers, _ = IndianAgriculturalDataProvider.get_retail_dealers()
        return df_dealers

    @classmethod
    def get_crop_market_info(cls, crop: str, state: str = None, district: str = None) -> Dict[str, Any]:
        """Query commodity market price and trend for a selected crop from Indian Agmarknet Mandis."""
        return IndianAgriculturalDataProvider.get_crop_mandi_info(crop, state=state, district=district)

    @classmethod
    def get_seed_prices(cls, crop: str, state: str = None) -> pd.DataFrame:
        """Query certified seed varieties and availability for a crop in India."""
        return IndianAgriculturalDataProvider.get_seed_varieties(crop, state=state)

    @classmethod
    def get_fertilizer_market_data(cls, fertilizer_name: str) -> pd.DataFrame:
        """Query statutory pricing and packaging for Indian fertilizers."""
        df_fert = IndianAgriculturalDataProvider.get_fertilizers_df()
        if df_fert.empty:
            return pd.DataFrame()

        fert_clean = str(fertilizer_name).strip().lower()
        matched = df_fert[df_fert["Fertilizer_Name"].str.strip().str.lower().str.contains(fert_clean[:4])].copy()
        if not matched.empty:
            return matched.reset_index(drop=True)
        return df_fert.reset_index(drop=True)

    @classmethod
    def get_nearby_providers(cls, district: str, state: str = None) -> Tuple[pd.DataFrame, int]:
        """
        Search authorized fertilizer retailers / PACS by Indian District.
        Strictly provides district-level dealer information without fabricating GPS proximity.
        """
        return IndianAgriculturalDataProvider.get_retail_dealers(state=state, district=district)

    @classmethod
    def get_states(cls) -> List[str]:
        """Get Indian states list."""
        return IndianAgriculturalDataProvider.get_states()

    @classmethod
    def get_districts(cls, state: str) -> List[str]:
        """Get districts for Indian state."""
        return IndianAgriculturalDataProvider.get_districts(state)


class ProviderDirectoryEngine:
    """Compatibility wrapper for the fertilizer/seed provider directory.

    Exposes ``get_providers(district_zone)`` returning a list of provider
    dicts, as expected by engine_test.py.
    """

    def get_providers(self, district_zone: str, state: str = None) -> List[Dict[str, Any]]:
        """Return a list of provider records for the given district zone.

        Parameters
        ----------
        district_zone:
            District name or zone string (e.g. ``'Pune District'``).
        state:
            Optional Indian state name to narrow the search.

        Returns
        -------
        list[dict]
            Each element is a dict representation of one provider row.
        """
        # Strip trailing 'District' / 'Zone' suffixes for cleaner matching.
        clean_district = district_zone.replace(" District", "").replace(" Zone", "").strip()
        df_providers, _count = MarketDataProvider.get_nearby_providers(
            district=clean_district, state=state
        )
        if df_providers is None or (hasattr(df_providers, "empty") and df_providers.empty):
            return []
        return df_providers.to_dict(orient="records")

