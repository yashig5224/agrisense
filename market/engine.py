"""
Agricultural Input-Cost Calculator Engine for AgriSense (Version 2).
Computes farm input costs (seed, fertilizer, operational expenses) dynamically.
"""

from typing import Dict, Any

class AgriculturalMarketEngine:
    """Input-cost budgeting calculator engine."""

    @classmethod
    def calculate_input_costs(
        cls,
        land_area_ha: float,
        seed_qty_kg_ha: float,
        seed_price_per_kg: float,
        fert_qty_kg_ha: float,
        fert_price_per_bag_50kg: float,
        additional_ops_cost_per_ha: float = 2500.0
    ) -> Dict[str, Any]:
        """
        Calculate total estimated farm input costs.
        
        Formula:
        Seed Cost = Land Area * Seed Qty * Seed Price
        Fertilizer Cost = Land Area * Fertilizer Qty * (Fertilizer Price per Bag / 50)
        Additional Ops Cost = Land Area * Operational Rate
        Total Cost = Seed Cost + Fertilizer Cost + Operational Cost
        """
        land_area = max(0.1, float(land_area_ha))
        seed_qty = max(0.0, float(seed_qty_kg_ha))
        seed_price = max(0.0, float(seed_price_per_kg))
        fert_qty = max(0.0, float(fert_qty_kg_ha))
        fert_price_bag = max(0.0, float(fert_price_per_bag_50kg))
        ops_rate = max(0.0, float(additional_ops_cost_per_ha))

        # Costs
        seed_cost = round(land_area * seed_qty * seed_price, 2)
        fert_cost_per_kg = fert_price_bag / 50.0
        fert_cost = round(land_area * fert_qty * fert_cost_per_kg, 2)
        ops_cost = round(land_area * ops_rate, 2)
        total_cost = round(seed_cost + fert_cost + ops_cost, 2)

        cost_per_ha = round(total_cost / land_area, 2) if land_area > 0 else 0.0

        return {
            "land_area_ha": land_area,
            "seed_cost": seed_cost,
            "fertilizer_cost": fert_cost,
            "operational_cost": ops_cost,
            "total_estimated_input_cost": total_cost,
            "estimated_cost_per_hectare": cost_per_ha,
            "disclaimer": "This is an input-cost estimation calculator for budgeting support, not a guaranteed profit predictor."
        }


class MarketIntelligenceEngine:
    """Compatibility wrapper for market intelligence queries.

    Delegates to MarketDataProvider (which uses IndianAgriculturalDataProvider)
    and normalises the crop price response for engine_test.py consumers.
    """

    def get_crop_price(self, crop: str, state: str = None, district: str = None) -> Dict[str, Any]:
        """Return the latest mandi price for a crop as a flat dict.

        Result includes ``Latest_Price_Per_Quintal`` and metadata fields from
        the underlying Agmarknet dataset.
        """
        from market.provider import MarketDataProvider
        raw = MarketDataProvider.get_crop_market_info(crop, state=state, district=district)

        # raw may be a dict with 'modal_price', 'min_price', 'max_price' etc.
        # Normalise to the key expected by engine_test.py.
        price_val = (
            raw.get("modal_price")
            or raw.get("Latest_Price_Per_Quintal")
            or raw.get("price")
            or 0.0
        )
        return {
            "crop": crop,
            "Latest_Price_Per_Quintal": price_val,
            "state": raw.get("state", state),
            "district": raw.get("district", district),
            "source": raw.get("source", "Agmarknet / IndianAgriculturalDataProvider"),
        }

    def calculate_input_costs(self, **kwargs) -> Dict[str, Any]:
        """Delegate to AgriculturalMarketEngine.calculate_input_costs."""
        return AgriculturalMarketEngine.calculate_input_costs(**kwargs)

