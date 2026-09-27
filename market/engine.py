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
