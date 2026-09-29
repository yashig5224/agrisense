import os
import sys
import pandas as pd

project_root = r"c:\Users\Harshal\OneDrive\Desktop\agrisense"
os.chdir(project_root)
sys.path.insert(0, project_root)

print("=== 1. TESTING CENTRAL INDIAN DATA PROVIDER ===")
from data.india.provider import IndianAgriculturalDataProvider

states = IndianAgriculturalDataProvider.get_states()
print(f"[PASS] Supported Indian States ({len(states)}): {states}")

pune_districts = IndianAgriculturalDataProvider.get_districts("Maharashtra")
print(f"[PASS] Maharashtra Districts ({len(pune_districts)}): {pune_districts}")

pune_info = IndianAgriculturalDataProvider.get_district_info("Maharashtra", "Pune")
print(f"[PASS] Pune Agro-Climatic Zone: {pune_info['Agro_Climatic_Zone']}")

pune_soil = IndianAgriculturalDataProvider.get_soil_profile("Maharashtra", "Pune")
print(f"[PASS] Pune Soil Health Card Profile: Available N={pune_soil['Avg_Available_N_kg_ha']} kg/ha ({pune_soil['N_Status']}), pH={pune_soil['Mean_pH']}")

pune_mandi = IndianAgriculturalDataProvider.get_crop_mandi_info("rice", state="Maharashtra", district="Pune")
print(f"[PASS] Agmarknet Mandi Price for Rice in Pune: INR {pune_mandi['latest_price']:,.2f}/quintal ({pune_mandi['market_apmc']})")

pune_dealers, d_count = IndianAgriculturalDataProvider.get_retail_dealers(state="Maharashtra", district="Pune")
print(f"[PASS] Authorized Retail Fertilizer Dealers in Pune: {d_count} licensed POS records")

rice_seeds = IndianAgriculturalDataProvider.get_seed_varieties("rice")
print(f"[PASS] Certified Rice Varieties in India ({len(rice_seeds)} varieties found): {list(rice_seeds['Seed_Variety'].unique())}")

urea_details = IndianAgriculturalDataProvider.get_fertilizer_details("Urea")
print(f"[PASS] Statutory Urea Price (DoF): INR {urea_details['Statutory_Max_Retail_Price_INR_50kg']}/50kg bag ({urea_details['Subsidized']})")

print("\n=== 2. TESTING REFACTORED MARKET DATA PROVIDER ===")
from market.provider import MarketDataProvider
crop_info = MarketDataProvider.get_crop_market_info("cotton", state="Maharashtra", district="Nagpur")
print(f"[PASS] Cotton Mandi Modal Price in Nagpur: INR {crop_info['latest_price']}/quintal ({crop_info['market_apmc']})")

print("\n=== 3. TESTING INDIAN LOCATION INTELLIGENCE REPORT ===")
from location.engine import LocationIntelligenceEngine
from data.loader import load_crop_dataset
df_raw = load_crop_dataset()

report = LocationIntelligenceEngine.generate_location_report(
    df_raw,
    state="Maharashtra",
    district="Pune",
    soil_inputs={"N": 90.0, "P": 42.0, "K": 43.0, "temperature": 24.5, "humidity": 78.0, "ph": 7.4, "rainfall": 195.0},
    farm_inputs={"land_area_ha": 2.5, "seed_qty_kg_ha": 25.0, "fert_qty_kg_ha": 150.0, "operational_rate_per_ha": 4000.0}
)
print(f"[PASS] Location Rec Crop: {report['recommended_crop']}")
print(f"[PASS] Seed Variety: {report['seed_info']['name']}")
print(f"[PASS] Fertilizer Recommended: {report['fertilizer_info']['recommendation']['recommended_fertilizer']} (Statutory MRP: INR {report['fertilizer_info']['bag_price_50kg']}/bag)")
print(f"[PASS] Authorized Dealers Count: {report['dealers_info']['dealer_count']} ({report['dealers_info']['scope']})")
print(f"[PASS] Mandi Price: INR {report['economic_model']['mkt_price_per_quintal']}/quintal")
print(f"[PASS] Total Input Cost: INR {report['economic_model']['total_input_cost']:,.2f}")
print(f"[PASS] Estimated Gross Revenue: INR {report['economic_model']['estimated_gross_revenue']:,.2f}")
print(f"[PASS] Estimated Gross Return: INR {report['economic_model']['estimated_gross_return']:,.2f}")


print("\n=== 4. TESTING ALL APP UI IMPORTS ===")
from preprocessing.ui import render_data_explorer_page
from analytics.ui import render_analytics_workspace
from decision_support.ui import render_decision_intelligence_page
from market.ui import render_market_intelligence_page
from location.ui import render_location_intelligence_page
print("[PASS] All UI modules cleanly imported without errors.")

print("\n=== ALL INDIAN DATA LAYER VERIFICATIONS PASSED SUCCESSFULLY ===")
