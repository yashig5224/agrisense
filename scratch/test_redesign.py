import os
import sys
import pandas as pd

project_root = r"c:\Users\Harshal\OneDrive\Desktop\agrisense"
os.chdir(project_root)
sys.path.insert(0, project_root)

print("=== TESTING REDESIGNED AGRISENSE MODULE IMPORTS ===")

from data.loader import load_crop_dataset
df_raw = load_crop_dataset()
print(f"[PASS] Data loader: Raw records = {len(df_raw)}")

from preprocessing.ui import render_data_explorer_page
print("[PASS] Imported preprocessing.ui (Data Explorer)")

from analytics.ui import render_analytics_workspace
print("[PASS] Imported analytics.ui (Analytics Workspace)")

from decision_support.ui import render_decision_intelligence_page
print("[PASS] Imported decision_support.ui (Decision Intelligence)")

from market.ui import render_market_intelligence_page
print("[PASS] Imported market.ui (Market Intelligence)")

from location.ui import render_location_intelligence_page
print("[PASS] Imported location.ui (Location Intelligence)")

print("=== ALL REDESIGNED MODULES VERIFIED SUCCESSFULLY ===")
