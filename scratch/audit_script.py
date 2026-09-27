import os
import sys
import glob
import pandas as pd
import numpy as np

project_root = r"c:\Users\Harshal\OneDrive\Desktop\agrisense"
os.chdir(project_root)
sys.path.insert(0, project_root)

print("=== 1. PROJECT STRUCTURE & IMPORTS CHECK ===")
py_files = glob.glob("**/*.py", recursive=True)
print(f"Total Python files: {len(py_files)}")

imported_modules = []
broken_imports = []

for py_f in py_files:
    if "venv" in py_f or ".git" in py_f:
        continue
    with open(py_f, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, l in enumerate(lines):
            l_str = l.strip()
            if l_str.startswith("import ") or l_str.startswith("from "):
                pass

print("\n=== 3. DATASETS PROVENANCE CHECK ===")
data_files = glob.glob("data/*.csv")
for df_path in data_files:
    try:
        df = pd.read_csv(df_path)
        print(f"File: {df_path} | Rows: {len(df)} | Cols: {len(df.columns)}")
        print(f"  Cols: {list(df.columns)}")
    except Exception as e:
        print(f"File: {df_path} | Error reading: {e}")

print("\n=== DATA CONSISTENCY CHECK ===")
if os.path.exists("data/Crop_recommendation.csv") and os.path.exists("data/Crop_market_prices.csv"):
    crop_rec = pd.read_csv("data/Crop_recommendation.csv")
    crop_market = pd.read_csv("data/Crop_market_prices.csv")
    rec_crops = set(crop_rec['label'].str.lower().unique()) if 'label' in crop_rec.columns else set()
    market_crops = set(crop_market['crop_name'].str.lower().unique()) if 'crop_name' in crop_market.columns else set()
    missing_in_market = rec_crops - market_crops
    print(f"Crops in Crop_recommendation but NOT in Crop_market_prices: {missing_in_market}")

if os.path.exists("data/Seed_fertilizer_providers.csv"):
    prov = pd.read_csv("data/Seed_fertilizer_providers.csv")
    if 'item_name' in prov.columns and 'item_category' in prov.columns:
        prov_seeds = set(prov[prov['item_category']=='Seed']['item_name'].str.lower().unique())
        prov_ferts = set(prov[prov['item_category']=='Fertilizer']['item_name'].str.lower().unique())
        print(f"Seeds in Providers CSV: {prov_seeds}")
        print(f"Fertilizers in Providers CSV: {prov_ferts}")
