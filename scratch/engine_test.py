import os
import sys
import pandas as pd

project_root = r"c:\Users\Harshal\OneDrive\Desktop\agrisense"
os.chdir(project_root)
sys.path.insert(0, project_root)

print("=== VERIFYING ALL MODULE ENGINES EXECUTIONS ===")

# 1. Preprocessing
from preprocessing.pipeline import AgriculturalDataPreprocessor
df_raw = pd.read_csv("data/Crop_recommendation.csv")
processor = AgriculturalDataPreprocessor(df_raw)
df_clean, summary = processor.execute_pipeline()
print(f"[PASS] Preprocessing: Raw rows = {summary['initial_rows']}, Cleaned rows = {summary['final_rows']}")

# 2. Apriori
from association.apriori_engine import AgriculturalAprioriEngine
f_sets, rules = AgriculturalAprioriEngine.mine_rules(df_clean, min_support=0.05, min_confidence=0.5)
print(f"[PASS] Apriori: Itemsets = {len(f_sets)}, Rules = {len(rules)}")

# 3. WEKA J48 & Naive Bayes
from classification.weka_j48 import WekaJ48Classifier
from classification.naive_bayes import GaussianNBClassifier

weka_metrics = WekaJ48Classifier.run_j48(df_clean)
print(f"[PASS] J48 Accuracy: {weka_metrics['accuracy']}%, Java Weka Executed: {weka_metrics['weka_executed']}")

nb_clf = GaussianNBClassifier()
nb_metrics = nb_clf.train_and_evaluate(df_clean)
print(f"[PASS] Naive Bayes Accuracy: {nb_metrics['accuracy']:.4f}")

# 4. Regression
from regression.engine import DualRegressionEngine
re = DualRegressionEngine()
reg_metrics = re.train_and_evaluate_all()
print(f"[PASS] Crop Yield Regression (RF R2): {reg_metrics['crop_yield']['rf']['metrics']['r2']:.4f}")
print(f"[PASS] Soil Carbon Regression (RF R2): {reg_metrics['soil_carbon']['rf']['metrics']['r2']:.4f}")

# 5. Clustering
from clustering.engine import DualClusteringEngine
cle = DualClusteringEngine()
cl_metrics = cle.run_clustering(n_clusters_crop=4, n_clusters_soil=4)
print(f"[PASS] Crop Clustering Silhouette: {cl_metrics['crop']['silhouette']:.4f}")

# 6. Fertilizer Rec
from decision_support.fertilizer_engine import FertilizerRecommendationEngine
fe = FertilizerRecommendationEngine()
rec_res = fe.recommend_fertilizer(
    crop='rice',
    soil_inputs={'N': 90, 'P': 42, 'K': 43, 'ph': 6.5, 'soil_type': 'Clayey'},
    mined_rules=[]
)
_fert_name = rec_res.get('recommended_fertilizer') or rec_res.get('primary_fertilizer', {}).get('name', 'N/A')
_fert_score = rec_res.get('confidence_score_pct') or rec_res.get('primary_fertilizer', {}).get('confidence_score', 'N/A')
print(f"[PASS] Fertilizer Rec: {_fert_name} (Confidence score: {_fert_score})")

# 7. Decision Support
from decision_support.engine import DecisionSupportEngine
dse = DecisionSupportEngine()
ds_res = dse.analyze(n=90, p=42, k=43, temp=20.8, hum=82.0, ph=6.5, rain=202.9)
print(f"[PASS] Decision Support Recommended Crop (J48): {ds_res['j48_crop']}, (NB): {ds_res['nb_crop']}")

# 8. Market Engine
from market.engine import MarketIntelligenceEngine
me = MarketIntelligenceEngine()
price = me.get_crop_price('rice')
print(f"[PASS] Market Price for Rice: Rs.{price['Latest_Price_Per_Quintal']}/Quintal")

# 9. Provider Directory
from market.provider import ProviderDirectoryEngine
pde = ProviderDirectoryEngine()
provs = pde.get_providers(district_zone='Pune District')
print(f"[PASS] Providers in Pune District: {len(provs)}")

# 10. Location Intelligence Engine
from location.engine import LocationIntelligenceEngine
df_loc = pd.read_csv("data/Crop_recommendation.csv")
loc_res = LocationIntelligenceEngine.generate_location_report(
    df_crop_raw=df_loc,
    state='Maharashtra',
    district='Pune',
    soil_inputs={'N': 90, 'P': 42, 'K': 43, 'temperature': 20.8, 'humidity': 82.0, 'ph': 6.5, 'rainfall': 202.9},
    farm_inputs={'land_area_ha': 2.5, 'soil_type': 'Clayey'}
)
print(f"[PASS] Location Rec Crop: {loc_res['recommended_crop']}")
print(f"[PASS] Gross Revenue: Rs.{loc_res['economic_model']['estimated_gross_revenue']:,.2f}")
print(f"[PASS] Estimated Gross Return: Rs.{loc_res['economic_model']['estimated_gross_return']:,.2f}")
