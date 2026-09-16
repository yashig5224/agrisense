# AgriSense — Smart Agriculture Decision Support System Using Data Mining

AgriSense is an enterprise-grade agricultural decision support system leveraging data mining algorithms, machine learning models, and soil microclimate analytics to optimize crop recommendations, soil nutrient balancing, and regional farming strategies.

---

## 1. Project Overview

Agriculture faces unprecedented challenges due to climate variability, soil degradation, and unbalanced fertilizer usage. **AgriSense** provides a unified data mining framework that transforms raw soil sensors, microclimate telemetry, and regional land records into actionable agronomic intelligence.

---

## 2. Problem Statement

Modern farmers and agricultural extension officers often lack actionable, data-driven tools to determine:
1. Which crops are best suited for specific soil nutrient levels (N, P, K, pH) and microclimate conditions (temperature, humidity, rainfall).
2. How environmental factors co-occur to influence crop growth patterns.
3. How to estimate expected crop yield outputs quantitatively.
4. How to segment agricultural land into uniform fertility zones for targeted management.

AgriSense addresses these challenges by integrating descriptive, predictive, and prescriptive data mining techniques.

---

## 3. Core Objectives

- Develop a modular Python + Streamlit analytics dashboard enforcing a clean, professional agricultural aesthetic.
- Implement robust data preprocessing (null imputation, duplicate removal, invalid value fixing, IQR outlier analysis, and feature scaling).
- Mine frequent co-occurrence patterns using the **Apriori Algorithm** (`mlxtend`).
- Perform crop classification using **WEKA's J48 Decision Tree Classifier** (C4.5) and **Gaussian Naive Bayes**.
- Build dual-dataset regression models to predict crop yields ($R^2$, RMSE, MAE, MSE).
- Segment agricultural land using **K-Means Clustering** and **2D PCA Visualization**.
- Deliver an interactive **Decision Support Engine** providing multi-model advisory predictions and domain interpretations.

---

## 4. System Architecture

```text
                                  +---------------------------------------+
                                  |     AgriSense Executive Dashboard     |
                                  +-------------------+-------------------+
                                                      |
         +------------------+-------------------------+-------------------------+------------------+
         |                  |                         |                         |                  |
+--------v-------+  +-------v--------+       +--------v-------+        +--------v-------+ +--------v-------+
|  Preprocessing |  | Association    |       | Classification |        | Regression     | | Clustering     |
|  Pipeline      |  | Rules (Apriori)|       | (J48 & Bayes)  |        | (Dual Dataset) | | (K-Means & PCA)|
+--------+-------+  +-------+--------+       +--------+-------+        +--------+-------+ +--------+-------+
         |                  |                         |                         |                  |
         +------------------+-------------------------+-------------------------+------------------+
                                                      |
                                          +-----------v-----------+
                                          | Decision Support      |
                                          | Multi-Model Inference |
                                          +-----------------------+
```

---

## 5. Datasets

1. **Crop Recommendation Dataset** (`data/Crop_recommendation.csv`):
   - **Records**: 1,115 agricultural field samples.
   - **Attributes**: Nitrogen (`N`), Phosphorus (`P`), Potassium (`K`), `temperature`, `humidity`, `ph`, `rainfall`, and `label` (22 crop classes including rice, maize, cotton, coffee, jute, etc.).
2. **Soil Organic Carbon Dataset** (`data/Soil_Organic_Carbon.csv`):
   - **Records**: 600 soil records.
   - **Attributes**: `Clay_Pct`, `Moisture_Pct`, `Nitrogen_N`, `Bulk_Density`, `Soil_pH`, `Temperature_C`, `Organic_Carbon_Pct`.
3. **Soil & Water Quality Dataset** (`data/Soil_Water_Quality.csv`):
   - **Records**: 550 irrigation samples.
   - **Attributes**: `EC_dS_m`, `Moisture_Pct`, `pH`, `Sodium_Adsorption_Ratio`, `Nitrate_PPM`, `Organic_Matter_Pct`.

---

## 6. Preprocessing Techniques

Implemented in `preprocessing/pipeline.py` via `AgriculturalDataPreprocessor`:
- **Missing Value Imputation**: Column-wise median or mean imputation.
- **Duplicate Removal**: Identification and removal of exact row copies.
- **Invalid Value Audit**: Out-of-bounds pH (0–14) boundary correction.
- **IQR Outlier Detection**: Upper and lower bound filtering ($Q_1 - 1.5 \times \text{IQR}$, $Q_3 + 1.5 \times \text{IQR}$).
- **Descriptive Statistics**: Real calculation of mean, std, min, median, max, skewness, and kurtosis.
- **Feature Scaling**: `StandardScaler` (Z-Score) and `MinMaxScaler` normalization.

---

## 7. Data Mining Modules

### 7.1. Association Rule Mining (Apriori)
- Continuous attributes are discretized into agronomic intervals (`N_High`, `Temp_Moderate`, `pH_Acidic`, `Rain_Heavy`, `Crop_rice`).
- Mined via `mlxtend.frequent_patterns.apriori` and `association_rules`.
- Evaluates Support, Confidence, Lift, Leverage, and Conviction.

### 7.2. WEKA J48 Decision Tree Classification
- Formats raw data into Weka ARFF (`data/crop_train.arff`, `data/crop_test.arff`).
- Invokes WEKA's J48 decision tree classifier via Java CLI (`weka.classifiers.trees.J48`), parsing decision tree rules, accuracy, correctly/incorrectly classified counts, and confusion matrices.
- Features a pure Python C4.5 tree fallback engine when `weka.jar` is missing.

### 7.3. Gaussian Naive Bayes Classification
- Probabilistic classifier trained on numeric soil/climate attributes using `sklearn.naive_bayes.GaussianNB`.
- Identical train/test split parameter (`random_state=42`, `stratify=y`) as WEKA J48 for factual comparison.
- Provides interactive single-sample prediction for live parameter inputs.

### 7.4. Dual-Dataset Regression
- Evaluates `RandomForestRegressor`, `GradientBoostingRegressor`, `LinearRegression`, and `Ridge`.
- Computes MAE, MSE, RMSE, and $R^2$.
- Renders actual vs predicted scatter plots with $y=x$ reference lines and residual error analysis ($y_{actual} - y_{pred}$).

### 7.5. Dual-Dataset K-Means Clustering & PCA
- Normalizes features using `StandardScaler`.
- Performs K-Means clustering for $K \in [2, 8]$, evaluating Inertia (SSE) and Silhouette Scores.
- Visualizes high-dimensional feature clusters using **2D PCA Projection** ($PC_1, PC_2$).
- Computes un-scaled cluster profile tables for post-hoc domain interpretation.

### 7.6. Decision Support Advisory Workflow
- Multi-model inference engine taking live soil and microclimate inputs (`N`, `P`, `K`, `temperature`, `humidity`, `pH`, `rainfall`).
- Concurrently queries WEKA J48, Naive Bayes, Apriori rules, K-Means cluster assigner, and yield regressor to generate an integrated agronomic advisory report.

---

## 8. Technology Stack

- **Language**: Python 3.9+
- **Web UI Framework**: Streamlit
- **Data Engineering**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, mlxtend
- **Data Mining Integration**: WEKA Java CLI (`weka.jar` / `weka.classifiers.trees.J48`)
- **Data Visualization**: Plotly, Matplotlib

---

## 9. Installation & WEKA Setup

### 9.1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 9.2. WEKA Environment Setup (Optional for Native Java Execution)
1. Install Java JRE (Java 8 or higher).
2. Download `weka.jar` from the [Official WEKA Download Page](https://waikato.github.io/weka-wiki/downloading_weka/).
3. Place `weka.jar` in your project root directory or set the environment variable `WEKA_JAR`:
   ```bash
   set WEKA_JAR=C:\path\to\weka.jar
   ```
*(Note: If `weka.jar` is not present, AgriSense automatically runs the built-in J48/C4.5 Python engine).*

---

## 10. Running the Application

Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

---

## 11. Complete Directory Structure

```
agrisense/
├── .streamlit/
│   └── config.toml                  # Light theme & agricultural palette configuration
├── data/
│   ├── loader.py                    # Multi-dataset loader module
│   ├── Crop_recommendation.csv      # Primary agricultural telemetry dataset
│   ├── Soil_Organic_Carbon.csv      # Soil organic carbon regression dataset
│   └── Soil_Water_Quality.csv       # Soil & water quality clustering dataset
├── preprocessing/
│   ├── pipeline.py                  # AgriculturalDataPreprocessor engine
│   └── ui.py                        # 5-tab Data Preprocessing & Quality Engineering UI
├── association/
│   ├── apriori_engine.py            # Categorical binning & mlxtend Apriori engine
│   └── ui.py                        # Rule mining UI, filters, charts & agronomic insights
├── classification/
│   ├── arff_generator.py            # Weka ARFF dataset exporter
│   ├── weka_j48.py                  # Weka J48 Decision Tree engine & C4.5 fallback
│   ├── naive_bayes.py               # Gaussian Naive Bayes & single-sample predictor
│   └── ui.py                        # J48, Naive Bayes, Factual Comparison & Live Predictor UI
├── regression/
│   ├── engine.py                    # Dual-dataset regression engine (MAE, MSE, RMSE, R²)
│   └── ui.py                        # Dual-dataset regression UI, residual & actual-vs-pred plots
├── clustering/
│   ├── engine.py                    # Dual-dataset K-Means (K=2-8) & PCA 2D engine
│   └── ui.py                        # Dual-dataset clustering UI, 2D PCA scatter & profile tables
├── decision_support/
│   ├── engine.py                    # Integrated multi-model decision advisory engine
│   └── ui.py                        # Prescriptive DSS advisory & fertilizer planning UI
├── visualization/
│   └── charts.py                    # Agricultural Plotly chart helpers
├── utils/
│   ├── theme.py                     # Light mode custom CSS theme injector
│   └── helpers.py                   # Metric cards & section header functions
├── app.py                           # Streamlit main dashboard entry point
├── requirements.txt                 # Project dependencies
└── README.md                        # Technical documentation
```

---

## 12. Project Limitations & Future Scope

### Limitations
- Predictions are generated from historical dataset distributions and should be validated against physical soil sample testing.
- WEKA CLI execution requires Java runtime setup on host operating system.

### Future Scope
- **Real-Time IoT Sensor Ingestion**: Connecting live LoRaWAN soil moisture and NPK sensors.
- **Deep Learning Image Analytics**: Incorporating CNN models for crop leaf disease detection.
- **GIS Geospatial Mapping**: Integrating interactive Leaflet / Mapbox GIS maps for spatial fertility zoning.
#   a g r i s e n s e  
 