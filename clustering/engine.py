"""
Agricultural Clustering Engine for Phase 7.
Handles Dataset 1 (Crop & Microclimate Telemetry) and Dataset 2 (Soil & Water Quality),
scales features using StandardScaler, runs K-Means for configurable K (2-8),
computes Inertia and Silhouette scores, generates 2D PCA visual projections,
and constructs un-scaled cluster profile tables for post-hoc interpretation.
"""

import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from typing import Dict, Any, Tuple

class AgriculturalClusteringEngine:
    """Engine for multi-dataset K-Means clustering and PCA visualization."""

    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    @classmethod
    def load_clustering_dataset(cls, dataset_key: str) -> Tuple[pd.DataFrame, list]:
        """Load clustering dataset and return (df, numeric_features)."""
        if dataset_key == "Dataset 1: Crop & Microclimate Telemetry":
            path = os.path.join(cls.DATA_DIR, "Crop_recommendation.csv")
            df = pd.read_csv(path)
            features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
            return df, features
        else: # Dataset 2
            path = os.path.join(cls.DATA_DIR, "Soil_Water_Quality.csv")
            df = pd.read_csv(path)
            features = ["EC_dS_m", "Moisture_Pct", "pH", "Sodium_Adsorption_Ratio", "Nitrate_PPM", "Organic_Matter_Pct"]
            return df, features

    @classmethod
    def execute_kmeans(
        cls,
        df: pd.DataFrame,
        feature_cols: list,
        n_clusters: int = 4,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Perform StandardScaler normalization, K-Means clustering, PCA, and profile calculation."""
        clean_df = df.dropna(subset=feature_cols).copy()
        X = clean_df[feature_cols]

        # 1. Feature Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 2. K-Means Execution
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        clean_df["Cluster_ID"] = [f"Cluster {i+1}" for i in cluster_labels]

        # 3. Metrics
        inertia = float(kmeans.inertia_)
        sil_score = float(silhouette_score(X_scaled, cluster_labels)) if len(np.unique(cluster_labels)) > 1 else 0.0

        # 4. PCA 2D Dimensionality Reduction (Strictly for 2D Visual Mapping)
        pca = PCA(n_components=2, random_state=random_state)
        pca_coords = pca.fit_transform(X_scaled)
        var_explained = (pca.explained_variance_ratio_ * 100).round(2)

        pca_df = pd.DataFrame({
            "PC1": pca_coords[:, 0],
            "PC2": pca_coords[:, 1],
            "Cluster_ID": clean_df["Cluster_ID"].values
        })

        # 5. Cluster Profile Table (Un-scaled original feature means)
        profile_df = clean_df.groupby("Cluster_ID")[feature_cols].mean().round(2).reset_index()
        counts = clean_df["Cluster_ID"].value_counts().reset_index()
        counts.columns = ["Cluster_ID", "Sample Count"]
        profile_df = profile_df.merge(counts, on="Cluster_ID").sort_values(by="Cluster_ID").reset_index(drop=True)

        return {
            "n_clusters": n_clusters,
            "inertia": round(inertia, 2),
            "silhouette_score": round(sil_score, 3),
            "clustered_df": clean_df,
            "pca_df": pca_df,
            "pca_variance_explained": var_explained,
            "cluster_profiles": profile_df,
            "scaled_centroids": kmeans.cluster_centers_
        }

    @staticmethod
    def generate_cluster_interpretations(profile_df: pd.DataFrame, feature_cols: list) -> list:
        """Interpret cluster profiles based on calculated mean values."""
        insights = []
        for _, row in profile_df.iterrows():
            cid = row["Cluster_ID"]
            cnt = row["Sample Count"]

            highlights = []
            for col in feature_cols:
                val = row[col]
                highlights.append(f"{col}: {val}")

            insight = f"**{cid}** ({cnt} records): Characterized by mean values -> " + ", ".join(highlights[:4]) + "."
            insights.append(insight)
        return insights


class DualClusteringEngine:
    """Compatibility wrapper that runs K-Means on both agricultural datasets
    and returns results in the shape expected by engine_test.py:

        {
            'crop':  {'silhouette': float, 'inertia': float, 'n_clusters': int, ...},
            'soil':  {'silhouette': float, ...},
        }
    """

    def run_clustering(
        self,
        n_clusters_crop: int = 4,
        n_clusters_soil: int = 4,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Execute K-Means clustering on both datasets and return combined results."""
        results: Dict[str, Any] = {}

        dataset_map = {
            "crop": ("Dataset 1: Crop & Microclimate Telemetry", n_clusters_crop),
            "soil": ("Dataset 2: Soil & Water Quality", n_clusters_soil),
        }

        for key, (ds_label, n_k) in dataset_map.items():
            df, features = AgriculturalClusteringEngine.load_clustering_dataset(ds_label)
            raw = AgriculturalClusteringEngine.execute_kmeans(
                df, feature_cols=features, n_clusters=n_k, random_state=random_state
            )
            results[key] = {
                "silhouette": raw["silhouette_score"],
                "inertia": raw["inertia"],
                "n_clusters": raw["n_clusters"],
                "cluster_profiles": raw["cluster_profiles"],
                "pca_df": raw["pca_df"],
                "pca_variance_explained": raw["pca_variance_explained"],
                "clustered_df": raw["clustered_df"],
            }

        return results

