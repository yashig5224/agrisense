"""
Agricultural Data Preprocessing Pipeline Module.
Implements robust end-to-end data cleaning, schema validation, outlier analysis,
feature scaling, label encoding, and dataset export.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Any, List

class AgriculturalDataPreprocessor:
    """Robust preprocessing pipeline for agricultural sensor, soil, and crop datasets."""

    NUMERIC_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    TARGET_COLUMN = "label"

    def __init__(self, df_raw: pd.DataFrame):
        self.df_raw = df_raw.copy()
        self.df_cleaned = df_raw.copy()
        self.label_encoder = LabelEncoder()
        self.scaler = None
        self.transformation_summary: Dict[str, Any] = {}

    def get_raw_metadata(self) -> Dict[str, Any]:
        """Compute structural metrics of the raw dataset."""
        null_count = int(self.df_raw.isna().sum().sum())
        duplicate_count = int(self.df_raw.duplicated().sum())
        invalid_ph_count = int(((self.df_raw["ph"] < 0) | (self.df_raw["ph"] > 14)).sum())
        negative_num_count = int(((self.df_raw[self.NUMERIC_FEATURES] < 0).sum().sum()))

        return {
            "total_rows": len(self.df_raw),
            "total_cols": len(self.df_raw.columns),
            "total_missing_cells": null_count,
            "duplicate_rows": duplicate_count,
            "invalid_ph_records": invalid_ph_count,
            "negative_value_records": negative_num_count,
            "dtypes": self.df_raw.dtypes.astype(str).to_dict(),
            "target_classes_count": int(self.df_raw[self.TARGET_COLUMN].nunique()) if self.TARGET_COLUMN in self.df_raw else 0
        }

    def get_missing_analysis(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Detailed missing value breakdown per feature."""
        target_df = self.df_raw if df is None else df
        missing_counts = target_df.isna().sum()
        missing_pct = (missing_counts / len(target_df) * 100).round(2)

        result = pd.DataFrame({
            "Feature Name": missing_counts.index,
            "Data Type": [str(target_df[c].dtype) for c in missing_counts.index],
            "Missing Count": missing_counts.values,
            "Missing Pct (%)": missing_pct.values
        })
        return result

    def get_duplicate_rows(self) -> pd.DataFrame:
        """Return subset of exact duplicate rows."""
        return self.df_raw[self.df_raw.duplicated(keep=False)].sort_values(by=list(self.df_raw.columns))

    def detect_outliers_iqr(self, df: pd.DataFrame = None, multiplier: float = 1.5) -> Dict[str, Any]:
        """Detect outliers using Interquartile Range (IQR) for numeric features."""
        target_df = self.df_raw if df is None else df
        outlier_summary = {}
        all_outlier_indices = set()

        for col in self.NUMERIC_FEATURES:
            if col in target_df.columns:
                series = target_df[col].dropna()
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (multiplier * iqr)
                upper_bound = q3 + (multiplier * iqr)

                outliers = target_df[(target_df[col] < lower_bound) | (target_df[col] > upper_bound)]
                outlier_indices = list(outliers.index)
                all_outlier_indices.update(outlier_indices)

                outlier_summary[col] = {
                    "q1": round(float(q1), 2),
                    "q3": round(float(q3), 2),
                    "iqr": round(float(iqr), 2),
                    "lower_bound": round(float(lower_bound), 2),
                    "upper_bound": round(float(upper_bound), 2),
                    "count": len(outliers),
                    "percentage": round(len(outliers) / len(target_df) * 100, 2)
                }

        return {
            "by_feature": outlier_summary,
            "total_outlier_rows": len(all_outlier_indices),
            "outlier_indices": list(all_outlier_indices)
        }

    def get_descriptive_stats(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Compute strict descriptive statistics for numerical columns."""
        target_df = self.df_raw if df is None else df
        num_df = target_df[self.NUMERIC_FEATURES]

        stats = num_df.describe().T.reset_index()
        stats.columns = ["Feature", "Count", "Mean", "Std Dev", "Min", "25%", "50% (Median)", "75%", "Max"]

        # Add Skewness and Kurtosis
        stats["Skewness"] = [round(float(num_df[c].skew()), 2) for c in stats["Feature"]]
        stats["Kurtosis"] = [round(float(num_df[c].kurtosis()), 2) for c in stats["Feature"]]

        for col in ["Mean", "Std Dev", "Min", "25%", "50% (Median)", "75%", "Max"]:
            stats[col] = stats[col].round(2)

        return stats

    def execute_pipeline(
        self,
        impute_strategy: str = "median",
        remove_duplicates: bool = True,
        handle_invalid: bool = True,
        remove_outliers: bool = False,
        iqr_multiplier: float = 1.5
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run complete data cleaning and transformation pipeline."""
        df = self.df_raw.copy()
        initial_rows = len(df)

        # 1. Duplicate Removal
        duplicates_removed = 0
        if remove_duplicates:
            dup_count = int(df.duplicated().sum())
            df = df.drop_duplicates()
            duplicates_removed = dup_count

        # 2. Invalid Value Handling
        invalid_removed = 0
        if handle_invalid:
            # Fix invalid pH (0 <= pH <= 14)
            invalid_mask = (df["ph"] < 0) | (df["ph"] > 14)
            invalid_removed = int(invalid_mask.sum())
            df.loc[invalid_mask, "ph"] = np.nan

        # 3. Missing Value Imputation
        missing_imputed = int(df.isna().sum().sum())
        for col in self.NUMERIC_FEATURES:
            if col in df.columns and df[col].isna().sum() > 0:
                if impute_strategy == "median":
                    fill_val = df[col].median()
                elif impute_strategy == "mean":
                    fill_val = df[col].mean()
                else:
                    fill_val = 0.0
                df[col] = df[col].fillna(fill_val)

        # 4. Outlier Handling (optional filtering)
        outliers_removed = 0
        if remove_outliers:
            outlier_info = self.detect_outliers_iqr(df, multiplier=iqr_multiplier)
            outlier_indices = outlier_info["outlier_indices"]
            outliers_removed = len(outlier_indices)
            df = df.drop(index=outlier_indices, errors="ignore")

        self.df_cleaned = df.reset_index(drop=True)

        summary = {
            "initial_rows": initial_rows,
            "final_rows": len(self.df_cleaned),
            "duplicates_removed": duplicates_removed,
            "invalid_ph_fixed": invalid_removed,
            "missing_cells_imputed": missing_imputed,
            "outliers_removed": outliers_removed,
            "features_processed": list(self.NUMERIC_FEATURES)
        }
        self.transformation_summary = summary
        return self.df_cleaned, summary

    def encode_target(self, df: pd.DataFrame = None) -> Tuple[pd.DataFrame, Dict[int, str]]:
        """Encode categorical target column using LabelEncoder."""
        target_df = self.df_cleaned if df is None else df.copy()
        if self.TARGET_COLUMN in target_df.columns:
            encoded_labels = self.label_encoder.fit_transform(target_df[self.TARGET_COLUMN])
            target_df[f"{self.TARGET_COLUMN}_encoded"] = encoded_labels
            mapping = dict(enumerate(self.label_encoder.classes_))
            return target_df, mapping
        return target_df, {}

    def scale_features(self, df: pd.DataFrame, method: str = "StandardScaler") -> Tuple[pd.DataFrame, Any]:
        """Scale numerical features using StandardScaler or MinMaxScaler."""
        scaled_df = df.copy()
        if method == "StandardScaler":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()

        scaled_array = self.scaler.fit_transform(scaled_df[self.NUMERIC_FEATURES])
        scaled_num_df = pd.DataFrame(scaled_array, columns=[f"{c}_scaled" for c in self.NUMERIC_FEATURES], index=scaled_df.index)

        return pd.concat([scaled_df, scaled_num_df], axis=1), self.scaler

    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Prepare train/test arrays."""
        X = df[self.NUMERIC_FEATURES]
        y = df[self.TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        return {
            "X_train_shape": X_train.shape,
            "X_test_shape": X_test.shape,
            "y_train_shape": y_train.shape,
            "y_test_shape": y_test.shape,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }
