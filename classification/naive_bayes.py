"""
Gaussian Naive Bayes Classifier Engine for AgriSense (Phase 5).
Trains GaussianNB on numerical soil and microclimate features and provides
comparative benchmarks against WEKA J48 and real-time single-sample crop prediction.
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, Any, Tuple

class AgriculturalNaiveBayesClassifier:
    """Gaussian Naive Bayes classifier wrapper."""

    FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    TARGET_COL = "label"

    @classmethod
    def train_and_evaluate(
        cls,
        df: pd.DataFrame,
        test_size: float = 0.20,
        var_smoothing: float = 1e-9,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Train GaussianNB classifier and compute evaluation metrics."""
        clean_df = df.dropna(subset=cls.FEATURE_COLS + [cls.TARGET_COL]).copy()

        X = clean_df[cls.FEATURE_COLS]
        y = clean_df[cls.TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        model = GaussianNB(var_smoothing=var_smoothing)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")

        classes = sorted(y_test.unique().tolist())
        cm = confusion_matrix(y_test, y_pred, labels=classes)

        # Per class stats
        p_c, r_c, f1_c, sup_c = precision_recall_fscore_support(y_test, y_pred, labels=classes)
        class_df = pd.DataFrame({
            "Class": classes,
            "Precision": p_c.round(3),
            "Recall": r_c.round(3),
            "F1-Score": f1_c.round(3),
            "Test Samples": sup_c
        })

        return {
            "model": model,
            "accuracy": round(acc * 100, 2),
            "precision": round(float(p), 3),
            "recall": round(float(r), 3),
            "f1": round(float(f1), 3),
            "correct_count": int((y_test == y_pred).sum()),
            "incorrect_count": int((y_test != y_pred).sum()),
            "total_test_instances": len(y_test),
            "confusion_matrix": cm,
            "classes": classes,
            "class_metrics": class_df,
            "feature_names": cls.FEATURE_COLS
        }

    @classmethod
    def predict_single_sample(
        cls,
        model: GaussianNB,
        input_data: Dict[str, float]
    ) -> Dict[str, Any]:
        """Predict crop label for a single input record."""
        sample_df = pd.DataFrame([[input_data[c] for c in cls.FEATURE_COLS]], columns=cls.FEATURE_COLS)
        pred_label = model.predict(sample_df)[0]
        probs = model.predict_proba(sample_df)[0]

        prob_df = pd.DataFrame({
            "Crop": model.classes_,
            "Probability": (probs * 100).round(2)
        }).sort_values(by="Probability", ascending=False).reset_index(drop=True)

        return {
            "predicted_crop": pred_label,
            "confidence_pct": round(prob_df.iloc[0]["Probability"], 2),
            "probabilities_table": prob_df
        }
