"""
ARFF File Generator for WEKA Data Mining Integration.
Converts Pandas DataFrames into WEKA Attribute-Relation File Format (.arff).
"""

import pandas as pd
import os

def export_to_arff(df: pd.DataFrame, output_path: str, relation_name: str = "crop_recommendation") -> str:
    """
    Export Pandas DataFrame to WEKA ARFF format.
    
    Attributes:
    N, P, K, temperature, humidity, ph, rainfall (numeric)
    label (nominal class)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    numeric_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    target_col = "label"

    lines = []
    lines.append(f"@relation {relation_name}\n")

    # Write numeric attributes
    for col in numeric_cols:
        lines.append(f"@attribute {col} numeric")

    # Write class attribute (nominal)
    if target_col in df.columns:
        classes = sorted(df[target_col].dropna().astype(str).unique())
        class_str = "{" + ", ".join(classes) + "}"
        lines.append(f"@attribute {target_col} {class_str}")

    lines.append("\n@data")

    # Write data rows
    cols_to_write = [c for c in numeric_cols if c in df.columns]
    if target_col in df.columns:
        cols_to_write.append(target_col)

    for _, row in df[cols_to_write].iterrows():
        row_vals = []
        for col in cols_to_write:
            val = row[col]
            if pd.isna(val):
                row_vals.append("?")
            elif col == target_col:
                row_vals.append(str(val))
            else:
                row_vals.append(str(round(float(val), 2)))
        lines.append(",".join(row_vals))

    arff_content = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(arff_content)

    return output_path
