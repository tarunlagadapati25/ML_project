import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(df, classification=True):
    """
    FINAL WORKING VERSION
    Compatible with:
    - Python 3.12
    - Latest pandas
    - No deprecated parameters
    """

    # Make a copy
    df = df.copy()

    # Strip spaces safely
    df = df.apply(lambda col: col.astype(str).str.strip())

    # Replace '?' with NaN
    df.replace("?", np.nan, inplace=True)

    # ---- SAFE NUMERIC CONVERSION ----
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            # If conversion fails, leave as categorical
            pass

    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    # Fill numeric missing values
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Identify label and weight columns
    label_column = df.columns[-1]       # last column
    weight_column = df.columns[-2]      # second last column

    weights = df[weight_column]

    y = None
    if classification:
        y = df[label_column].apply(
            lambda x: 1 if ">50K" in str(x) else 0
        )

    # Drop label and weight from features
    drop_cols = [weight_column]
    if classification:
        drop_cols.append(label_column)

    X = df.drop(columns=drop_cols)

    # Encode categorical columns
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, weights
