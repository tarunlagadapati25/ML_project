"""
data_preprocessing.py

This module:
1. Loads the dataset
2. Assigns column names
3. Encodes categorical variables
4. Scales numerical features
5. Separates target variable (for classification)
6. Extracts weight column

Used by both classification and segmentation scripts.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_dataset(data_path, columns_path):
    """
    Load dataset and apply column names.
    """

    # Read column names
    with open(columns_path, "r") as file:
        columns = [line.strip() for line in file.readlines()]

    # Load dataset (no header in original file)
    df = pd.read_csv(data_path, header=None)

    # Assign column names
    df.columns = columns

    return df


def preprocess_data(df, classification=True):
    """
    Preprocess data for ML models.

    classification=True  → includes income label
    classification=False → for clustering only
    """

    df = df.copy()

    # Extract weight column (important for weighted training)
    weights = df["weight"]

    # Convert income into binary label if classification
    y = None
    if classification:
        y = df["income"].apply(
            lambda x: 1 if ">50K" in str(x) else 0
        )

    # Drop columns not needed as features
    drop_cols = ["weight"]
    if classification:
        drop_cols.append("income")

    X = df.drop(columns=drop_cols)

    # Encode categorical features
    categorical_cols = X.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, weights
