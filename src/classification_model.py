"""
classification_model.py

Purpose:
Train and evaluate a supervised classification model
to predict income level (>50K or <=50K).

Outputs saved in:
../outputs/
"""

import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from data_preprocessing import load_dataset, preprocess_data


def run_classification():

    print("Step 1: Loading dataset...")

    # Load data
    df = load_dataset(
        "../data/census-bureau.data",
        "../data/census-bureau.columns"
    )

    print("Step 2: Preprocessing data...")

    # Prepare data
    X, y, weights = preprocess_data(df, classification=True)

    print("Step 3: Splitting dataset...")

    # Split data (80/20 split)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Step 4: Training Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train, sample_weight=w_train)

    print("Step 5: Predicting...")

    y_pred = model.predict(X_test)

    print("Step 6: Evaluating model...")

    accuracy = accuracy_score(y_test, y_pred)

    # Create outputs folder if not exists
    os.makedirs("../outputs", exist_ok=True)

    # Save classification results
    with open("../outputs/classification_results.txt", "w") as f:
        f.write("Classification Model Results\n")
        f.write("-----------------------------------\n")
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write(classification_report(y_test, y_pred))

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.savefig("../outputs/confusion_matrix.png")
    plt.close()

    # Save feature importance
    importances = model.feature_importances_
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(importances)), importances)
    plt.title("Feature Importance")
    plt.savefig("../outputs/feature_importance.png")
    plt.close()

    print("Classification completed successfully.")
    print("Results saved in outputs/ folder.")


if __name__ == "__main__":
    run_classification()
