import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

from data_preprocessing import preprocess_data


def load_dataset():
    print("Loading dataset...")

    # Load column names
    with open("../data/census-bureau.columns", "r") as f:
        column_names = [line.strip() for line in f.readlines()]

    # Load dataset
    df = pd.read_csv(
        "../data/census-bureau.data",
        header=None,
        names=column_names
    )

    return df


def run_classification():

    df = load_dataset()

    print("Preprocessing data...")
    X, y, weights = preprocess_data(df, classification=True)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Predicting...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

    # Create outputs folder
    os.makedirs("../outputs", exist_ok=True)

    # Save results
    with open("../outputs/classification_results.txt", "w") as f:
        f.write("Classification Results\n")
        f.write("-----------------------\n")
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

    # Get feature importances
    importances = model.feature_importances_

    

    # Get feature names (after preprocessing)
    feature_names = df.columns[:-2]  # drop weight + label

    # Create dataframe for sorting
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    # Sort by importance
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Take top 15 features for visualization
    top_features = importance_df.head(15)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(top_features["Feature"], top_features["Importance"])
    plt.xlabel("Importance Score")
    plt.title("Top 15 Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plt.savefig("../outputs/feature_importance.png")
    plt.close()

    print("Classification completed. Check outputs folder.")


if __name__ == "__main__":
    run_classification()
