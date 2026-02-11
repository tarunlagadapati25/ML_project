"""
segmentation_model.py

Purpose:
Create customer segments using KMeans clustering.

Outputs saved in:
../outputs/
"""

import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from data_preprocessing import load_dataset, preprocess_data


def run_segmentation():

    print("Step 1: Loading dataset...")

    df = load_dataset(
        "../data/census-bureau.data",
        "../data/census-bureau.columns"
    )

    print("Step 2: Preprocessing data for clustering...")

    # Prepare data without income label
    X, _, _ = preprocess_data(df, classification=False)

    print("Step 3: Running KMeans clustering...")

    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X)

    df["Cluster"] = clusters

    # Create outputs folder
    os.makedirs("../outputs", exist_ok=True)

    # Save cluster distribution
    with open("../outputs/cluster_distribution.txt", "w") as f:
        f.write("Cluster Distribution\n")
        f.write("-----------------------------------\n")
        f.write(df["Cluster"].value_counts().to_string())

    print("Step 4: Creating PCA visualization...")

    # Reduce dimensions for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
    plt.title("Customer Segmentation")
    plt.savefig("../outputs/cluster_visualization.png")
    plt.close()

    print("Segmentation completed successfully.")
    print("Results saved in outputs/ folder.")


if __name__ == "__main__":
    run_segmentation()
