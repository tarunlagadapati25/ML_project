import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from data_preprocessing import preprocess_data


def load_dataset():
    with open("../data/census-bureau.columns", "r") as f:
        column_names = [line.strip() for line in f.readlines()]

    df = pd.read_csv(
        "../data/census-bureau.data",
        header=None,
        names=column_names
    )

    return df


def run_segmentation():

    print("Loading dataset for segmentation...")
    df = load_dataset()

    print("Preprocessing data...")
    X, _, _ = preprocess_data(df, classification=False)

    print("Running KMeans clustering...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    df["Cluster"] = clusters

    # Create outputs folder
    os.makedirs("../outputs", exist_ok=True)

    # Save cluster distribution
    with open("../outputs/cluster_distribution.txt", "w") as f:
        f.write("Cluster Distribution\n")
        f.write(df["Cluster"].value_counts().to_string())

    # PCA visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
    plt.title("Customer Segmentation")
    plt.savefig("../outputs/cluster_visualization.png")
    plt.close()

    print("Segmentation completed. Check outputs folder.")


if __name__ == "__main__":
    run_segmentation()
