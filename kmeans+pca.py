"""
K-Means Clustering - A simple tutorial for students
===================================================
K-Means is an *unsupervised* method: we have no labels. The goal is to group
similar points into K clusters. The algorithm alternates between:
  1. Assign each point to the nearest cluster center (centroid).
  2. Update each centroid to be the mean of the points assigned to it.

Here we use data with *several features* (Iris: 4D). We cannot "see" the
centroids in 4D, so we run K-Means in full dimension and use PCA only to
*visualize* the result in 2D. This is more realistic: clustering happens in
the real feature space, not in a toy 2D space.

Go to https://scikit-learn.org/stable/modules/clustering.html#k-means for more.
"""

import numpy as np  # numerical arrays and math
from sklearn import datasets  # built-in datasets (e.g. Iris)
from sklearn.cluster import KMeans  # K-Means clustering
from sklearn.preprocessing import StandardScaler  # scale features before clustering
from sklearn.decomposition import PCA  # project to 2D for visualization only
import os

print("K-Means tutorial: cluster data in 4D, visualize in 2D via PCA.\n")

# ---------------------------------------------------------------------------
# Step 1 (Data Loading): Load data with more than 2 features
# ---------------------------------------------------------------------------
# Iris: 150 samples, 4 features (sepal length/width, petal length/width).
# We don't use true labels for clustering, but we keep them to visualize and compare.
iris = datasets.load_iris()
X = iris.data  # shape (150, 4)
y_true = iris.target  # true species (0=Setosa, 1=Versicolor, 2=Virginica) — for visualization only
target_names = iris.target_names

print(f"Dataset shape: {X.shape} (samples, features)")
print("Features: sepal length, sepal width, petal length, petal width")
print("We cannot plot 4D directly; K-Means will work in 4D.\n")

# ---------------------------------------------------------------------------
# Step 2 (Preprocessing): Standardize features
# ---------------------------------------------------------------------------
# K-Means uses distances. If one feature has much larger scale (e.g. petal length
# in cm vs sepal width in cm), it would dominate. Scaling puts all features on
# comparable footing.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------------------
# Step 3 (Model): Fit K-Means in the full 4D space
# ---------------------------------------------------------------------------
# n_clusters=3: Iris has 3 species; in practice you might try several K (e.g. elbow).
# n_init=10: run 10 times with different random starts; keep the run with lowest inertia.
K = 3
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans.fit(X_scaled)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_  # shape (3, 4) — we cannot plot 4D directly!

print(f"Number of clusters (K): {K}")
print(f"Points per cluster: {np.bincount(labels)}")
print("Centroids are in 4D (one vector per cluster); we'll project them to 2D for plotting.\n")
print(f"Inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")

# ---------------------------------------------------------------------------
# Step 4: Visualize in 2D by projecting with PCA (for illustration only)
# ---------------------------------------------------------------------------
# Clustering was done in 4D. To *visualize* we project data and centroids
# onto the first 2 principal components. So the 2D plot is a view of the
# 4D result, not the space where K-Means actually ran.
try:
    import matplotlib.pyplot as plt

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)  # project data to 2D
    centroids_2d = pca.transform(centroids)  # project centroids to same 2D space

    IMAGE_DIR = "images"
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # --- Figure 1: Raw data in 2D (PCA projection — no cluster colors) ---
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c="gray", alpha=0.6, s=50, edgecolors="k")
    plt.xlabel("PC1 (projected)")
    plt.ylabel("PC2 (projected)")
    plt.title("Raw data in 2D (PCA) — K-Means was run in 4D")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "kmeans_data_raw.png"), dpi=120)
    plt.close()

    # --- Figure 2: True classes (ground truth) — for comparison with K-Means ---
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap="viridis", alpha=0.7, s=50, edgecolors="k")
    plt.xlabel("PC1 (projected)")
    plt.ylabel("PC2 (projected)")
    plt.title("True classes (Iris species) — ground truth")
    plt.legend(scatter.legend_elements()[0], list(target_names), loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "kmeans_true_classes.png"), dpi=120)
    plt.close()

    # --- Figure 3: Clustering result in 2D — points colored by cluster, centroids in 2D ---
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="viridis", alpha=0.7, s=50, edgecolors="k")
    plt.scatter(
        centroids_2d[:, 0], centroids_2d[:, 1],
        c="red", marker="X", s=200, edgecolors="black", linewidths=2,
        label="Centroids (projected to 2D)"
    )
    plt.xlabel("PC1 (projected)")
    plt.ylabel("PC2 (projected)")
    plt.title(f"K-Means result (K={K}) — clusters found in 4D, shown in 2D")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "kmeans_clusters.png"), dpi=120)
    plt.close()

    print(f"\nImages saved in '{IMAGE_DIR}/': kmeans_data_raw.png, kmeans_true_classes.png, kmeans_clusters.png")
except ImportError:
    print("\n(Install matplotlib to plot the results.)")
