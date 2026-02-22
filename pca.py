"""
PCA for Visualizing Dimension Reduction
=========================================================================
PCA (Principal Component Analysis) finds directions in the data that capture
the most variance. By projecting data onto the top 2 components, we can
visualize high-dimensional data in 2D while keeping as much structure as possible.

Think of it as: "Which 2 directions in the data are most informative?"
We use the Wine dataset (13 chemical features) and reduce it to 2 dimensions for plotting.
This is a more realistic setting: many features → 2D summary.

Go to https://scikit-learn.org/stable/modules/decomposition.html#pca for more.
"""

import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("PCA tutorial: reduce 13D Wine data to 2D for visualization.\n")

# ---------------------------------------------------------------------------
# Step 1 (Data): Load a dataset with more than 2 features
# ---------------------------------------------------------------------------
# Wine: 178 samples, 13 features (alcohol, malic acid, ash, alkalinity, Mg, phenols, etc.), 3 cultivars
wine = datasets.load_wine()
X, y = wine.data, wine.target
target_names = wine.target_names
feature_names = wine.feature_names

print(f"Dataset shape: {X.shape} (samples, features)")
print(f"Features: {', '.join(feature_names)}")
print(f"Classes: {list(target_names)}\n")

# ---------------------------------------------------------------------------
# Step 2 (Preprocessing): Standardize the features
# ---------------------------------------------------------------------------
# PCA is sensitive to scale. Features with larger variance would dominate.
# StandardScaler: subtract mean, divide by std → each feature has mean=0, std=1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data standardized (zero mean, unit variance per feature).")

# ---------------------------------------------------------------------------
# Step 3 (PCA): Fit PCA and reduce to 2 dimensions
# ---------------------------------------------------------------------------
# n_components=2: we keep only the 2 main directions (principal components).
# These are the directions of MAXIMUM VARIANCE in the data.
n_components = 2
pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA fitted. Reduced from {X_scaled.shape[1]} features to {n_components} dimensions.")
print(f"New shape: {X_pca.shape}")

# ---------------------------------------------------------------------------
# Step 4: How much information did we keep?
# ---------------------------------------------------------------------------
# explained_variance_ratio_ tells us the fraction of total variance in each component.
# Sum of these = total variance retained by our 2D projection.
explained = pca.explained_variance_ratio_
total_explained = explained.sum()
print(f"\nExplained variance per component: {explained}")
print(f"Total variance retained in 2D: {total_explained:.1%}")
print(f"(So we keep more than 95% of the structure while dropping {(X_scaled.shape[1] - n_components)} dimensions.)\n")

# ---------------------------------------------------------------------------
# Step 5: Visualize — original first 2 features vs PCA projection
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    import os

    IMAGE_DIR = "images"
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # Colors for the 3 wine cultivars
    colors = ["#e74c3c", "#3498db", "#2ecc71"]  # red, blue, green
    target_ids = [0, 1, 2]

    # --- Figure 1: Original data (only first 2 features) ---
    # This is one possible 2D view; we're ignoring the other 11 features.
    plt.figure(figsize=(8, 6))
    for id in target_ids:
        mask = y == id
        plt.scatter(
            X[mask, 0], X[mask, 1],
            c=colors[id], label=target_names[id], edgecolors="k", s=50
        )
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title("Original data: first 2 features only (2 of 13 dimensions)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "pca_original_2d.png"), dpi=120)
    plt.close()

    # --- Figure 2: PCA projection (best 2D summary of all 13 features) ---
    # PCA uses all 13 features to find the 2 directions that preserve the most variance.
    plt.figure(figsize=(8, 6))
    for id in target_ids:
        mask = y == id
        plt.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=colors[id], label=target_names[id], edgecolors="k", s=50
        )
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")
    plt.title(f"PCA projection (2D from 13D) — variance retained: {total_explained:.1%}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "pca_projection_2d.png"), dpi=120)
    plt.close()

    # --- Figure 3: Scree plot — how much variance each component captures ---
    # Useful to decide how many components to keep (e.g. for 95% variance).
    pca_full = PCA().fit(X_scaled)
    n_features = X_scaled.shape[1]
    plt.figure(figsize=(8, 4))
    plt.bar(
        range(1, len(pca_full.explained_variance_ratio_) + 1),
        pca_full.explained_variance_ratio_,
        color="steelblue",
        edgecolor="black",
    )
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.title(f"Variance explained by each component (all {n_features} dimensions)")
    plt.xticks(range(1, n_features + 1))
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "pca_scree.png"), dpi=120)
    plt.close()

    print(f"Images saved in '{IMAGE_DIR}/': pca_original_2d.png, pca_projection_2d.png, pca_scree.png")
except ImportError:
    print("(Install matplotlib to generate the plots.)")
