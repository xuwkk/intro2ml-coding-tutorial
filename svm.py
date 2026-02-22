"""
SVM Binary Classification
==========================================================
We use a dataset that is NOT linearly separable (two interlocking "moons").
A linear SVM cannot separate them well; a kernel (e.g. RBF) maps the data
to a space where a linear boundary works.

Go to https://scikit-learn.org/stable/modules/svm.html# for more details.
"""

import numpy as np  # numerical arrays and math
from sklearn import datasets  # built-in datasets (e.g. make_moons)
from sklearn.model_selection import train_test_split  # split data into train/test
from sklearn.svm import SVC  # Support Vector Classifier (SVM for classification)
from sklearn.metrics import accuracy_score, confusion_matrix  # evaluate predictions

print("This is a toy example for binary classification using the `make_moons` dataset.\n")

# ---------------------------------------------------------------------------
# Step 1 (Data Loading): Load a non-linearly separable dataset
# ---------------------------------------------------------------------------
# make_moons: two crescent shapes that overlap — no single line can separate them.
# noise=0.15 adds some randomness so the problem is not trivial.
X, y = datasets.make_moons(n_samples=200, noise=0.15, random_state=42)

print(f"Dataset shape: {X.shape}")
print(f"Classes: {np.unique(y)} (two interlocking moons)\n")

# ---------------------------------------------------------------------------
# Step 2 (Data Preprocessing): Split into train and test sets
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print(f"Train samples: {len(y_train)} | Test samples: {len(y_test)}")

# ---------------------------------------------------------------------------
# Step 3 (Model Training): Linear SVM vs Kernel SVM (RBF)
# ---------------------------------------------------------------------------
# Linear SVM: decision boundary is a straight line. Struggles with moons!
clf_linear = SVC(kernel="linear", C=1.0)
clf_linear.fit(X_train, y_train)

# RBF (Radial Basis Function) kernel: implicitly maps data to a higher-dimensional
# space where a linear boundary can separate the classes. Much better for moons.
clf_rbf = SVC(kernel="rbf", C=1.0, gamma="scale")
clf_rbf.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# Step 4 (Model Evaluation): Compare both models
# ---------------------------------------------------------------------------
for name, clf in [("Linear SVM", clf_linear), ("RBF kernel SVM", clf_rbf)]:
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} — Test accuracy: {acc:.2%}")
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------------------------
# Step 5: Plot dataset and decision boundaries (linear vs RBF)
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    import os
    
    # Directory where all images will be saved
    IMAGE_DIR = "images"
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # Grid for decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # --- Figure 1: Dataset only (not linearly separable) ---
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdYlBu", edgecolors="k", s=50)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Dataset: two moons (not linearly separable)")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "dataset_moons.png"), dpi=120)
    plt.close()

    # --- Figure 2: Linear SVM (poor fit) ---
    # Predict on a fine grid: stack xx,yy into (N,2) points, predict, then reshape to grid for contourf
    Z_linear = clf_linear.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z_linear, alpha=0.3, cmap="RdYlBu")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdYlBu", edgecolors="k", s=50)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Linear SVM — cannot separate the moons well")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "svm_linear_boundary.png"), dpi=120)
    plt.close()

    # --- Figure 3: RBF kernel SVM (good fit) ---
    Z_rbf = clf_rbf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z_rbf, alpha=0.3, cmap="RdYlBu")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdYlBu", edgecolors="k", s=50)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("RBF kernel SVM — non-linear boundary separates the moons")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "svm_rbf_boundary.png"), dpi=120)
    plt.close()

    print(f"\nImages saved in '{IMAGE_DIR}/': dataset_moons.png, svm_linear_boundary.png, svm_rbf_boundary.png")
except ImportError:
    print("\n(Install matplotlib to plot the decision boundary.)")
