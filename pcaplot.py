import glob
import pandas as pd
import numpy as np

# Use non-GUI backend for matplotlib to prevent freezing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def plot_pca_knn(csv_folder=".", k=5, max_points=10000):
    # 1) Load & concatenate only matching CSVs
    csv_files = glob.glob(f"{csv_folder}/wifisignal_data_*.csv")
    if not csv_files:
        raise FileNotFoundError(f"No matching CSV files found in {csv_folder} with prefix 'wifisignal_data_'")

    print(f"Loading {len(csv_files)} CSV files...")

    df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

    # 2) Extract features & labels
    pkt_cols = [c for c in df.columns if c.startswith("pkt")]
    if not pkt_cols:
        raise ValueError("No feature columns found starting with 'pkt'")

    if 'label' not in df.columns:
        raise ValueError("Missing 'label' column in dataset")

    X = df[pkt_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    labels = df['label'].astype(str).values

    print(f"Data shape: {X.shape}, Number of labels: {len(np.unique(labels))}")

    # 3) PCA → 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    print("PCA explained variance ratios:", pca.explained_variance_ratio_)

    # 4) Downsample (optional but necessary for large data)
    if X_pca.shape[0] > max_points:
        print(f"Downsampling from {X_pca.shape[0]} to {max_points} points for plotting...")
        np.random.seed(42)
        indices = np.random.choice(X_pca.shape[0], max_points, replace=False)
        X_pca = X_pca[indices]
        labels = labels[indices]

    # 5) Build kNN graph in PCA space
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_pca)
    dists, idx = nbrs.kneighbors(X_pca)

    edges = set()
    N = X_pca.shape[0]
    for i in range(N):
        for j in idx[i, 1:]:  # skip self
            edge = (i, j) if i < j else (j, i)
            edges.add(edge)

    # 6) Plot and save
    fig, ax = plt.subplots(figsize=(12, 9))

    # 6a) Draw kNN edges
    for i, j in edges:
        xi, yi = X_pca[i]
        xj, yj = X_pca[j]
        ax.plot([xi, xj], [yi, yj], color="gray", alpha=0.3, linewidth=0.5)

    # 6b) Draw points by label
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))
    for idx_lbl, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=25,
            color=cmap(idx_lbl),
            label=lbl,
            edgecolor="none",
            alpha=0.8
        )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(f"PCA + kNN Graph (k={k})")
    ax.legend(title="Activity", loc='best')
    plt.tight_layout()

    output_path = "pca_knn_plot.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ Plot saved as '{output_path}'")


if __name__ == "__main__":
    # requires: pip install pandas scikit-learn matplotlib
    plot_pca_knn(csv_folder="./data", k=3)

# Downsize sampling to 10000 points, so will not get stuck at PCA graph generation.