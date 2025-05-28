import glob
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.neighbors   import NearestNeighbors

import matplotlib.pyplot as plt

def plot_pca_knn(csv_folder="./data", k=5):
    # 1) Load & concatenate all CSVs
    csv_files = glob.glob(f"{csv_folder}/*.csv")
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_folder}")
    df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

    # 2) Extract features & labels
    pkt_cols = [c for c in df.columns if c.startswith("pkt")]
    X = df[pkt_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values  # shape (N,60)
    labels = df["label"].astype(str).values                               # shape (N,)

    # 3) PCA → 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)  # shape (N,2)

    # 4) Build kNN graph in PCA space
    nbrs    = NearestNeighbors(n_neighbors=k+1).fit(X_pca)
    dists, idx = nbrs.kneighbors(X_pca)
    # idx[:,0] is self → skip it; build undirected edge set
    edges = set()
    N = X_pca.shape[0]
    for i in range(N):
        for j in idx[i, 1:]:
            edge = (i, j) if i < j else (j, i)
            edges.add(edge)

    # 5) Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # 5a) Draw edges first
    for i, j in edges:
        xi, yi = X_pca[i]
        xj, yj = X_pca[j]
        ax.plot([xi, xj], [yi, yj],
                color="gray", alpha=0.3, linewidth=0.5)

    # 5b) Scatter packets colored by label
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
    ax.legend(title="Activity")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # requires: pip install pandas scikit-learn matplotlib
    plot_pca_knn(csv_folder="./data", k=3)
