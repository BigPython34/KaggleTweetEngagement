import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os


ARTIFACTS_DIR = 'results/artifacts'


def check_pca_variance():
    pca_path = os.path.join(ARTIFACTS_DIR, 'pca_v.pkl')
    if not os.path.exists(pca_path):
        print("PCA model not found.")
        return
    pca = joblib.load(pca_path)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"Total Variance Explained by {pca.n_components_} components: {explained_variance:.4f}")
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.savefig('results/pca_variance.png')
    print("Saved variance plot to results/pca_variance.png")


if __name__ == "__main__":
    check_pca_variance()
