import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
import os


DATA_PATH = 'data/train.csv'
OUTPUT_DIR = 'results'
MAX_CLUSTERS = 20
RANDOM_STATE = 42
SAMPLE_SIZE = 10000
PCA_COMPONENTS = 50  


def analyze_clusters():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    v_cols = [c for c in df.columns if c.startswith('V')]
    if not v_cols:
        print("No embedding columns (V...) found!")
        return
    print(f"Found {len(v_cols)} embedding columns.")
    X = df[v_cols].fillna(0).values
    print("Scaling data...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Reducing dimensions with PCA (n={PCA_COMPONENTS})...")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.2%}")
    inertias = []
    silhouettes = []
    k_range = range(2, MAX_CLUSTERS + 1)
    print(f"Starting analysis for k=2 to {MAX_CLUSTERS} on PCA data...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(X_pca)
        inertias.append(kmeans.inertia_)
        if len(X_pca) > SAMPLE_SIZE:
            indices = np.random.choice(len(X_pca), SAMPLE_SIZE, replace=False)
            X_sample = X_pca[indices]
            labels_sample = kmeans.predict(X_sample)
        else:
            X_sample = X_pca
            labels_sample = kmeans.labels_
        score = silhouette_score(X_sample, labels_sample)
        silhouettes.append(score)
        print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={score:.4f}")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:red'
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia (Elbow)', color=color)
    ax1.plot(k_range, inertias, marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title('Elbow Method & Silhouette Score Analysis')
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Silhouette Score', color=color)  
    ax2.plot(k_range, silhouettes, marker='x', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  
    plt.grid(True)
    output_file = os.path.join(OUTPUT_DIR, 'cluster_analysis.png')
    plt.savefig(output_file)
    print(f"\nAnalysis complete. Plot saved to {output_file}")
    print("Check the plot to find the 'elbow' or the peak silhouette score.")


if __name__ == "__main__":
    analyze_clusters()
