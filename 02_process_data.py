import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
import joblib
import os

# Configuration
DATA_DIR = 'data'
PROCESSED_DIR = 'data_processed'
ARTIFACTS_DIR = 'results/artifacts'

TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
AUTHOR_DATA_PATH = os.path.join(DATA_DIR, 'authorData.csv')

RANDOM_STATE = 42

# Hyperparameters determined by analysis
PCA_COMPONENTS = 150  # Increased to 150 (approx 98% variance)
CLUSTER_MACRO_K = 3
CLUSTER_MICRO_K = 10

def process_domain(df, top_n=20):
    """Keeps top N domains, maps others to 'other'."""
    if "shared_url_domain" not in df.columns:
        return df
        
    # Normalize
    df["shared_url_domain"] = df["shared_url_domain"].fillna("missing").astype(str).str.lower()
    
    # Calculate counts on the full dataset (if possible) or just return cleaned
    # We will handle the "Top N" logic inside process_data using train+test
    return df

def create_time_features(df):
    """Extracts temporal features from timestamp."""
    df = df.copy()
    if "timestamp" in df.columns:
        # Convert ms to datetime
        ts = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        
        df["_hour"] = ts.dt.hour
        df["_dow"] = ts.dt.dayofweek
        df["_month"] = ts.dt.month
        df["_is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
        
        # Cyclical features for hour
        df["_hour_sin"] = np.sin(2 * np.pi * df["_hour"] / 24)
        df["_hour_cos"] = np.cos(2 * np.pi * df["_hour"] / 24)
        
        # Drop original timestamp
        df = df.drop(columns=["timestamp"])
    return df

def add_author_features(train_df, test_df):
    """
    Enriches data with author statistics from authorData.csv.
    """
    print("   Loading authorData.csv for feature engineering...")
    if not os.path.exists(AUTHOR_DATA_PATH):
        print("   Warning: authorData.csv not found. Skipping author features.")
        return train_df, test_df

    # Read only necessary columns to save memory
    cols_to_load = ["author", "engagement", "contains_video", "is_reply", "is_retweet", "contains_image"]
    try:
        author_df = pd.read_csv(AUTHOR_DATA_PATH, usecols=cols_to_load)
    except ValueError:
        # Fallback if columns are different
        author_df = pd.read_csv(AUTHOR_DATA_PATH)
        
    # Ensure booleans
    for c in ["contains_video", "is_reply", "is_retweet", "contains_image"]:
        if c in author_df.columns:
            author_df[c] = author_df[c].astype(bool).astype(int)

    # Calculate stats
    # 1. Global Stats
    author_stats = author_df.groupby("author").agg({
        "engagement": ["mean", "max", "std", "count"],
        "is_reply": "mean",      # Ratio of replies
        "is_retweet": "mean",    # Ratio of retweets
        "contains_image": "mean" # Ratio of images
    }).reset_index()
    
    # Flatten MultiIndex
    author_stats.columns = [
        "author", 
        "author_mean_eng", "author_max_eng", "author_std_eng", "author_count",
        "author_reply_ratio", "author_retweet_ratio", "author_image_ratio"
    ]

    # 2. Video Specific Stats (Do they perform better on video?)
    if "contains_video" in author_df.columns:
        video_stats = author_df[author_df["contains_video"] == 1].groupby("author")["engagement"].mean().reset_index()
        video_stats.columns = ["author", "author_video_mean_eng"]
        author_stats = author_stats.merge(video_stats, on="author", how="left")
    
    # Log transform the engagement stats to reduce skew
    for col in ["author_mean_eng", "author_max_eng", "author_std_eng", "author_video_mean_eng"]:
        if col in author_stats.columns:
            author_stats[col] = np.log1p(author_stats[col])

    print(f"   Computed rich stats for {len(author_stats)} authors.")

    # Merge with train and test
    # We use 'left' join to keep all rows from train/test
    train_merged = train_df.merge(author_stats, on="author", how="left")
    test_merged = test_df.merge(author_stats, on="author", how="left")

    # Fill missing values
    # For mean/max, fill with global average
    for col in ["author_mean_eng", "author_max_eng"]:
        global_val = author_stats[col].mean()
        train_merged[col] = train_merged[col].fillna(global_val)
        test_merged[col] = test_merged[col].fillna(global_val)
    
    # For ratios, fill with global average
    for col in ["author_reply_ratio", "author_retweet_ratio", "author_image_ratio"]:
        global_val = author_stats[col].mean()
        train_merged[col] = train_merged[col].fillna(global_val)
        test_merged[col] = test_merged[col].fillna(global_val)
    
    # For std, fill with 0 (assume consistent if unknown)
    train_merged["author_std_eng"] = train_merged["author_std_eng"].fillna(0)
    test_merged["author_std_eng"] = test_merged["author_std_eng"].fillna(0)
    
    # For video mean, fill with the author's general mean (if they never posted video, assume average)
    # If author is new, it will already be filled with global mean above
    if "author_video_mean_eng" in train_merged.columns:
        train_merged["author_video_mean_eng"] = train_merged["author_video_mean_eng"].fillna(train_merged["author_mean_eng"])
        test_merged["author_video_mean_eng"] = test_merged["author_video_mean_eng"].fillna(test_merged["author_mean_eng"])

    train_merged["author_count"] = train_merged["author_count"].fillna(0)
    
    # --- Interaction Features ---
    # Does this author perform better on weekends?
    if "_is_weekend" in train_merged.columns:
        train_merged["author_x_weekend"] = train_merged["author_mean_eng"] * train_merged["_is_weekend"]
        test_merged["author_x_weekend"] = test_merged["author_mean_eng"] * test_merged["_is_weekend"]
        
    if "contains_video" in train_merged.columns:
        train_merged["author_x_video"] = train_merged["author_mean_eng"] * train_merged["contains_video"]
        test_merged["author_x_video"] = test_merged["author_mean_eng"] * test_merged["contains_video"]

    # NEW: Interactions with feature1 (Score?) and feature2 (Sentiment?)
    if "feature1" in train_merged.columns:
        train_merged["author_x_feature1"] = train_merged["author_mean_eng"] * train_merged["feature1"]
        test_merged["author_x_feature1"] = test_merged["author_mean_eng"] * test_merged["feature1"]
        
        # Unstable zone (40-60) where errors are high
        train_merged["is_unstable_zone"] = ((train_merged["feature1"] >= 40) & (train_merged["feature1"] <= 60)).astype(int)
        test_merged["is_unstable_zone"] = ((test_merged["feature1"] >= 40) & (test_merged["feature1"] <= 60)).astype(int)

    if "feature2" in train_merged.columns:
        # Interaction with raw feature2
        train_merged["author_x_feature2"] = train_merged["author_mean_eng"] * train_merged["feature2"]
        test_merged["author_x_feature2"] = test_merged["author_mean_eng"] * test_merged["feature2"]

        # Controversial flag (-5 seems to be high variance/viral potential)
        train_merged["is_controversial"] = (train_merged["feature2"] == -5).astype(int)
        test_merged["is_controversial"] = (test_merged["feature2"] == -5).astype(int)
        
        # Interaction with controversial status
        train_merged["author_x_controversial"] = train_merged["author_mean_eng"] * train_merged["is_controversial"]
        test_merged["author_x_controversial"] = test_merged["author_mean_eng"] * test_merged["is_controversial"]
        
    # NEW: Engagement Rate Proxy
    if "followers" in train_merged.columns:
        # Log of followers to match scale of author_mean_eng (which is already log1p)
        train_merged["log_followers"] = np.log1p(train_merged["followers"])
        test_merged["log_followers"] = np.log1p(test_merged["followers"])
        
        train_merged["author_eng_rate"] = train_merged["author_mean_eng"] / (train_merged["log_followers"] + 1)
        test_merged["author_eng_rate"] = test_merged["author_mean_eng"] / (test_merged["log_followers"] + 1)

    return train_merged, test_merged

def process_data():
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)

    print("1. Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    
    print(f"   Train shape: {train.shape}")
    print(f"   Test shape: {test.shape}")

    # --- Cleaning & Basic Prep ---
    print("2. Cleaning and Time Features...")
    
    # Cap engagement (Target) - Only on Train
    if "engagement" in train.columns:
        # We switch from LOG to SQRT transform
        # SQRT preserves more magnitude differences than LOG, helping the model
        # distinguish between "Big" (10k) and "Huge" (100k)
        train["transformed_engagement"] = np.sqrt(train["engagement"])

    # Time Features
    train = create_time_features(train)
    test = create_time_features(test)
    
    # --- Domain Features ---
    print("2b. Processing Domains...")
    train = process_domain(train)
    test = process_domain(test)
    
    # Identify Top Domains in Train
    top_domains = train["shared_url_domain"].value_counts().nlargest(20).index.tolist()
    
    # Apply mapping
    train["domain_group"] = train["shared_url_domain"].apply(lambda x: x if x in top_domains else "other")
    test["domain_group"] = test["shared_url_domain"].apply(lambda x: x if x in top_domains else "other")
    
    # We will drop the original raw domain column later

    # --- Author Features (New!) ---
    print("3. Adding Author Features...")
    train, test = add_author_features(train, test)
    
    # Now we can drop 'author' and other non-numeric cols
    # We keep 'domain_group' (it will be OneHotEncoded) but drop raw 'shared_url_domain'
    cols_to_drop = ["author", "shared_url_domain"]
    train = train.drop(columns=cols_to_drop, errors="ignore")
    test = test.drop(columns=cols_to_drop, errors="ignore")

    # Handle Booleans
    bool_cols = train.select_dtypes(include=["bool"]).columns.tolist()
    for c in bool_cols:
        train[c] = train[c].astype(int)
        if c in test.columns:
            test[c] = test[c].astype(int)

    # Treat feature2 as Categorical (it has discrete values -5, 0, 5)
    # if "feature2" in train.columns:
    #     train["feature2"] = train["feature2"].astype(str)
    #     test["feature2"] = test["feature2"].astype(str)
    # Reverting: Treating feature2 as categorical hurt performance.
    # It seems the ordinal nature (-5 < 0 < 5) is important for the model.
    pass

    # Fill NaNs in categorical features
    cat_cols = train.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        train[c] = train[c].fillna("missing")
        if c in test.columns:
            test[c] = test[c].fillna("missing")

    # --- Advanced Processing (V Columns) ---
    print("4. Processing Embeddings (PCA + Clustering)...")
    
    v_cols = [c for c in train.columns if c.startswith('V')]
    print(f"   Found {len(v_cols)} embedding columns.")

    # Extract V matrices
    X_train_v = train[v_cols].fillna(0).values
    X_test_v = test[v_cols].fillna(0).values
    
    # Combine for Transductive Learning (Better PCA/Scaling)
    X_all_v = np.vstack([X_train_v, X_test_v])

    # Scaling
    print("   Fitting Scaler on Train + Test...")
    scaler = RobustScaler()
    X_all_scaled = scaler.fit_transform(X_all_v)
    
    X_train_scaled = X_all_scaled[:len(train)]
    X_test_scaled = X_all_scaled[len(train):]
    
    # Save Scaler
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'scaler_v.pkl'))

    # PCA
    print(f"   Applying PCA (n={PCA_COMPONENTS}) on Train + Test...")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_all_pca = pca.fit_transform(X_all_scaled)
    
    X_train_pca = X_all_pca[:len(train)]
    X_test_pca = X_all_pca[len(train):]
    
    # Save PCA
    joblib.dump(pca, os.path.join(ARTIFACTS_DIR, 'pca_v.pkl'))

    # Add PCA components to DataFrames
    pca_cols = [f"pca_{i}" for i in range(PCA_COMPONENTS)]
    train_pca_df = pd.DataFrame(X_train_pca, columns=pca_cols, index=train.index)
    test_pca_df = pd.DataFrame(X_test_pca, columns=pca_cols, index=test.index)
    
    train = pd.concat([train, train_pca_df], axis=1)
    test = pd.concat([test, test_pca_df], axis=1)

    # Clustering (Macro)
    print(f"   Clustering Macro (k={CLUSTER_MACRO_K})...")
    kmeans_macro = KMeans(n_clusters=CLUSTER_MACRO_K, random_state=RANDOM_STATE, n_init=10)
    train["cluster_macro"] = kmeans_macro.fit_predict(X_train_pca)
    test["cluster_macro"] = kmeans_macro.predict(X_test_pca)
    joblib.dump(kmeans_macro, os.path.join(ARTIFACTS_DIR, 'kmeans_macro.pkl'))

    # Clustering (Micro)
    print(f"   Clustering Micro (k={CLUSTER_MICRO_K})...")
    kmeans_micro = KMeans(n_clusters=CLUSTER_MICRO_K, random_state=RANDOM_STATE, n_init=10)
    train["cluster_micro"] = kmeans_micro.fit_predict(X_train_pca)
    test["cluster_micro"] = kmeans_micro.predict(X_test_pca)
    joblib.dump(kmeans_micro, os.path.join(ARTIFACTS_DIR, 'kmeans_micro.pkl'))

    # Drop original V columns to save space
    print("5. Finalizing...")
    train = train.drop(columns=v_cols)
    test = test.drop(columns=v_cols)

    # Save to Parquet
    print("6. Saving to Parquet...")
    train_out = os.path.join(PROCESSED_DIR, 'train_processed.parquet')
    test_out = os.path.join(PROCESSED_DIR, 'test_processed.parquet')
    
    train.to_parquet(train_out, index=False)
    test.to_parquet(test_out, index=False)
    
    print("Done! Files saved:")
    print(f" - {train_out}")
    print(f" - {test_out}")
    print(f" - Models saved in {ARTIFACTS_DIR}/")

if __name__ == "__main__":
    process_data()
