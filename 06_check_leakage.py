import pandas as pd
import os

# Configuration
DATA_DIR = 'data'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
AUTHOR_DATA_PATH = os.path.join(DATA_DIR, 'authorData.csv')

def check_leakage():
    print("1. Loading data...")
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(AUTHOR_DATA_PATH):
        print("Error: Data files not found.")
        return

    # Load train (drop V columns for comparison)
    train = pd.read_csv(TRAIN_PATH)
    v_cols = [c for c in train.columns if c.startswith('V')]
    train_core = train.drop(columns=v_cols)
    
    # Load authorData
    author_data = pd.read_csv(AUTHOR_DATA_PATH)
    
    print(f"   Train shape (core cols): {train_core.shape}")
    print(f"   AuthorData shape: {author_data.shape}")

    # 2. Check for duplicates
    print("2. Checking for overlap...")
    
    # We want to see if rows in train exist in authorData
    # We'll use a subset of columns that should uniquely identify a post if no ID is present
    # timestamp + author + engagement is a strong signature
    
    cols_to_match = ['timestamp', 'author', 'engagement', 'word_count']
    
    # Ensure columns exist in both
    cols_to_match = [c for c in cols_to_match if c in train_core.columns and c in author_data.columns]
    print(f"   Matching on columns: {cols_to_match}")
    
    # Merge to find intersection
    merged = train_core.merge(author_data, on=cols_to_match, how='inner')
    
    num_matches = len(merged)
    pct_matches = (num_matches / len(train)) * 100
    
    print(f"\nRESULTS:")
    print(f"Found {num_matches} matches out of {len(train)} training rows.")
    print(f"Overlap: {pct_matches:.2f}%")
    
    if pct_matches > 50:
        print("\n[CRITICAL WARNING] HIGH LEAKAGE DETECTED!")
        print("The 'authorData.csv' file contains the training samples.")
        print("Calculating 'author_mean_eng' using the full authorData file includes the target value itself.")
        print("This artificially inflates your validation score.")
        print("\nRECOMMENDATION: Use K-Fold Target Encoding instead of simple averaging.")
    elif pct_matches > 0:
        print("\n[WARNING] Some leakage detected.")
        print("Some training samples are in the author history.")
    else:
        print("\n[OK] No direct leakage detected (based on exact row match).")

if __name__ == "__main__":
    check_leakage()
