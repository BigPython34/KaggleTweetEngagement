import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os


MODEL_PATH = 'results/models/stacking_model.pkl'
TRAIN_FILE = 'data_processed/train_processed.parquet'
OUTPUT_DIR = 'results'


def plot_feature_importance():
    print("1. Loading model and data...")
    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        return
    pipeline = joblib.load(MODEL_PATH)
    stacking_model = pipeline.named_steps['regressor']
    xgb_model = None
    if hasattr(stacking_model, 'named_estimators_'):
        xgb_model = stacking_model.named_estimators_.get('xgb')
    if xgb_model is None:
        for (name, _), fitted_model in zip(stacking_model.estimators, stacking_model.estimators_):
            if name == 'xgb':
                xgb_model = fitted_model
                break
    if xgb_model is None:
        print("XGBoost model not found in stack.")
        return
    print("2. Reconstructing feature names...")
    df = pd.read_parquet(TRAIN_FILE)
    cols_to_drop = ['engagement', 'transformed_engagement', 'log_engagement']
    if 'Id' in df.columns:
        cols_to_drop.append('Id')
    X = df.drop(columns=cols_to_drop, errors='ignore')
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in ['cluster_macro', 'cluster_micro']:
        if col in X.columns and col not in cat_cols:
            cat_cols.append(col)
    num_cols = [c for c in X.columns if c not in cat_cols]
    preprocessor = pipeline.named_steps['preprocessor']
    ohe = preprocessor.named_transformers_['cat']
    try:
        ohe_feature_names = ohe.get_feature_names_out(cat_cols)
        feature_names = num_cols + list(ohe_feature_names)
        if len(feature_names) != len(xgb_model.feature_importances_):
             print(f"Warning: Feature names count ({len(feature_names)}) != Model features ({len(xgb_model.feature_importances_)})")
             feature_names = [f"Feature_{i}" for i in range(len(xgb_model.feature_importances_))]
    except Exception as e:
        print(f"Could not retrieve feature names automatically: {e}")
        print("Using generic indices.")
        feature_names = [f"Feature_{i}" for i in range(xgb_model.feature_importances_.shape[0])]
    print("3. Plotting XGBoost Importance...")
    importances = xgb_model.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(12, 12))
    sns.barplot(x='Importance', y='Feature', data=fi_df.head(50), palette='viridis')
    plt.title('Top 50 Features Importance (XGBoost)')
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'feature_importance_xgb.png')
    plt.savefig(output_path)
    print(f"   Saved plot to {output_path}")
    print("\nTop 20 Features:")
    print(fi_df.head(20))
    print("\n--- Specific Feature Ranks ---")
    pca_feats = fi_df[fi_df['Feature'].str.contains('pca')]
    cluster_feats = fi_df[fi_df['Feature'].str.contains('cluster')]
    print(f"Top 3 PCA features:\n{pca_feats.head(3)}")
    print(f"Top 3 Cluster features:\n{cluster_feats.head(3)}")


if __name__ == "__main__":
    plot_feature_importance()
