import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


TRAIN_FILE = 'data_processed/train_processed.parquet'
MODEL_PATH = 'results/models/stacking_model.pkl'
RANDOM_STATE = 42


def analyze_errors():
    print("1. Loading data and model...")
    if not pd.io.common.file_exists(TRAIN_FILE):
        print(f"Error: {TRAIN_FILE} not found.")
        return
    df = pd.read_parquet(TRAIN_FILE)
    model = joblib.load(MODEL_PATH)
    y_trans = df['transformed_engagement'].values
    cols_to_drop = ['engagement', 'transformed_engagement', 'log_engagement', 'Id']
    X = df.drop(columns=cols_to_drop, errors='ignore')
    print("2. Replicating Train/Val Split...")
    X_train, X_val, y_train_trans, y_val_trans = train_test_split(
        X, y_trans, test_size=0.2, random_state=RANDOM_STATE
    )
    val_indices = X_val.index
    print(f"   Validation set size: {len(X_val)}")
    print("3. Predicting on Validation Set...")
    y_pred_trans = model.predict(X_val)
    y_val = y_val_trans ** 2
    y_pred = y_pred_trans ** 2
    y_pred = np.clip(y_pred, 0, None)
    residuals = y_val - y_pred
    abs_residuals = np.abs(residuals)
    squared_errors = residuals ** 2
    rmse = np.sqrt(np.mean(squared_errors))
    print(f"   Validation RMSE: {rmse:.4f}")
    analysis_df = X_val.copy()
    analysis_df['True_Engagement'] = y_val
    analysis_df['Predicted_Engagement'] = y_pred
    analysis_df['Error'] = residuals
    analysis_df['Abs_Error'] = abs_residuals
    print("\n--- Top 10 Worst Under-predictions (Viral posts missed) ---")
    under_predictions = analysis_df.sort_values('Error', ascending=False).head(10)
    print(under_predictions[['True_Engagement', 'Predicted_Engagement', 'Error', 'author_mean_eng', 'feature1', 'feature2']].to_string())
    print("\n--- Top 10 Worst Over-predictions (False alarms) ---")
    over_predictions = analysis_df.sort_values('Error', ascending=True).head(10)
    print(over_predictions[['True_Engagement', 'Predicted_Engagement', 'Error', 'author_mean_eng', 'feature1', 'feature2']].to_string())
    print("\n--- Correlation of Features with Absolute Error ---")
    num_cols = analysis_df.select_dtypes(include=[np.number]).columns
    correlations = analysis_df[num_cols].corrwith(analysis_df['Abs_Error']).sort_values(ascending=False)
    print("Top 10 features correlated with high error:")
    print(correlations.head(10))
    if 'feature2' in analysis_df.columns:
        print("\n--- Error by Feature 2 (Sentiment?) ---")
        f2_vals = analysis_df['feature2'].unique()
        print(f"Unique values of feature2 in Val: {f2_vals}")
        f2_stats = analysis_df.groupby('feature2').agg({
            'Abs_Error': 'mean',
            'Error': 'mean',
            'True_Engagement': 'mean',
            'Predicted_Engagement': 'mean',
            'feature2': 'count'
        }).rename(columns={'feature2': 'Count'})
        print(f2_stats)
    if 'feature1' in analysis_df.columns:
        print("\n--- Error vs Feature 1 (Score?) ---")
        analysis_df['f1_bin'] = pd.cut(analysis_df['feature1'], bins=5)
        f1_stats = analysis_df.groupby('f1_bin').agg({
            'Abs_Error': 'mean',
            'Error': 'mean',
            'True_Engagement': 'mean',
            'Predicted_Engagement': 'mean',
            'feature1': 'count'
        }).rename(columns={'feature1': 'Count'})
        print(f1_stats)


if __name__ == "__main__":
    analyze_errors()
