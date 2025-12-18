import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
TRAIN_PARQUET = 'data_processed/train_processed.parquet'
TRAIN_CSV = 'data/train.csv'
MODEL_PATH = 'results/models/stacking_model.pkl'
OUTPUT_DIR = 'results'

def analyze_errors():
    print("1. Loading data and model...")
    if not os.path.exists(TRAIN_PARQUET) or not os.path.exists(MODEL_PATH):
        print("Files not found.")
        return

    # Load processed data for prediction
    df_proc = pd.read_parquet(TRAIN_PARQUET)
    
    # Load original data for context (text, author name, etc.)
    df_orig = pd.read_csv(TRAIN_CSV)
    
    # Load model
    model = joblib.load(MODEL_PATH)
    
    # Prepare X and y
    # Note: The model pipeline expects the columns present in df_proc (minus target)
    # We need to handle columns exactly as in training
    y_true_log = df_proc['log_engagement']
    
    cols_to_drop = ['engagement', 'log_engagement']
    if 'Id' in df_proc.columns:
        cols_to_drop.append('Id')
        
    X = df_proc.drop(columns=cols_to_drop)
    
    print("2. Making predictions on training set...")
    y_pred_log = model.predict(X)
    
    # Convert back to original scale
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.clip(y_pred, 0, None)
    
    # Calculate Errors
    residuals = y_true - y_pred
    squared_errors = residuals ** 2
    
    # Create analysis dataframe
    analysis = df_orig.copy()
    analysis['True_Engagement'] = y_true
    analysis['Predicted_Engagement'] = y_pred
    analysis['Error'] = residuals
    analysis['Abs_Error'] = residuals.abs()
    analysis['Squared_Error'] = squared_errors
    
    # Sort by Squared Error (MSE contributors)
    worst_errors = analysis.sort_values('Squared_Error', ascending=False)
    
    print("\n=== TOP 10 WORST PREDICTIONS (Biggest MSE Contributors) ===")
    cols_to_show = ['author', 'followers', 'True_Engagement', 'Predicted_Engagement', 'Error']
    print(worst_errors[cols_to_show].head(10))
    
    print("\n=== ERROR ANALYSIS ===")
    print(f"Total MSE on Train: {squared_errors.mean():.2f}")
    print(f"RMSE on Train: {np.sqrt(squared_errors.mean()):.2f}")
    
    # Underestimation vs Overestimation
    underestimated = analysis[analysis['Error'] > 0]
    overestimated = analysis[analysis['Error'] < 0]
    
    print(f"\nUnderestimated count: {len(underestimated)} (Model predicted too low)")
    print(f"Overestimated count: {len(overestimated)} (Model predicted too high)")
    
    print(f"Avg Error when Underestimating: {underestimated['Abs_Error'].mean():.2f}")
    print(f"Avg Error when Overestimating: {overestimated['Abs_Error'].mean():.2f}")

    # Plot True vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True Engagement')
    plt.ylabel('Predicted Engagement')
    plt.title('True vs Predicted Engagement (Training Set)')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'error_analysis_scatter.png'))
    print(f"\nSaved scatter plot to {os.path.join(OUTPUT_DIR, 'error_analysis_scatter.png')}")

if __name__ == "__main__":
    analyze_errors()
