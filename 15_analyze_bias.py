import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import os


TRAIN_PARQUET = 'data_processed/train_processed.parquet'
MODEL_PATH = 'results/models/stacking_model.pkl'
OUTPUT_DIR = 'results'
RANDOM_STATE = 42


def analyze_bias():
    print("1. Loading data...")
    if not os.path.exists(TRAIN_PARQUET) or not os.path.exists(MODEL_PATH):
        print("Files not found.")
        return
    df = pd.read_parquet(TRAIN_PARQUET)
    y_true_log = df['log_engagement'].values
    y_true = np.expm1(y_true_log)
    cols_to_drop = ['engagement', 'log_engagement']
    if 'Id' in df.columns:
        cols_to_drop.append('Id')
    X = df.drop(columns=cols_to_drop)
    pipeline = joblib.load(MODEL_PATH)
    print("2. Generating Out-Of-Fold predictions (Unbiased)...")
    y_pred_log = cross_val_predict(
        pipeline, 
        X, 
        y_true_log, 
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1
    )
    y_pred = np.expm1(y_pred_log)
    y_pred = np.clip(y_pred, 0, None)
    print("3. Analyzing Bias vs Prediction Magnitude...")
    results = pd.DataFrame({
        'True': y_true,
        'Pred': y_pred
    })
    results['Ratio'] = results['True'] / (results['Pred'] + 1)
    results['Pred_Bin'] = pd.cut(results['Pred'], bins=[0, 100, 1000, 5000, 10000, 50000, 100000, 1000000, np.inf])
    bias_analysis = results.groupby('Pred_Bin', observed=True)['Ratio'].agg(['mean', 'count', 'median'])
    print("\n=== Bias Analysis (Ratio True/Pred) ===")
    print(bias_analysis)
    high_val_mask = results['Pred'] > 1000
    if high_val_mask.sum() > 50:
        X_bias = np.log1p(results.loc[high_val_mask, 'Pred']).values.reshape(-1, 1)
        y_bias = results.loc[high_val_mask, 'Ratio'].values
        reg = LinearRegression()
        reg.fit(X_bias, y_bias)
        a = reg.coef_[0]
        b = reg.intercept_
        print(f"\n=== Suggested Correction Model (for Pred > 1000) ===")
        print(f"Modeled Ratio = {a:.4f} * log1p(Pred) + {b:.4f}")
        results['Pred_Corrected'] = results['Pred']
        correction_factor = reg.predict(X_bias)
        correction_factor = np.maximum(1.0, correction_factor)
        results.loc[high_val_mask, 'Pred_Corrected'] = results.loc[high_val_mask, 'Pred'] * correction_factor
        mse_orig = ((results['True'] - results['Pred'])**2).mean()
        mse_corr = ((results['True'] - results['Pred_Corrected'])**2).mean()
        print(f"\nOriginal MSE: {mse_orig:.2f}")
        print(f"Corrected MSE: {mse_corr:.2f}")
        print(f"Improvement: {(mse_orig - mse_corr) / mse_orig * 100:.2f}%")
        plt.figure(figsize=(10, 6))
        plt.scatter(results.loc[high_val_mask, 'Pred'], results.loc[high_val_mask, 'Ratio'], alpha=0.3, label='Actual Ratios')
        x_range = np.linspace(1000, results['Pred'].max(), 100)
        y_model = reg.predict(np.log1p(x_range).reshape(-1, 1))
        plt.plot(x_range, y_model, 'r-', lw=3, label='Fitted Correction')
        plt.xscale('log')
        plt.xlabel('Predicted Engagement')
        plt.ylabel('Ratio (True / Predicted)')
        plt.title('Systematic Bias Analysis')
        plt.legend()
        plt.grid(True, which="both", ls="-")
        plt.savefig(os.path.join(OUTPUT_DIR, 'bias_correction.png'))
        print(f"Saved plot to {os.path.join(OUTPUT_DIR, 'bias_correction.png')}")
        return a, b
    else:
        print("Not enough high value predictions to fit a correction.")
        return 0, 1


if __name__ == "__main__":
    analyze_bias()
