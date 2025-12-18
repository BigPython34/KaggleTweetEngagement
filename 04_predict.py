import pandas as pd
import numpy as np
import joblib
import os


TEST_FILE = 'data_processed/test_processed.parquet'
MODEL_PATH = 'results/models/stacking_model.pkl'
OUTPUT_DIR = 'results'
SUBMISSION_FILE = 'submission.csv'


def make_predictions():
    print("1. Loading resources...")
    if not os.path.exists(TEST_FILE):
        print(f"Error: {TEST_FILE} not found.")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return
    df_test = pd.read_parquet(TEST_FILE)
    model = joblib.load(MODEL_PATH)
    print(f"   Test data shape: {df_test.shape}")
    if 'Id' in df_test.columns:
        ids = df_test['Id'].values
        df_test = df_test.drop(columns=['Id'])
    else:
        print("Warning: 'Id' column not found in test data. Generating sequential IDs.")
        ids = np.arange(1, len(df_test) + 1)
    print("2. Making predictions...")
    y_pred_trans = model.predict(df_test)
    y_pred = y_pred_trans ** 2
    y_pred = np.clip(y_pred, 0, None)
    print(f"   Predictions stats: Min={y_pred.min():.2f}, Max={y_pred.max():.2f}, Mean={y_pred.mean():.2f}")
    print("3. Creating submission file...")
    submission = pd.DataFrame({
        'Id': ids,
        'engagement': y_pred
    })
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    output_path = os.path.join(OUTPUT_DIR, SUBMISSION_FILE)
    submission.to_csv(output_path, index=False)
    print(f"Done! Submission saved to {output_path}")
    print("You can now upload this file to the competition.")


if __name__ == "__main__":
    make_predictions()
