import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, KFold
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import joblib
import os


TRAIN_FILE = 'data_processed/train_processed.parquet'
ARTIFACTS_DIR = 'results/artifacts'
RANDOM_STATE = 42
N_ITER = 50  


def optimize_xgboost():
    print("1. Loading processed data...")
    if not os.path.exists(TRAIN_FILE):
        print(f"Error: {TRAIN_FILE} not found.")
        return
    df = pd.read_parquet(TRAIN_FILE)
    y = df['log_engagement'].values
    X = df.drop(columns=['engagement', 'log_engagement'])
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in ['cluster_macro', 'cluster_micro']:
        if col in X.columns and col not in cat_cols:
            cat_cols.append(col)
    num_cols = [c for c in X.columns if c not in cat_cols]
    preprocessor = ColumnTransformer([
        ('num', RobustScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('xgb', XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1))
    ])
    param_dist = {
        'xgb__n_estimators': [500, 1000, 1500, 2000],
        'xgb__learning_rate': [0.01, 0.03, 0.05, 0.1],
        'xgb__max_depth': [3, 5, 6, 8, 10],
        'xgb__min_child_weight': [1, 3, 5, 7],
        'xgb__gamma': [0, 0.1, 0.2, 0.5],
        'xgb__subsample': [0.6, 0.7, 0.8, 0.9],
        'xgb__colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'xgb__reg_alpha': [0, 0.01, 0.1, 1],
        'xgb__reg_lambda': [0.1, 1, 5, 10]
    }
    print(f"2. Starting Randomized Search ({N_ITER} iterations)...")
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=N_ITER,
        scoring='neg_root_mean_squared_error',
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        verbose=3,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    search.fit(X, y)
    print("\n3. Optimization Complete!")
    print(f"   Best RMSE: {-search.best_score_:.4f} (Log Scale)")
    print(f"   Best Parameters: {search.best_params_}")
    best_params_path = os.path.join(ARTIFACTS_DIR, 'best_xgb_params.pkl')
    joblib.dump(search.best_params_, best_params_path)
    print(f"   Saved best params to {best_params_path}")


if __name__ == "__main__":
    optimize_xgboost()
