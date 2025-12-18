import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge, LassoCV
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

# Configuration
TRAIN_FILE = 'data_processed/train_processed.parquet'
ARTIFACTS_DIR = 'results/artifacts'
MODEL_DIR = 'results/models'
RANDOM_STATE = 42

def train_model():
    print("1. Loading processed data...")
    if not os.path.exists(TRAIN_FILE):
        print(f"Error: {TRAIN_FILE} not found. Run 02_process_data.py first.")
        return

    df = pd.read_parquet(TRAIN_FILE)
    
    # Define Target (Now using SQRT transform)
    y = df['transformed_engagement'].values
    
    # Drop target and Id if present
    cols_to_drop = ['engagement', 'transformed_engagement', 'log_engagement']
    if 'Id' in df.columns:
        cols_to_drop.append('Id')
        
    X = df.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"   Features shape: {X.shape}")
    
    # Identify columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    # Ensure clusters are treated as categorical even if they are integers
    for col in ['cluster_macro', 'cluster_micro']:
        if col in X.columns and col not in cat_cols:
            cat_cols.append(col)
            
    num_cols = [c for c in X.columns if c not in cat_cols]
    
    print(f"   Categorical columns: {cat_cols}")
    print(f"   Numerical columns: {len(num_cols)}")

    # Preprocessing Pipeline
    # We use RobustScaler for numericals (good for outliers)
    # OneHotEncoder for categoricals (including our clusters)
    # Added SimpleImputer to handle NaNs in numerical columns (e.g. in test set)
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
    
    # Define Base Models (The "Powerful" part)
    # We use a diverse set of strong models
    estimators = [
        ('ridge', Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ('xgb', XGBRegressor(
            n_estimators=2000, learning_rate=0.005, max_depth=8, # Slower learning, deeper trees
            subsample=0.7, colsample_bytree=0.6, min_child_weight=3,
            gamma=0.1, reg_alpha=0.5, reg_lambda=5,
            objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1
        )),
        # NEW: "Big Author" Specialist
        # Deeper trees (max_depth=10) to capture specific viral patterns of top authors
        ('xgb_deep', XGBRegressor(
            n_estimators=1500, learning_rate=0.005, max_depth=12, # Even deeper
            subsample=0.6, colsample_bytree=0.5, min_child_weight=1,
            objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1
        )),
        ('lgb', LGBMRegressor(
            n_estimators=1500, learning_rate=0.01, num_leaves=128, # More leaves
            subsample=0.7, colsample_bytree=0.7,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(256, 128, 64), activation="relu", # Deeper network
            solver="adam", max_iter=1000, random_state=RANDOM_STATE
        ))
    ]
    
    # Meta Learner
    # LassoCV is great for selecting the best combination of base models
    meta_learner = LassoCV(cv=5, random_state=RANDOM_STATE)
    
    # Stacking
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        passthrough=True # Pass original features to meta learner as well
    )
    
    # --- SIMPLIFICATION OPTION ---
    # If you want to test just XGBoost, uncomment the line below and comment out the stacking part
    # model_regressor = estimators[1][1] # Use the first XGBoost
    model_regressor = stacking_regressor # Default to stacking

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model_regressor)
    ])
    
    # Split for validation
    print("2. Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    # Train
    print("3. Training Stacking Model (Validation Step)...")
    model.fit(X_train, y_train)
    
    # Evaluate
    print("4. Evaluating...")
    y_pred_trans = model.predict(X_val)
    
    # Inverse transform (SQRT -> SQUARE)
    y_pred = y_pred_trans ** 2
    y_val_orig = y_val ** 2
    
    rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred))
    print(f"   Validation RMSE: {rmse:.4f}")
    
    # --- NEW: Refit on Full Data ---
    print("5. Re-training on FULL dataset (Train + Val)...")
    # This ensures we use 100% of the data for the final submission
    model.fit(X, y)
    
    # Save
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    print("5. Saving model...")
    model_path = os.path.join(MODEL_DIR, 'stacking_model.pkl')
    joblib.dump(model, model_path)
    print(f"   Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
