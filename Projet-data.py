from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, ElasticNetCV, LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_squared_error
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import warnings


warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
df = pd.read_csv(r'train.csv')
drop_cols = ["author", "shared_url_domain"]
cap_value = df["engagement"].quantile(0.985)
df["engagement"] = df["engagement"].clip(upper=cap_value)
data = df.drop(columns=drop_cols, errors="ignore").copy()
bool_cols = data.select_dtypes(include=["bool"]).columns.tolist()
for c in bool_cols:
    data[c] = data[c].astype(int)
y_raw = data["engagement"].astype(float).values
y = np.log1p(y_raw)
X_full = data.drop(columns=["engagement"], errors="ignore")
cluster_grid = [7,10,12,15, 20, 30]
pca_grid = [10,20,30,50]
time_options = [True]
meta_options = ["lasso", "elastic"]
results = []
for n_clusters in cluster_grid:
    for pca_components in pca_grid:
        for use_time_features in time_options:
            for meta_choice in meta_options:
                print(f"\n=� Running config: clusters={n_clusters}, PCA={pca_components}, time={use_time_features}, meta={meta_choice}")
                X = X_full.copy()
                if use_time_features and "timestamp" in X.columns:
                    ts = pd.to_datetime(X["timestamp"], unit="ms", errors="coerce")
                    X["_hour"] = ts.dt.hour
                    X["_dow"] = ts.dt.dayofweek
                    X["_month"] = ts.dt.month
                    X["_is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
                    X["_hour_sin"] = np.sin(2 * np.pi * X["_hour"] / 24)
                    X["_hour_cos"] = np.cos(2 * np.pi * X["_hour"] / 24)
                    X = X.drop(columns=["timestamp"], errors="ignore")
                elif "timestamp" in X.columns:
                    X = X.drop(columns=["timestamp"], errors="ignore")
                num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
                cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
                embedding_cols = [c for c in X.columns if c.startswith("V")]
                non_embedding_num = [c for c in num_cols if c not in embedding_cols]
                if cat_cols:
                    X[cat_cols] = X[cat_cols].fillna("missing")
                emb_scaler = RobustScaler()
                emb_scaled = emb_scaler.fit_transform(X[embedding_cols].fillna(0).values)
                kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, random_state=RANDOM_STATE)
                emb_labels = kmeans.fit_predict(emb_scaled)
                X["emb_cluster"] = pd.Series(emb_labels, index=X.index).astype("category")
                cat_cols = (cat_cols or []) + ["emb_cluster"]
                num_non_emb_transformer = Pipeline([("scaler", RobustScaler())])
                embedding_transformer = Pipeline([
                    ("scaler", RobustScaler()),
                    ("pca", PCA(n_components=pca_components, random_state=RANDOM_STATE))
                ])
                categorical_transformer = Pipeline([
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                ])
                preprocessor = ColumnTransformer([
                    ("num", num_non_emb_transformer, non_embedding_num),
                    ("emb", embedding_transformer, embedding_cols),
                    ("cat", categorical_transformer, cat_cols)
                ])
                ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
                xgb = XGBRegressor(
                    n_estimators=500, learning_rate=0.05, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8,
                    objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1
                )
                mlp = MLPRegressor(
                    hidden_layer_sizes=(128, 64), activation="relu",
                    solver="adam", learning_rate_init=0.001, alpha=0.001,
                    max_iter=400, random_state=RANDOM_STATE
                )
                lgb = LGBMRegressor(
                    n_estimators=700, learning_rate=0.03, max_depth=-1,
                    num_leaves=64, subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=-1
                )
                elastic = ElasticNetCV(cv=5, random_state=RANDOM_STATE, l1_ratio=0.5, n_jobs=-1, max_iter=5000)
                estimators = [
                    ("ridge", ridge),
                    ("xgb", xgb),
                    ("mlp", mlp),
                    ("lgb", lgb),
                    ("elastic", elastic),
                ]
                if meta_choice == "lasso":
                    meta = LassoCV(
                        alphas=np.logspace(-3, 1, 10),
                        cv=5, random_state=RANDOM_STATE,
                        max_iter=10000, n_jobs=-1
                    )
                elif meta_choice == "elastic":
                    meta = ElasticNetCV(
                        l1_ratio=[0.2, 0.5, 0.8],
                        alphas=np.logspace(-3, 1, 10),
                        cv=5, random_state=RANDOM_STATE,
                        n_jobs=-1, max_iter=10000
                    )
                else:
                    meta = XGBRegressor(
                        n_estimators=400, learning_rate=0.02, max_depth=5,
                        subsample=0.8, colsample_bytree=0.8,
                        objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1
                    )
                stack = StackingRegressor(
                    estimators=estimators,
                    final_estimator=meta,
                    cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                    n_jobs=-1,
                    passthrough=True
                )
                model = Pipeline([
                    ("preprocessor", preprocessor),
                    ("stack", stack)
                ])
                X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
                    X, y, y_raw, test_size=0.2, random_state=RANDOM_STATE
                )
                model.fit(X_train, y_train)
                y_pred_log = model.predict(X_test)
                y_pred = np.expm1(y_pred_log)
                y_pred = np.clip(y_pred, 0, y_raw_train.max())
                rmse = np.sqrt(mean_squared_error(y_raw_test, y_pred))
                results.append({
                    "n_clusters": n_clusters,
                    "pca": pca_components,
                    "time_features": use_time_features,
                    "meta": meta_choice,
                    "rmse": rmse
                })
                print(f" RMSE: {rmse:.4f}")
res_df = pd.DataFrame(results).sort_values("rmse")
print("\n<� Top Configurations:")
print(res_df.head(10))
best_cfg = res_df.iloc[0]
print(f"\n=% Best: clusters={best_cfg.n_clusters}, PCA={best_cfg.pca}, time={best_cfg.time_features}, meta={best_cfg.meta}, RMSE={best_cfg.rmse:.4f}")
