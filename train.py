import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn import set_config
set_config(transform_output="pandas")

warnings.filterwarnings("ignore")

DATA_DIR = "data"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_csv_file():
    preferred_names = [
        "track_data_final.csv",
        "spotify.csv",
        "spotify_data clean.csv",
        "spotify_data_clean.csv"
    ]

    for name in preferred_names:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            return path

    csv_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("ไม่พบไฟล์ CSV ในโฟลเดอร์ data/")
    return os.path.join(DATA_DIR, csv_files[0])


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("%", "percent", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("/", "_", regex=False)
    )
    return df


def convert_bool_columns(df: pd.DataFrame) -> pd.DataFrame:
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(int)

    for col in df.columns:
        if df[col].dtype == "object":
            unique_sample = set(
                str(x).strip().lower()
                for x in df[col].dropna().astype(str).head(100).tolist()
            )
            if unique_sample and unique_sample.issubset({"true", "false", "0", "1"}):
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .map({"true": 1, "false": 0, "1": 1, "0": 0})
                )
    return df


def choose_target_column(df: pd.DataFrame) -> str:
    possible_targets = [
        "track_popularity",
        "popularity",
        "song_popularity",
        "popularity_score"
    ]
    for col in possible_targets:
        if col in df.columns:
            return col
    raise ValueError(
        f"ไม่พบ target column สำหรับ popularity\ncolumns ที่มี: {list(df.columns)}"
    )


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "track_duration_ms" in df.columns and "duration_min" not in df.columns:
        df["duration_min"] = pd.to_numeric(df["track_duration_ms"], errors="coerce") / 60000.0

    if "album_release_date" in df.columns:
        dt = pd.to_datetime(df["album_release_date"], errors="coerce")
        df["release_year"] = dt.dt.year
        df["release_month"] = dt.dt.month

    if "artist_genres" in df.columns:
        df["primary_genre"] = (
            df["artist_genres"]
            .fillna("unknown")
            .astype(str)
            .str.split(",")
            .str[0]
            .str.strip()
            .replace("", "unknown")
        )

    return df


def select_features(df: pd.DataFrame, target_col: str):
    candidate_features = [
        "track_number",
        "track_duration_ms",
        "duration_min",
        "explicit",
        "artist_popularity",
        "artist_followers",
        "album_total_tracks",
        "release_year",
        "release_month",
        "album_type",
        "primary_genre"
    ]

    selected = [c for c in candidate_features if c in df.columns and c != target_col]
    return selected


def clean_dataframe(df: pd.DataFrame, target_col: str, selected_features):
    df = df.copy()

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col]).copy()
    df = df[(df[target_col] >= 0) & (df[target_col] <= 100)].copy()

    numeric_candidates = [
        "track_number",
        "track_duration_ms",
        "duration_min",
        "explicit",
        "artist_popularity",
        "artist_followers",
        "album_total_tracks",
        "release_year",
        "release_month"
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "duration_min" in df.columns:
        df = df[(df["duration_min"].isna()) | ((df["duration_min"] >= 0.5) & (df["duration_min"] <= 20))]

    if "release_year" in df.columns:
        df = df[(df["release_year"].isna()) | ((df["release_year"] >= 1950) & (df["release_year"] <= 2035))]

    if "artist_followers" in df.columns:
        df["artist_followers_log"] = np.log1p(df["artist_followers"])
        if "artist_followers_log" not in selected_features:
            selected_features.append("artist_followers_log")

    if "track_duration_ms" in selected_features and "duration_min" in selected_features:
        selected_features.remove("track_duration_ms")

    df = df.drop_duplicates().reset_index(drop=True)
    return df, selected_features


def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor, numeric_features, categorical_features


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    return {
        "model_name": name,
        "model": model,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2)
    }


def save_feature_importance(best_model, X: pd.DataFrame):
    try:
        model = best_model.named_steps["model"]

        
        if not hasattr(model, "feature_importances_"):
            return pd.DataFrame(columns=["feature", "importance"])

        importances = model.feature_importances_

        
        features = X.columns.tolist()

        
        min_len = min(len(features), len(importances))

        fi = pd.DataFrame({
            "feature": features[:min_len],
            "importance": importances[:min_len]
        }).sort_values("importance", ascending=False)

        fi.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
        return fi

    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])


def main():
    csv_path = find_csv_file()
    print(f"Using dataset: {csv_path}")

    df = pd.read_csv(csv_path)
    original_shape = df.shape

    df = normalize_columns(df)
    df = convert_bool_columns(df)
    df = create_engineered_features(df)

    print("\n=== ALL COLUMNS ===")
    print(df.columns.tolist())

    target_col = choose_target_column(df)
    selected_features = select_features(df, target_col)

    print("\n=== Selected Features (before cleaning) ===")
    print(selected_features)

    if len(selected_features) < 5:
        raise ValueError(
            f"เลือก feature ได้แค่ {len(selected_features)} ตัว: {selected_features}\n"
            f"น้อยเกินไปสำหรับงานนี้"
        )

    df, selected_features = clean_dataframe(df, target_col, selected_features)

    print("\n=== Selected Features (final) ===")
    print(selected_features)

    X = df[selected_features].copy()
    y = df[target_col].copy()

    cleaned_path = os.path.join(OUTPUT_DIR, "cleaned_data.csv")
    df[[*selected_features, target_col]].to_csv(cleaned_path, index=False)

    preprocessor, numeric_features, categorical_features = build_preprocessor(X)

    pipelines = {
        "LinearRegression": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ]),
        "RandomForest": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
        ]),
        "GradientBoosting": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(random_state=42))
        ])
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n=== Baseline Comparison ===")
    baseline_results = []

    for name, pipe in pipelines.items():
        result = evaluate_model(name, pipe, X_train, X_test, y_train, y_test)
        baseline_results.append(result)
        print(
            f"{name:18s} | MAE={result['mae']:.4f} | "
            f"RMSE={result['rmse']:.4f} | R2={result['r2']:.4f}"
        )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    rf_param_dist = {
        "model__n_estimators": [100, 150],
        "model__max_depth": [None, 10],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt"]
    }

    print("\n=== RandomizedSearchCV on RandomForest ===")
    rf_search = RandomizedSearchCV(
        estimator=rf_pipeline,
        param_distributions=rf_param_dist,
        n_iter=5,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    rf_search.fit(X_train, y_train)

    gb_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(random_state=42))
    ])

    gb_param_dist = {
        "model__n_estimators": [100, 150, 200],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
        "model__max_depth": [2, 3, 4],
        "model__subsample": [0.8, 0.9, 1.0],
        "model__min_samples_split": [2, 5, 10]
    }

    print("\n=== RandomizedSearchCV on GradientBoosting ===")
    gb_search = RandomizedSearchCV(
        estimator=gb_pipeline,
        param_distributions=gb_param_dist,
        n_iter=5,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    gb_search.fit(X_train, y_train)

    tuned_candidates = [
        ("RandomForest", rf_search.best_estimator_, rf_search.best_params_),
        ("GradientBoosting", gb_search.best_estimator_, gb_search.best_params_)
    ]

    best_name = None
    best_model = None
    best_params = None
    best_rmse = float("inf")
    best_metrics = None

    print("\n=== Tuned Model Comparison ===")
    for name, model, params in tuned_candidates:
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        print(f"{name:18s} | MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_model = model
            best_params = params
            best_metrics = (mae, rmse, r2)

    final_mae, final_rmse, final_r2 = best_metrics

    cv_scores = cross_val_score(
        best_model, X, y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    cv_rmse_scores = -cv_scores

    print("\n=== Final Best Model ===")
    print("Best Model :", best_name)
    print("Best Params:", best_params)
    print(f"Test MAE   : {final_mae:.4f}")
    print(f"Test RMSE  : {final_rmse:.4f}")
    print(f"Test R2    : {final_r2:.4f}")
    print(f"CV RMSE    : {cv_rmse_scores.mean():.4f} ± {cv_rmse_scores.std():.4f}")

    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(selected_features, os.path.join(MODEL_DIR, "feature_columns.pkl"))

    fi = save_feature_importance(best_model, X)

    metrics = {
        "dataset_path": csv_path,
        "original_shape": {
            "rows": int(original_shape[0]),
            "cols": int(original_shape[1])
        },
        "cleaned_shape": {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1])
        },
        "target_column": target_col,
        "selected_features": selected_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "baseline_models": [
            {
                "model_name": r["model_name"],
                "mae": r["mae"],
                "rmse": r["rmse"],
                "r2": r["r2"]
            }
            for r in baseline_results
        ],
        "best_model": best_name,
        "best_params": best_params,
        "test_mae": float(final_mae),
        "test_rmse": float(final_rmse),
        "test_r2": float(final_r2),
        "cv_rmse_mean": float(cv_rmse_scores.mean()),
        "cv_rmse_std": float(cv_rmse_scores.std()),
        "top_feature_importance": fi.head(10).to_dict(orient="records")
    }

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print("\nSaved:")
    print("- models/best_model.pkl")
    print("- models/feature_columns.pkl")
    print("- models/metrics.json")
    print("- outputs/cleaned_data.csv")
    print("- outputs/feature_importance.csv")


if __name__ == "__main__":
    main()
