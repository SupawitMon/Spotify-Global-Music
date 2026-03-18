import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "data"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_csv_file():
    preferred_names = [
        "spotify.csv",
        "spotify_data clean.csv",
        "spotify_data_clean.csv",
        "track_data_final.csv"
    ]

    for name in preferred_names:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            return path

    csv_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("ไม่พบไฟล์ CSV ใน data/")
    return os.path.join(DATA_DIR, csv_files[0])


def normalize_columns(df):
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


def main():
    csv_path = find_csv_file()
    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass

    if "duration_ms" in df.columns and "duration_min" not in df.columns:
        df["duration_min"] = pd.to_numeric(df["duration_ms"], errors="coerce") / 60000

    report = []
    report.append("=== EDA SUMMARY ===")
    report.append(f"Dataset path: {csv_path}")
    report.append(f"Shape: {df.shape}")
    report.append("")
    report.append("Columns:")
    report.extend(df.columns.tolist())
    report.append("")
    report.append("Missing values:")
    report.append(df.isnull().sum().to_string())
    report.append("")
    report.append("Describe numeric:")
    report.append(df.describe(include="number").to_string())

    with open(os.path.join(OUTPUT_DIR, "eda_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    if "popularity" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.hist(df["popularity"].dropna(), bins=30)
        plt.title("Popularity Distribution")
        plt.xlabel("Popularity")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "popularity_distribution.png"), dpi=200)
        plt.close()

    features = [
        "danceability", "energy", "loudness", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "tempo", "duration_min"
    ]
    available = [c for c in features if c in df.columns]

    if available and "popularity" in df.columns:
        corr = df[available + ["popularity"]].corr(numeric_only=True)["popularity"].drop("popularity")
        corr = corr.sort_values()

        plt.figure(figsize=(9, 5))
        plt.barh(corr.index, corr.values)
        plt.title("Correlation with Popularity")
        plt.xlabel("Correlation")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "feature_correlation.png"), dpi=200)
        plt.close()

    for feat in ["danceability", "energy", "valence", "tempo"]:
        if feat in df.columns:
            plt.figure(figsize=(8, 5))
            plt.hist(df[feat].dropna(), bins=30)
            plt.title(f"{feat.capitalize()} Distribution")
            plt.xlabel(feat)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"{feat}_distribution.png"), dpi=200)
            plt.close()

    print("EDA completed")
    print("Saved files in outputs/")


if __name__ == "__main__":
    main()