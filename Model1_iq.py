import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

# ----------------------------
# CONFIG (EDIT THESE PATHS)
# ----------------------------
TRAIN_PATH = "/Users/aunghtetkhine/Desktop/deng-ai-hack-ml-2026/Lag/train_iq_lagged.csv"
TEST_PATH  = "/Users/aunghtetkhine/Desktop/deng-ai-hack-ml-2026/Lag/test_iq_lagged.csv"
OUT_PATH   = "submission_iq.csv"

TARGET = "total_cases"
ID_COLS = ["id", "city", "year", "weekofyear", "week_start_date"]
HOLDOUT_FRAC = 0.1  # time-based MAE check (last 10%)

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
def add_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    w = df["weekofyear"].astype(float)
    df["woy_sin"] = np.sin(2 * np.pi * w / 52.0)
    df["woy_cos"] = np.cos(2 * np.pi * w / 52.0)
    return df

def add_iq_context_flags_2000_2010(df: pd.DataFrame) -> pd.DataFrame:
    """
    Iquitos-specific context inspired by published surveillance patterns:
      - DENV-3 dominated ~2001–2007
      - DENV-4 dominated ~2008–2010
      - Large outbreak seasons reported around 2000–03 and 2008–10
      - DHF noted during a DENV-3 epidemic around 2004
    We encode these as time-window indicator flags to help the model learn regime shifts.
    """
    df = df.copy()
    df["week_start_date"] = pd.to_datetime(df["week_start_date"], errors="coerce")

    # Serotype-era proxy flags (time windows)
    df["iq_era_denv3_2001_2007"] = df["year"].between(2001, 2007).astype(int)
    df["iq_era_denv4_2008_2010"] = df["year"].between(2008, 2010).astype(int)

    # Outbreak-window proxy flags (broad)
    df["iq_outbreak_window_2000_2003"] = df["year"].between(2000, 2003).astype(int)
    df["iq_outbreak_window_2008_2010"] = df["year"].between(2008, 2010).astype(int)

    # DHF mention around 2004 during DENV-3 epidemic (proxy)
    df["iq_dhf_proxy_2004"] = (df["year"] == 2004).astype(int)

    # Optional: rainy-season emphasis (Iquitos outbreaks often peak ~Oct–Apr)
    # Approx weeks: 40..52 and 1..17
    w = df["weekofyear"].astype(int)
    df["iq_rainy_season_proxy"] = ((w >= 40) | (w <= 17)).astype(int)

    # Post-2001 regime (after major DENV-3 intro)
    df["iq_post_2001"] = (df["year"] >= 2002).astype(int)

    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_seasonality(df)
    df = add_iq_context_flags_2000_2010(df)
    return df

# ----------------------------
# MAIN
# ----------------------------
def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    # Feature engineering
    train = build_features(train)
    test = build_features(test)

    # Sort by time to avoid leakage + create time-based holdout
    train["week_start_date"] = pd.to_datetime(train["week_start_date"], errors="coerce")
    train = train.sort_values("week_start_date").reset_index(drop=True)

    n = len(train)
    holdout_n = max(1, int(np.ceil(HOLDOUT_FRAC * n)))
    split_idx = n - holdout_n

    train_fit = train.iloc[:split_idx].copy()
    train_val = train.iloc[split_idx:].copy()

    # Prepare X/y
    feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]

    X_fit = train_fit[feature_cols]
    y_fit = train_fit[TARGET].astype(float)

    X_val = train_val[feature_cols]
    y_val = train_val[TARGET].astype(float)

    # Model (counts-friendly)
    model = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.05,
        max_depth=6,
        max_iter=1200,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=42,
    )

    model.fit(X_fit, y_fit)

    # Time-based validation
    val_pred = model.predict(X_val)
    val_pred = np.clip(val_pred, 0, None)  # keep non-negative
    mae = mean_absolute_error(y_val, val_pred)
    print(f"[IQ] Time-holdout MAE (last {holdout_n} weeks): {mae:.4f}")

    # Refit on full train
    X_full = train[feature_cols]
    y_full = train[TARGET].astype(float)
    model.fit(X_full, y_full)

    # Predict test
    test_pred = model.predict(test[feature_cols])
    test_pred = np.clip(test_pred, 0, None)

    # Optional: integer-ish output (often helps for submission sanity)
    # If you want raw floats, comment the next line out.
    test_pred = np.rint(test_pred).astype(int)

    sub = pd.DataFrame({
        "id": test["id"],
        "total_cases": test_pred
    })

    sub.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")
    print(sub.head(10))

if __name__ == "__main__":
    main()
