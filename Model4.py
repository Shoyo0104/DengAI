import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

# ----------------------------
# CONFIG
# ----------------------------
TRAIN_PATH = "/Users/aunghtetkhine/Desktop/deng-ai-hack-ml-2026/Lag/train_sj_lagged.csv"
TEST_PATH  = "/Users/aunghtetkhine/Desktop/deng-ai-hack-ml-2026/Lag/test_sj_lagged.csv"
OUT_PATH   = "sample2_submission_sj.csv"

TARGET = "total_cases"
ID_COLS = ["id", "city", "year", "weekofyear", "week_start_date"]
HOLDOUT_FRAC = 0.1  # time-based MAE check

# ----------------------------
# SAFE FEATURE ENGINEERING
# ----------------------------
def add_seasonality(df):
    df = df.copy()
    w = df["weekofyear"].astype(float)
    df["woy_sin"] = np.sin(2 * np.pi * w / 52.0)
    df["woy_cos"] = np.cos(2 * np.pi * w / 52.0)
    return df

def add_pr_context_flags(df):
    df = df.copy()
    df["week_start_date"] = pd.to_datetime(df["week_start_date"], errors="coerce")

    df["epidemic_1994_1995"] = (
        (df["week_start_date"] >= "1994-06-01") &
        (df["week_start_date"] <= "1995-05-31")
    ).astype(int)

    df["epidemic_year_1992"] = (df["year"] == 1992).astype(int)
    df["epidemic_year_1994"] = (df["year"] == 1994).astype(int)
    df["epidemic_year_1998"] = (df["year"] == 1998).astype(int)

    df["drought_1994"] = (df["year"] == 1994).astype(int)
    df["drought_1997"] = (df["year"] == 1997).astype(int)
    df["drought_2001_2002"] = df["year"].isin([2001, 2002]).astype(int)

    df["post_1994"] = (df["year"] >= 1995).astype(int)
    return df

# ----------------------------
# LOAD + SORT
# ----------------------------
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

train = train[train["city"] == "sj"].copy()
test  = test[test["city"] == "sj"].copy()

train["week_start_date"] = pd.to_datetime(train["week_start_date"], errors="coerce")
test["week_start_date"]  = pd.to_datetime(test["week_start_date"], errors="coerce")

train = train.sort_values(["year", "weekofyear", "week_start_date"]).reset_index(drop=True)
test  = test.sort_values(["year", "weekofyear", "week_start_date"]).reset_index(drop=True)

train = add_seasonality(add_pr_context_flags(train)).copy()
test  = add_seasonality(add_pr_context_flags(test)).copy()

# ----------------------------
# FEATURES
# ----------------------------
feature_cols = [
    c for c in train.columns
    if c not in ID_COLS + [TARGET]
    and pd.api.types.is_numeric_dtype(train[c])
]

# ----------------------------
# TIME-BASED HOLDOUT (MAE)
# ----------------------------
n = len(train)
split = int((1 - HOLDOUT_FRAC) * n)

tr = train.iloc[:split].copy()
va = train.iloc[split:].copy()

tr[feature_cols] = tr[feature_cols].ffill()
va[feature_cols] = va[feature_cols].ffill()

med = tr[feature_cols].median(numeric_only=True)
tr[feature_cols] = tr[feature_cols].fillna(med)
va[feature_cols] = va[feature_cols].fillna(med)

X_tr, y_tr = tr[feature_cols], tr[TARGET]
X_va, y_va = va[feature_cols], va[TARGET]

# ----------------------------
# MODEL (TREE-ONLY, MAE LOSS)
# ----------------------------
model = HistGradientBoostingRegressor(
    loss="absolute_error",   # directly minimizes MAE
    learning_rate=0.05,
    max_depth=6,
    max_leaf_nodes=96,
    min_samples_leaf=20,
    l2_regularization=1.0,
    early_stopping=True,
    random_state=42,
)

model.fit(X_tr, y_tr)
pred_va = np.clip(model.predict(X_va), 0, None)

mae = mean_absolute_error(y_va, pred_va)
print(f"[SJ] Holdout MAE: {mae:.4f}")

# ----------------------------
# TRAIN FULL + PREDICT TEST
# ----------------------------
train[feature_cols] = train[feature_cols].ffill().fillna(med)
test[feature_cols]  = test[feature_cols].ffill().fillna(med)

model.fit(train[feature_cols], train[TARGET])
pred_test = np.clip(np.round(model.predict(test[feature_cols])), 0, None).astype(int)

submission = test[["id"]].copy()
submission["total_cases"] = pred_test
submission.to_csv(OUT_PATH, index=False)

print(f"Saved: {OUT_PATH}")
