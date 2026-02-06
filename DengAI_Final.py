import pandas as pd
import numpy as np
import xgboost as xgb

# 1. Load Data
train_raw = pd.read_csv('/kaggle/input/deng-ai-hack-ml-2026/train.csv')
test_raw  = pd.read_csv('/kaggle/input/deng-ai-hack-ml-2026/test.csv')

# 2. Winning Feature Engineering (377pt Logic)
def feature_engineering(df):
    df = df.sort_values(["city", "year", "weekofyear"]).reset_index(drop=True)
    
    # NDVI: Mean
    ndvi_cols = ['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw']
    df['ndvi_mean'] = df[ndvi_cols].mean(axis=1)
    
    # Seasonality
    df['week_sin'] = np.sin(2 * np.pi * df['weekofyear'] / 53)
    df['week_cos'] = np.cos(2 * np.pi * df['weekofyear'] / 53)
    
    # Climate Drivers from the 377-pt baseline
    climate_cols = [
        "reanalysis_specific_humidity_g_per_kg", 
        "reanalysis_dew_point_temp_k",
        "station_min_temp_c",
        "station_precip_mm",
        "ndvi_mean"
    ]
    
    # Impute
    for city in df['city'].unique():
        mask = df['city'] == city
        df.loc[mask, climate_cols] = df.loc[mask, climate_cols].interpolate(method='linear').ffill().bfill()

    # Biological Lags (4, 8, 12) & Rolling (8)
    for col in climate_cols:
        for lag in [4, 8, 12]:
            df[f"{col}_lag_{lag}"] = df.groupby("city")[col].shift(lag)
        df[f"{col}_roll_8"] = df.groupby("city")[col].transform(lambda x: x.rolling(window=8).mean())

    return df

# 3. Process
combo = pd.concat([train_raw.assign(is_train=1), test_raw.assign(is_train=0, total_cases=np.nan)], ignore_index=True)
combo = feature_engineering(combo)

train_feat = combo[combo["is_train"] == 1].copy()
test_feat  = combo[combo["is_train"] == 0].copy()

features = ['week_sin', 'week_cos'] + [c for c in train_feat.columns if "_lag_" in c or "_roll_" in c]

# 4. Modeling (Ensemble Strategy)
final_preds = []

for city in ["sj", "iq"]:
    c_train = train_feat[train_feat["city"] == city].dropna(subset=features + ["total_cases"])
    c_test  = test_feat[test_feat["city"] == city]
    
    X_train = c_train[features]
    y_train = c_train["total_cases"]
    X_test  = c_test[features]
    
    # --- [Model A] The 377-pt Tweedie Engine ---
    model_tweedie = xgb.XGBRegressor(
        objective='reg:tweedie',
        tweedie_variance_power=1.5, 
        n_estimators=1500,
        learning_rate=0.005,
        max_depth=4 if city == "sj" else 3,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42
    )
    
    # --- [Model B] The Robust Poisson Engine (to offset Tweedie's errors) ---
    model_poisson = xgb.XGBRegressor(
        objective='count:poisson',
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Training both models
    model_tweedie.fit(X_train, y_train)
    model_poisson.fit(X_train, y_train)
    
    # Predicting
    preds_tweedie = model_tweedie.predict(X_test)
    preds_poisson = model_poisson.predict(X_test)
    
    # --- [Weighted Average Ensemble] ---
    # Giving 70% weight to our champion model and 30% to the backup Poisson
    preds = (preds_tweedie * 0.7) + (preds_poisson * 0.3)
    
    # Post-processing: SJ Smoothing
    if city == "sj":
        preds = pd.Series(preds).rolling(window=5, min_periods=1, center=True).mean().values
        
    out = c_test[["id", "city", "year", "weekofyear"]].copy()
    out["total_cases"] = np.round(np.maximum(0, preds)).astype(int)
    final_preds.append(out)

# 5. Export
submission = pd.concat(final_preds).sort_values("id")
submission[['id', 'total_cases']].to_csv('submission_weighted_ensemble.csv', index=False)

print("🚀 Weighted Ensemble (7:3) saved! Let's hit the 300s!")