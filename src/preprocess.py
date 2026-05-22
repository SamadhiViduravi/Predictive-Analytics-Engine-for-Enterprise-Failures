import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC

print("🔄 Initializing Production Data Preprocessing Pipeline...")

data_path = "dataset/"
processed_path = "data_processed/"
os.makedirs(processed_path, exist_ok=True)

# =========================================================================
# 1. LOAD RAW MICROSOFT COMPONENTS
# =========================================================================
print("Reading raw CSV components...")
telemetry = pd.read_csv(os.path.join(data_path, "PdM_telemetry.csv"), parse_dates=["datetime"])
errors     = pd.read_csv(os.path.join(data_path, "PdM_errors.csv"),    parse_dates=["datetime"])
failures   = pd.read_csv(os.path.join(data_path, "PdM_failures.csv"),  parse_dates=["datetime"])
machines   = pd.read_csv(os.path.join(data_path, "PdM_machines.csv"))

# Memory optimisation
telemetry["machineID"] = telemetry["machineID"].astype(np.int16)
for col in ["volt", "rotate", "pressure", "vibration"]:
    telemetry[col] = telemetry[col].astype(np.float32)

# =========================================================================
# 2. TIME-SERIES ROLLING WINDOW FEATURE ENGINEERING
# =========================================================================
print("Calculating historical rolling averages and standard deviations...")
telemetry = telemetry.sort_values(by=["machineID", "datetime"])
sensor_cols = ["volt", "rotate", "pressure", "vibration"]

telemetry_feat = telemetry[["datetime", "machineID"]].copy()
for col in sensor_cols:
    grp = telemetry.groupby("machineID")[col]
    telemetry_feat[f"{col}_mean_3h"]  = grp.rolling(window=3,  min_periods=1).mean().reset_index(level=0, drop=True).astype(np.float32)
    telemetry_feat[f"{col}_std_3h"]   = grp.rolling(window=3,  min_periods=1).std().fillna(0).reset_index(level=0, drop=True).astype(np.float32)
    telemetry_feat[f"{col}_mean_24h"] = grp.rolling(window=24, min_periods=1).mean().reset_index(level=0, drop=True).astype(np.float32)
    telemetry_feat[f"{col}_std_24h"]  = grp.rolling(window=24, min_periods=1).std().fillna(0).reset_index(level=0, drop=True).astype(np.float32)

# =========================================================================
# 3. ERROR LOG AGGREGATION
# =========================================================================
print("Aggregating error log matrix...")
error_dummies = pd.get_dummies(errors, columns=["errorID"], prefix="", prefix_sep="")
error_pivoted = error_dummies.groupby(["machineID", "datetime"]).sum().reset_index()
for col in error_pivoted.columns:
    if col not in ["machineID", "datetime"]:
        error_pivoted[col] = error_pivoted[col].astype(np.uint8)

final_df = pd.merge(telemetry_feat, error_pivoted, on=["machineID", "datetime"], how="left").fillna(0)

# =========================================================================
# 4. MACHINE METADATA
# =========================================================================
print("Integrating structural machine profiles...")
machines["machineID"] = machines["machineID"].astype(np.int16)
machines["age"]       = machines["age"].astype(np.uint8)
machines["model"]     = machines["model"].astype("category")
final_df = pd.merge(final_df, machines, on="machineID", how="left")

# =========================================================================
# 5. TARGET ENGINEERING — 24-HOUR PREDICTION WINDOW
# =========================================================================
# KEY INSIGHT: Instead of labelling only the exact failure hour (which gives
# the model almost zero signal), we label the 24 hours BEFORE each failure.
# This is the "prediction horizon" — the window in which we want to alert
# maintenance teams before the component actually breaks.
print("Building 24-hour prediction horizon failure labels...")
failures["machineID"] = failures["machineID"].astype(np.int16)

label_mapping = {"comp1": 1, "comp2": 2, "comp3": 3, "comp4": 4}
final_df["target"] = 0  # Default: nominal

for _, row in failures.iterrows():
    machine_id    = row["machineID"]
    failure_time  = row["datetime"]
    failure_class = label_mapping.get(row["failure"], 0)
    if failure_class == 0:
        continue

    # Mark all rows for this machine in the 24h window before failure
    window_start = failure_time - pd.Timedelta(hours=24)
    mask = (
        (final_df["machineID"] == machine_id) &
        (final_df["datetime"] >= window_start) &
        (final_df["datetime"] <= failure_time)
    )
    # Only overwrite if not already labelled (first failure wins)
    final_df.loc[mask & (final_df["target"] == 0), "target"] = failure_class

final_df["target"] = final_df["target"].astype(np.uint8)

failure_counts = final_df["target"].value_counts().sort_index()
print(f"Label distribution after windowing:\n{failure_counts}\n")

# =========================================================================
# 6. PREPARE FEATURE MATRIX
# =========================================================================
drop_cols = ["datetime", "machineID", "target"]
X = final_df.drop(columns=drop_cols)
y = final_df["target"]

# Datetime-derived features
datetime_series = final_df["datetime"]
X["hour"]        = datetime_series.dt.hour.astype(np.uint8)
X["day_of_week"] = datetime_series.dt.dayofweek.astype(np.uint8)

# Ensure categorical dtype for XGBoost native handling
X["model"] = X["model"].astype("category")

# =========================================================================
# 7. SPLIT FIRST — THEN SMOTE ON TRAINING DATA ONLY
# =========================================================================
# This is the critical fix. SMOTE must never see test data.
# Splitting first guarantees the test set contains only real sensor readings.
print("Performing stratified train/test split on REAL data before any resampling...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Real test set locked: {len(X_test):,} rows — untouched, no synthetic data.")
print(f"Training set before SMOTE: {len(X_train):,} rows")

# Apply SMOTE only to training data
categorical_features_mask = [i for i, col in enumerate(X_train.columns) if col == "model"]

target_resample_dict = {
    0: len(y_train[y_train == 0]),
    1: 5000,
    2: 5000,
    3: 5000,
    4: 5000
}

print("Synthesizing SMOTE-NC balance vectors on training partition only...")
smote_nc = SMOTENC(
    categorical_features=categorical_features_mask,
    sampling_strategy=target_resample_dict,
    random_state=42,
)
X_train_resampled, y_train_resampled = smote_nc.fit_resample(X_train, y_train)
print(f"Training set after SMOTE: {len(X_train_resampled):,} rows")

# =========================================================================
# 8. SAVE ALL SPLITS
# =========================================================================
print("Serializing train/test splits to disk...")

train_df = pd.DataFrame(X_train_resampled, columns=X_train.columns)
train_df["target"] = y_train_resampled

test_df = pd.DataFrame(X_test.values, columns=X_test.columns)
test_df["target"] = y_test.values

train_df.to_pickle(os.path.join(processed_path, "train_data.pkl"))
test_df.to_pickle(os.path.join(processed_path, "test_data.pkl"))

print(f"✅ Preprocessing complete!")
print(f"   Train (SMOTE-balanced): {os.path.join(processed_path, 'train_data.pkl')}")
print(f"   Test  (real data only): {os.path.join(processed_path, 'test_data.pkl')}")
