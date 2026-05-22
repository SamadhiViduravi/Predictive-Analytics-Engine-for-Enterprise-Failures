import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

print("🚀 Starting Precision-Calibrated XGBoost Multi-Class Training Pipeline...")

# =========================================================================
# 1. LOAD PRE-SPLIT DATA
# =========================================================================
processed_path = "data_processed/"
train_file = os.path.join(processed_path, "train_data.pkl")
test_file  = os.path.join(processed_path, "test_data.pkl")

if not os.path.exists(train_file) or not os.path.exists(test_file):
    raise FileNotFoundError("Missing split files. Run 'python src/preprocess.py' first.")

print("Loading SMOTE-balanced training matrix...")
train_df = pd.read_pickle(train_file)
test_df  = pd.read_pickle(test_file)

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"].astype(np.uint8)

X_test = test_df.drop(columns=["target"])
y_test = test_df["target"].astype(np.uint8)

print(f"Training on {len(X_train):,} rows (SMOTE-augmented)")
print(f"Evaluating on {len(X_test):,} rows (real sensor data only)\n")

# =========================================================================
# 2. ENFORCE CATEGORICAL DTYPE AFTER PICKLE RELOAD
# =========================================================================
# Pickle does not always preserve pandas category dtype for string columns.
# We explicitly re-cast any object/string columns in both splits, then align
# their category levels so XGBoost sees a consistent encoding.
CATEGORICAL_COLS = ["model"]

for col in CATEGORICAL_COLS:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype(str).astype("category")
        X_test[col]  = X_test[col].astype(str).astype("category")

        # Align category levels: test must know all categories seen in train
        combined_cats = X_train[col].cat.categories.union(X_test[col].cat.categories)
        X_train[col] = X_train[col].cat.set_categories(combined_cats)
        X_test[col]  = X_test[col].cat.set_categories(combined_cats)

print(f"Categorical columns aligned: {CATEGORICAL_COLS}")
# SMOTE converts numeric columns to object dtype — cast them all back
for col in X_train.columns:
    if col not in CATEGORICAL_COLS:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col]  = pd.to_numeric(X_test[col],  errors="coerce")

# =========================================================================
# 3. TRAIN XGBOOST
# =========================================================================
print("Configuring regularized Multi-Class XGBoost engine...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    reg_alpha=1.0,
    reg_lambda=3.0,
    objective="multi:softprob",
    num_class=5,
    random_state=42,
    eval_metric="mlogloss",
    enable_categorical=True,
    tree_method="hist",
    n_jobs=-1
)

print("Fitting model on SMOTE-augmented training data...")
model.fit(X_train, y_train)

# =========================================================================
# 4. EVALUATE ON REAL HELD-OUT DATA
# =========================================================================
print("Evaluating on real (non-synthetic) test set...\n")
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)

target_names = [
    "Nominal Operational State",
    "Component 1 Failure",
    "Component 2 Failure",
    "Component 3 Failure",
    "Component 4 Failure"
]

print("=" * 55)
print("   VERIFICATION REPORT (REAL DATA ONLY — NO SMOTE)")
print("=" * 55)
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

try:
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    print(f"Macro ROC-AUC (OvR): {auc:.4f}")
except Exception as e:
    print(f"ROC-AUC skipped: {e}")

print("=" * 55)

# =========================================================================
# 5. SAVE ARTIFACTS
# =========================================================================
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/xgb_multi_model.pkl")
joblib.dump(list(X_train.columns), "model/feature_columns.pkl")
print("\n✅ Model artifacts saved to /model/")
print("   Run 'uvicorn src.main:app --reload' to start the API.")