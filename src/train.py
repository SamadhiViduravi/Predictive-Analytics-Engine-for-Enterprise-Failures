import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


def train_and_verify_model():
    print("Initializing Model Training & Verification Pipeline...")

    # 1. Load Preprocessed Data Splits
    data_dir = "dataset"
    try:
        X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
        y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    except FileNotFoundError as e:
        raise FileNotFoundError("Processed files missing. Execute src/preprocess.py first.") from e

    # 2. Configure XGBoost to handle class imbalance
    # scale_pos_weight balances the weight of the rare failure classes
    negative_instances = len(y_train) - sum(y_train)
    positive_instances = sum(y_train)
    imbalance_ratio = negative_instances / positive_instances

    print(f"Dataset imbalance ratio calculated: 1:{imbalance_ratio:.2f}")

    # 3. Initialize Model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=imbalance_ratio,  # Crucial for highly accurate minority class capture
        random_state=42,
        eval_metric="logloss"
    )

    # 4. Train the Model
    print("Fitting XGBoost Classifier on scaled training matrices...")
    model.fit(X_train, y_train)

    # 5. Predict on Unseen Test Data to Verify Accuracy
    print("Executing predictive inference on verification subset...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 6. Generate Verification Report
    print("\n" + "=" * 50)
    print("         ENTERPRISE ENGINE VERIFICATION REPORT")
    print("=" * 50)

    # Print classic precision, recall, f1 metrics
    print(classification_report(y_test, y_pred, target_names=["Normal", "Failure"]))

    # Print area under curve
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Performance Metric: {auc:.4f}")

    # Print Raw Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix Array Breakdown:")
    print(f"  True Normals (Correctly Predicted Safe):  {cm[0][0]}")
    print(f"  False Alarms (Wrongly Flapped Alarm):     {cm[0][1]}")
    print(f"  Missed Breakdowns (Catastrophic Escapes): {cm[1][0]}")
    print(f"  True Failures (Correctly Caught Breaks):  {cm[1][1]}")
    print("=" * 50)

    # 7. Serialize trained model artifact
    model_output_path = os.path.join(data_dir, "predictive_model.pkl")
    joblib.dump(model, model_output_path)
    print(f"\nModel verified and successfully serialized to: {model_output_path}")


if __name__ == "__main__":
    train_and_verify_model()