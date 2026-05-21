import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run_preprocessing_pipeline():
    print("Initializing Data Preprocessing Pipeline...")

    # Paths
    raw_data_path = os.path.join("dataset", "industrial_telemetry.csv")
    output_dir = "dataset"

    # 1. Load Raw Dataset
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_data_path}. Run generate_dataset.py first.")
    df = pd.read_csv(raw_data_path)

    # 2. Feature Engineering: Capture Heat Dissipation Signal
    print("Engineering physical features...")
    df['Temp_Difference_K'] = df['Process_Temperature_K'] - df['Air_Temperature_K']

    # 3. Separate Features and Target Matrix
    X = df.drop(columns=['Machine_ID', 'Failure'])
    y = df['Failure']

    # 4. Stratified Train/Test Split (80/20)
    print("Splitting datasets with target stratification...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Feature Scaling
    print("Normalizing physical features using StandardScaler...")
    feature_columns = X_train.columns
    scaler = StandardScaler()

    # Fit on training data ONLY to prevent data leakage
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to clean DataFrames
    X_train_final = pd.DataFrame(X_train_scaled, columns=feature_columns)
    X_test_final = pd.DataFrame(X_test_scaled, columns=feature_columns)

    # 6. Serialize and Save Artifacts
    print("Saving processed arrays and scaler artifact to disk...")
    joblib.dump(scaler, os.path.join(output_dir, "numerical_scaler.pkl"))

    X_train_final.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test_final.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print(f"Pipeline complete! Processed sets successfully exported to standard path: '{output_dir}/'")


if __name__ == "__main__":
    run_preprocessing_pipeline()