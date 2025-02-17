from pathlib import Path
import configparser
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Define directories
dataset = Path("dataset.csv")
system_configs_dir = Path("system-configs")
des_output_conv_dir = Path("DES_output_conv")
des_output_gemm_dir = Path("DES_output_gemm")
cycle_col_name = " Total Cycles"

data = []
if dataset.exists():
    df = pd.read_csv(dataset)
else:
    print("dataset doesn't exist, preparing ...")
    for config_path in system_configs_dir.glob("system_*"):
        uuid = config_path.stem.split("_")[1]
        conv_path = des_output_conv_dir / f"lenet_DSE_run_{uuid}" / "COMPUTE_REPORT.csv"
        gemm_path = des_output_gemm_dir / f"lenet_DSE_run_{uuid}" / "COMPUTE_REPORT.csv"

        if conv_path.exists() and gemm_path.exists():
            conv_df = pd.read_csv(conv_path)
            gemm_df = pd.read_csv(gemm_path)

            total_cycles = conv_df[cycle_col_name].sum() + gemm_df[cycle_col_name].sum()

            # Parse config file
            config = configparser.ConfigParser()
            config.read(config_path)

            ah = int(config["architecture_presets"]["ArrayHeight"])
            aw = int(config["architecture_presets"]["ArrayWidth"])
            ifs = int(config["architecture_presets"]["IfmapSramSzkB"])
            fs = int(config["architecture_presets"]["FilterSramSzkB"])
            ofs = int(config["architecture_presets"]["OfmapSramSzkB"])
            dfw = config["architecture_presets"]["Dataflow"]

            data.append([ah, aw, ifs, fs, ofs, dfw, total_cycles])

    # Create DataFrame
    df = pd.DataFrame(
        data, columns=["ah", "aw", "ifs", "fs", "ofs", "dfw", "total_cycles"]
    )
    df.to_csv(dataset, index=False)

# Encode categorical variable
label_encoder = LabelEncoder()
df["dfw"] = label_encoder.fit_transform(df["dfw"])

# Split data
X = df.drop(columns=["total_cycles"])
y = df["total_cycles"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Define XGBoost model
model = xgb.XGBRegressor(random_state=42)

# Define hyperparameter grid
param_grid = {
    # "n_estimators": [50, 100, 200, 300, 500],
    "n_estimators": [250],
    # "learning_rate": [0.01, 0.1, 0.2, 0.5, 0.8, 1.0],
    "learning_rate": [0.2],
    # "max_depth": [3, 5, 7, 10, 15, 30],
    "max_depth": [13],
    # "subsample": [0.8, 1.0],
    # "colsample_bytree": [0.8, 1.0],
}

# Perform GridSearchCV
grid_search = GridSearchCV(
    model, param_grid, scoring="neg_mean_squared_error", cv=5, n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")

# Save the best model
joblib.dump(best_model, "trained_xgboost_model.pkl")
print("Best XGBoost model saved as 'trained_xgboost_model.pkl'.")

# Save the label encoder for later use
joblib.dump(label_encoder, "label_encoder.pkl")
print("Label encoder saved as 'label_encoder.pkl'.")
