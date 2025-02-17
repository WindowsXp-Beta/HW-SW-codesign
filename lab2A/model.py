import argparse
import configparser
import joblib
import numpy as np
from pathlib import Path


def load_config(config_path, encoder):
    """Load system configuration from a given file."""
    config = configparser.ConfigParser()
    config.read(config_path)

    ah = int(config["architecture_presets"]["ArrayHeight"])
    aw = int(config["architecture_presets"]["ArrayWidth"])
    ifs = int(config["architecture_presets"]["IfmapSramSzkB"])
    fs = int(config["architecture_presets"]["FilterSramSzkB"])
    ofs = int(config["architecture_presets"]["OfmapSramSzkB"])
    dfw = config["architecture_presets"]["Dataflow"]

    dfw = encoder.transform([dfw])[0]

    return np.array([[ah, aw, ifs, fs, ofs, dfw]])


def main():
    parser = argparse.ArgumentParser(description="Predict using trained XGBoost model.")
    parser.add_argument("-c", "--config", required=True, help="Path to the system.cfg file")
    parser.add_argument("-p", "--path", required=True)
    args = parser.parse_args()

    model_path = Path(args.path) / "trained_xgboost_model.pkl"
    encoder_path = Path(args.path) / "label_encoder.pkl"

    # Load model and encoder
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)

    # Load config and make prediction
    input_data = load_config(args.config, label_encoder)
    prediction = model.predict(input_data)[0]
    print(f"Predicted Total Cycles: {prediction}")


if __name__ == "__main__":
    main()
