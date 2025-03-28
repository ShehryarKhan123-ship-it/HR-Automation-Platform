import os
import joblib
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Define paths
BASE_DIR = os.path.expanduser("~")
MODEL_PATH = os.path.join(BASE_DIR, "Desktop", "random_forest_model.pkl")

def predict_new_candidates(test_csv):
    try:
        logging.debug("Loading model")

        # Load trained model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No model found at '{MODEL_PATH}'")

        model = joblib.load(MODEL_PATH)

        # Load test dataset
        if not os.path.exists(test_csv):
            raise FileNotFoundError(f"No such file or directory: '{test_csv}'")

        df = pd.read_csv(test_csv)

        # Ensure test data has the same features as the training data
        if "label" in df.columns:
            df = df.drop(columns=["label"])  # Drop label if present (not needed for prediction)

        logging.debug("Making predictions")
        predictions = model.predict(df)

        # Add predictions to dataframe
        df["Prediction"] = ["Selected" if pred == 1 else "Not Selected" for pred in predictions]

        # Save results
        output_csv = os.path.join(BASE_DIR, "Desktop", "predictions.csv")
        df.to_csv(output_csv, index=False)
        logging.debug(f"✅ Predictions saved to {output_csv}")

    except Exception as e:
        logging.error(f"❌ Prediction failed: {str(e)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <test_csv>")
    else:
        predict_new_candidates(sys.argv[1])
