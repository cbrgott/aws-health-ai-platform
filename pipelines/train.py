import os
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# SageMaker provides these directories as environment variables
TRAIN_DIR = Path(os.environ.get("SM_CHANNEL_TRAIN", "data/processed"))
MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "models"))


def main():

    # Load training data
    train_df = pd.read_csv(TRAIN_DIR / "train.csv")

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    # Define model pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    # Train
    model.fit(X_train, y_train)

    # Save model artifact
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(model, model_path)

    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()