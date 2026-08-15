import os
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

# SageMaker provides these directories as environment variables
TRAIN_DIR = Path(os.environ.get("SM_CHANNEL_TRAIN", "data/processed"))
MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "models"))


def main():

    # Load training data
    train_df = pd.read_csv(TRAIN_DIR / "train.csv")

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    # Define model pipeline
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=4,
        random_state=42,
    )

    # Train
    model.fit(X_train, y_train)

    # Save model artifact
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(model, model_path)

    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()