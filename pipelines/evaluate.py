from pathlib import Path
import json
import os
import joblib
import pandas as pd
import tarfile
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# SageMaker paths, with local fallbacks
MODEL_DIR = Path(os.getenv("SM_MODEL_DIR", "models"))
TEST_DIR = Path(os.getenv("SM_CHANNEL_TEST", "data/processed"))
OUTPUT_DIR = Path(os.getenv("SM_OUTPUT_DATA_DIR", "evaluation"))


def main():
    test_df = pd.read_csv(TEST_DIR / "test.csv")

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    model_path = MODEL_DIR / "model.joblib"

    if not model_path.exists():
        archive_path = MODEL_DIR / "model.tar.gz"

        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(MODEL_DIR)

    model = joblib.load(model_path)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "evaluation.json"

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nEvaluation saved to: {output_path}")


if __name__ == "__main__":
    main()