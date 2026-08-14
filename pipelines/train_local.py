from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
                 
# Load processed datasets
train_df = pd.read_csv(DATA_DIR / "train.csv")
validation_df = pd.read_csv(DATA_DIR / "validation.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

# Separate features and target
X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]

X_validation = validation_df.drop(columns=["target"])
y_validation = validation_df["target"]

X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

def evaluate_model(name, model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print(f"\n{name}")
    print(f"Accuracy : {accuracy_score(y, y_pred):.4f}")
    print(f"Precision: {precision_score(y, y_pred):.4f}")
    print(f"Recall   : {recall_score(y, y_pred):.4f}")
    print(f"F1       : {f1_score(y, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y, y_prob):.4f}")

def main():

    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    # Train
    logistic_model.fit(X_train, y_train)

    # Validation
    evaluate_model(
        "Validation",
        logistic_model,
        X_validation,
        y_validation,
    )

    # Final test
    evaluate_model(
        "Test",
        logistic_model,
        X_test,
        y_test,
    )

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "heart_disease_model.joblib"
    joblib.dump(logistic_model, model_path)

    print(f"\nModel saved: {model_path}")

if __name__ == "__main__":
    main()