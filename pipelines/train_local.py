from pathlib import Path
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

    random_forest = RandomForestClassifier(
        random_state=42
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 3, 5, 8],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    grid_search = GridSearchCV(
        estimator=random_forest,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
    )
    logistic_model.fit(X_train, y_train)
    grid_search.fit(X_train, y_train)

    best_random_forest = grid_search.best_estimator_

    print("\nBest Random Forest parameters:")
    print(grid_search.best_params_)

    print("\n--- VALIDATION COMPARISON ---")

    evaluate_model(
        "Logistic Regression",
        logistic_model,
        X_validation,
        y_validation,
    )

    evaluate_model(
        "Tuned Random Forest",
        best_random_forest,
        X_validation,
        y_validation,
    )

    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    # Final test
    evaluate_model(
        "Test",
        logistic_model,
        X_test,
        y_test,
    )

    evaluate_model(
        "Tuned Random Forest - Test",
        best_random_forest,
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