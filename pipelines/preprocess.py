import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from pathlib import Path
import os

INPUT_DIR = Path(
    os.getenv("SM_PROCESSING_INPUT_DIR", "data")
)
OUTPUT_DIR = Path(
    os.getenv("SM_PROCESSING_OUTPUT_DIR", "data/processed")
)
INPUT_FILE = INPUT_DIR / "processed.cleveland.data"

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

df = pd.read_csv(
    INPUT_FILE,
    names=COLUMNS,
    na_values="?"
)

# Convert the original UCI target into binary classification
df["target"] = (df["target"] > 0).astype(int)

print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["target"].value_counts().sort_index())

# Separate features and target
X = df.drop(columns=["target"])
y = df["target"]

# First split: 70% train, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

# Second split: divide the remaining 30% equally
X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)

print("\nDataset splits:")
print("Train:", X_train.shape)
print("Validation:", X_validation.shape)
print("Test:", X_test.shape)

# Fit the imputer using training data only
imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index,
)

X_validation = pd.DataFrame(
    imputer.transform(X_validation),
    columns=X_validation.columns,
    index=X_validation.index,
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index,
)

print("\nMissing values after imputation:")
print("Train:", X_train.isnull().sum().sum())
print("Validation:", X_validation.isnull().sum().sum())
print("Test:", X_test.isnull().sum().sum())

train_df = X_train.copy()
train_df["target"] = y_train

validation_df = X_validation.copy()
validation_df["target"] = y_validation

test_df = X_test.copy()
test_df["target"] = y_test

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
validation_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

print("\nProcessed datasets saved:")
print(OUTPUT_DIR / "train.csv")
print(OUTPUT_DIR / "validation.csv")
print(OUTPUT_DIR / "test.csv")