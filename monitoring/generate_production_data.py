import pandas as pd


BASELINE_FILE = "data/monitoring/train_features.csv"

NORMAL_OUTPUT = "monitoring/production_normal.csv"
DRIFTED_OUTPUT = "monitoring/production_drifted.csv"


df = pd.read_csv(BASELINE_FILE)


# Normal production sample
normal = df.sample(
    n=100,
    replace=True,
    random_state=42,
)

normal.to_csv(
    NORMAL_OUTPUT,
    index=False,
)


# Drifted production sample
drifted = df.sample(
    n=100,
    replace=True,
    random_state=42,
).copy()

drifted["age"] = drifted["age"] + 25
drifted["chol"] = drifted["chol"] + 150
drifted["thalach"] = drifted["thalach"] - 50

drifted.to_csv(
    DRIFTED_OUTPUT,
    index=False,
)


print("Created:")
print(NORMAL_OUTPUT)
print(DRIFTED_OUTPUT)