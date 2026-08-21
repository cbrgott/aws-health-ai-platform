import json
import pandas as pd


INPUT_FILE = "monitoring/production_drifted.csv"
OUTPUT_FILE = "monitoring/drifted_batch.json"

df = pd.read_csv(INPUT_FILE)

payload = {
    "instances": df.to_dict(orient="records")
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(payload, f)

print(f"Created {OUTPUT_FILE}")
print(f"Observations: {len(df)}")