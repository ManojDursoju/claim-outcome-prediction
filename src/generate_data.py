
import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_STATE = 42
TOTAL_RECORDS = 400

rng = np.random.default_rng(RANDOM_STATE)

payers = ["Aetna", "Cigna", "UnitedHC", "BlueCross", "Medicare"]
providers = ["Dr.Rao", "Dr.Shah", "Dr.Lee", "Dr.Kumar", "Dr.Patel"]

procedures = ["99213", "99214", "99215", "93000", "80053"]
diagnoses = ["M54.5", "J06.9", "E11.9", "I10", "R51.9"]

# Generate controlled class distribution
status_counts = {
    "Paid": 280,
    "Partially_Paid": 80,
    "Denied": 40
}

records = []

for status, count in status_counts.items():

    for _ in range(count):

        payer = rng.choice(payers)
        provider = rng.choice(providers)
        cpt_code = rng.choice(procedures)
        diagnosis_code = rng.choice(diagnoses)

        patient_age = int(rng.integers(18, 85))

        # Create outcome-related billing patterns
        if status == "Paid":
            billed_amount = rng.uniform(100, 300)
        elif status == "Partially_Paid":
            billed_amount = rng.uniform(250, 600)
        else:
            billed_amount = rng.uniform(500, 1000)

        records.append({
            "payer": payer,
            "provider": provider,
            "cpt_code": cpt_code,
            "diagnosis_code": diagnosis_code,
            "billed_amount": round(billed_amount, 2),
            "patient_age": patient_age,
            "status": status
        })

# Create DataFrame and shuffle
df = pd.DataFrame(records)
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# Add unique claim IDs
df.insert(0, "id", range(1, len(df) + 1))

# Save dataset
output_path = Path(__file__).resolve().parents[1] / "data" / "claims.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)

# Data quality checks
print("\nCLAIMS DATASET GENERATED")
print("-" * 40)
print("Shape:", df.shape)
print("\nClass Distribution:")
print(df["status"].value_counts())
print("\nClass Percentages:")
print(df["status"].value_counts(normalize=True).mul(100).round(2))
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())
print("\nDataset saved to:", output_path)
print("\nSample Records:")
print(df.head())
