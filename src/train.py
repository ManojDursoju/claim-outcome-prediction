
import joblib
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features import (
    add_training_features,
    add_reference_features,
    calculate_reference_rates
)


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "claims.csv"
MODEL_PATH = ROOT / "models" / "claim_pipeline.joblib"
REPORT_PATH = ROOT / "reports" / "evaluation_results.txt"

RANDOM_STATE = 42

CATEGORICAL_FEATURES = [
    "payer",
    "provider",
    "cpt_code",
    "diagnosis_code"
]

NUMERIC_FEATURES = [
    "billed_amount",
    "patient_age",
    "payer_denial_rate",
    "provider_denial_rate"
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "status"

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)

X = df.drop(columns=["id", TARGET])
y = df[TARGET]

# --------------------------------------------------
# 3. Stratified train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True).round(3))

# --------------------------------------------------
# 4. Feature engineering
# --------------------------------------------------

train_data = X_train.copy()
train_data[TARGET] = y_train

# Training features exclude each row's own target
X_train_engineered = add_training_features(train_data)

# Calculate reference rates using training data only
payer_rates = calculate_reference_rates(
    train_data, "payer"
)

provider_rates = calculate_reference_rates(
    train_data, "provider"
)

global_denial_rate = (y_train == "Denied").mean()

# Test features use training references only
X_test_engineered = add_reference_features(
    X_test,
    payer_rates,
    provider_rates,
    global_denial_rate
)

X_train_final = X_train_engineered[FEATURES]
X_test_final = X_test_engineered[FEATURES]

# --------------------------------------------------
# 5. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES
        ),
        (
            "numeric",
            "passthrough",
            NUMERIC_FEATURES
        )
    ]
)

# --------------------------------------------------
# 6. Build and train model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=4,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

print("\nTraining model...")

pipeline.fit(X_train_final, y_train)

print("Model training completed.")

# --------------------------------------------------
# 7. Model evaluation
# --------------------------------------------------

y_pred = pipeline.predict(X_test_final)

accuracy = accuracy_score(y_test, y_pred)

class_report = classification_report(
    y_test,
    y_pred,
    labels=["Paid", "Partially_Paid", "Denied"],
    zero_division=0
)

matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=["Paid", "Partially_Paid", "Denied"]
)

print("\nMODEL EVALUATION")
print("=" * 50)

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(class_report)

print("\nConfusion Matrix:")
print(matrix)

# --------------------------------------------------
# 8. Save trained pipeline and metadata
# --------------------------------------------------

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

artifact = {
    "pipeline": pipeline,
    "payer_rates": payer_rates,
    "provider_rates": provider_rates,
    "global_denial_rate": global_denial_rate,
    "features": FEATURES,
    "classes": pipeline.classes_.tolist()
}

joblib.dump(artifact, MODEL_PATH)

# --------------------------------------------------
# 9. Save evaluation report
# --------------------------------------------------

with open(REPORT_PATH, "w", encoding="utf-8") as file:

    file.write("CLAIM OUTCOME PREDICTION - MODEL EVALUATION\n")
    file.write("=" * 50 + "\n")

    file.write(f"Total records: {len(df)}\n")
    file.write(f"Training records: {len(X_train)}\n")
    file.write(f"Testing records: {len(X_test)}\n")

    file.write(f"\nAccuracy: {accuracy:.4f}\n")

    file.write("\nClassification Report:\n")
    file.write(class_report)

    file.write("\nConfusion Matrix:\n")
    file.write(str(matrix))

print("\nSaved model:", MODEL_PATH)
print("Saved evaluation:", REPORT_PATH)

print("\nTRAINING PIPELINE COMPLETED SUCCESSFULLY")
