
# Claim Outcome Prediction

An end-to-end Machine Learning application that predicts insurance claim outcomes — Paid, Partially Paid, or Denied — using Python, Scikit-learn, and FastAPI.

## Project Overview

This project demonstrates a complete ML workflow, including synthetic data generation, preprocessing, feature engineering, model training, evaluation, model persistence, and REST API deployment.

## Problem Statement

Insurance claims can receive three possible outcomes:

- Paid
- Partially_Paid
- Denied

The objective is to predict the likely outcome using claim attributes such as payer, provider, CPT code, diagnosis code, billed amount, and patient age.

## Technology Stack

- Python 3.11
- Pandas and NumPy
- Scikit-learn
- Random Forest Classifier
- FastAPI and Pydantic
- Joblib
- Pytest

## Project Structure

```text
claim-outcome-prediction/
├── app/
│   └── main.py
├── data/
│   └── claims.csv
├── models/
│   └── claim_pipeline.joblib
├── reports/
│   └── evaluation_results.txt
├── src/
│   ├── generate_data.py
│   ├── features.py
│   └── train.py
├── tests/
│   └── test_api.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset

A reproducible synthetic dataset of 400 insurance claims was generated.

| Outcome | Records | Percentage |
|---|---:|---:|
| Paid | 280 | 70% |
| Partially_Paid | 80 | 20% |
| Denied | 40 | 10% |
| Total | 400 | 100% |

The dataset contains eight columns, including the claim ID and prediction target.

## Feature Engineering

Two additional features were developed:

1. payer_denial_rate
2. provider_denial_rate

Training features use leave-one-out target encoding. Test and inference features use reference rates calculated exclusively from training data to prevent test-set leakage.

## Model Training

- Algorithm: Random Forest Classifier
- Number of estimators: 200
- Class balancing: Balanced class weights
- Train/test split: 80:20
- Stratification: Enabled
- Random state: 42

The preprocessing steps and trained classifier are saved together as a reusable pipeline.

## Model Evaluation

Test set: 80 records.

Accuracy: 87.50%

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Paid | 0.98 | 0.93 | 0.95 |
| Partially_Paid | 0.62 | 0.94 | 0.75 |
| Denied | 1.00 | 0.38 | 0.55 |

Macro F1-score: 0.75

### Confusion Matrix

Rows represent actual outcomes and columns represent predicted outcomes.

| Actual / Predicted | Paid | Partial | Denied |
|---|---:|---:|---:|
| Paid | 52 | 4 | 0 |
| Partially_Paid | 1 | 15 | 0 |
| Denied | 0 | 5 | 3 |

### Findings and Improvement

The model is weakest at identifying Denied claims, with a recall of 0.38. It correctly identifies 3 of the 8 denied claims in the test set and misses 5.

An improvement would be to generate more representative minority-class examples, investigate additional claim-related features, and compare class-weighting and resampling strategies using stratified cross-validation.

## API Usage

Start the API:

```bash
uvicorn app.main:app --reload
```

API documentation:

http://127.0.0.1:8000/docs

### Prediction Endpoint

POST `/predict`

Example request:

```json
{
  "payer": "Aetna",
  "provider": "Dr.Rao",
  "cpt_code": "99213",
  "diagnosis_code": "M54.5",
  "billed_amount": 220.00,
  "patient_age": 45
}
```

Example response structure:

```json
{
  "predicted_status": "Paid",
  "probabilities": {
    "Paid": 0.8833,
    "Partially_Paid": 0.0997,
    "Denied": 0.0169
  }
}
```

Probabilities shown above are an illustrative API response.

### API Validation

The API validates required fields and rejects invalid input with HTTP 400 responses and descriptive error messages.

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/ManojDursoju/claim-outcome-prediction.git

cd claim-outcome-prediction
```

### 2. Create virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate synthetic data

```bash
python src/generate_data.py
```

### 5. Train the model

```bash
python -m src.train
```

### 6. Start the API

```bash
uvicorn app.main:app --reload
```

### 7. Run automated tests

```bash
python -m pytest tests/ -v
```

## Testing Results

All four API tests passed:

- Health endpoint
- Valid prediction
- Invalid billed amount
- Missing required field

## Assumptions and Limitations

- The dataset is synthetically generated and does not represent real insurance claims.
- The outcome distribution is intentionally imbalanced.
- Billed amount ranges are generated differently for each target class, creating an artificial predictive relationship.
- Evaluation uses a single stratified train/test split.
- Predicted probabilities have not been independently calibrated.
- The model is a technical demonstration, not a production insurance decision system.

## Future Improvements

- Evaluate additional classification algorithms.
- Improve recall for the Denied class.
- Perform cross-validation and probability calibration.
- Integrate real, anonymized claims data.
- Add model monitoring and deployment.

## Author

**Manoj Dursoju**

GitHub: https://github.com/ManojDursoju
