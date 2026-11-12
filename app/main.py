
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator


# -----------------------------------------
# Load saved model
# -----------------------------------------

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "claim_pipeline.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. Run python -m src.train first."
    )

artifact = joblib.load(MODEL_PATH)

pipeline = artifact["pipeline"]
payer_rates = artifact["payer_rates"]
provider_rates = artifact["provider_rates"]
global_denial_rate = artifact["global_denial_rate"]


# -----------------------------------------
# Initialize FastAPI
# -----------------------------------------

app = FastAPI(
    title="Claim Outcome Prediction API",
    description="Predict insurance claim outcomes using machine learning.",
    version="1.0.0"
)


# -----------------------------------------
# Request schema and validation
# -----------------------------------------

class ClaimRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    payer: str = Field(min_length=1, max_length=100)
    provider: str = Field(min_length=1, max_length=100)
    cpt_code: str = Field(min_length=1, max_length=20)
    diagnosis_code: str = Field(min_length=1, max_length=20)

    billed_amount: float = Field(gt=0, le=1_000_000)
    patient_age: int = Field(ge=0, le=120)

    @field_validator(
        "payer",
        "provider",
        "cpt_code",
        "diagnosis_code"
    )
    @classmethod
    def validate_non_empty(cls, value):
        value = value.strip()

        if not value:
            raise ValueError("Field cannot contain only whitespace")

        return value


# Return HTTP 400 for invalid request data
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid claim request",
            "details": [
                {
                    "field": ".".join(str(x) for x in error["loc"]),
                    "message": error["msg"]
                }
                for error in exc.errors()
            ]
        }
    )


# -----------------------------------------
# Health endpoint
# -----------------------------------------

@app.get("/")
def home():
    return {
        "message": "Claim Outcome Prediction API is running",
        "docs": "/docs",
        "prediction_endpoint": "POST /predict"
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


# -----------------------------------------
# Prediction endpoint
# -----------------------------------------

@app.post("/predict")
def predict_claim(claim: ClaimRequest):

    claim_data = pd.DataFrame([claim.model_dump()])

    # Reuse training-derived feature references
    claim_data["payer_denial_rate"] = (
        claim_data["payer"]
        .map(payer_rates)
        .fillna(global_denial_rate)
    )

    claim_data["provider_denial_rate"] = (
        claim_data["provider"]
        .map(provider_rates)
        .fillna(global_denial_rate)
    )

    # Generate prediction
    predicted_status = pipeline.predict(claim_data)[0]

    # Calculate class probabilities
    raw_probabilities = pipeline.predict_proba(claim_data)[0]

    probabilities = {
        str(label): round(float(probability), 4)
        for label, probability in zip(
            pipeline.classes_,
            raw_probabilities
        )
    }

    return {
        "predicted_status": str(predicted_status),
        "probabilities": probabilities
    }
