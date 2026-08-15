from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, computed_field
from typing import Literal, Annotated
import pickle
import pandas as pd
import os

# ── Load ML model ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Insurance Premium Predictor",
    description="Predict insurance premium category using ML",
    version="1.0.0",
)

# ── CORS (allow all origins for deployment flexibility) ────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── City tier lists ────────────────────────────────────────────────────────────
tier_1_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
tier_2_cities = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam",
    "Coimbatore", "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur",
    "Raipur", "Amritsar", "Varanasi", "Agra", "Dehradun", "Mysore", "Jabalpur",
    "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik", "Allahabad", "Udaipur",
    "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
    "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode",
    "Warangal", "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri",
]


# ── Pydantic input model ───────────────────────────────────────────────────────
class UserInput(BaseModel):
    age: Annotated[int, Field(..., gt=0, lt=120, description="Age of the user")]
    weight: Annotated[float, Field(..., gt=0, description="Weight of the user in kg")]
    height: Annotated[float, Field(..., gt=0, lt=2.5, description="Height of the user in metres")]
    income_lpa: Annotated[float, Field(..., gt=0, description="Annual income in LPA")]
    smoker: Annotated[bool, Field(..., description="Is the user a smoker?")]
    city: Annotated[str, Field(..., description="City the user belongs to")]
    occupation: Annotated[
        Literal["retired", "freelancer", "student", "government_job",
                "business_owner", "unemployed", "private_job"],
        Field(..., description="Occupation of the user"),
    ]

    @computed_field
    @property
    def bmi(self) -> float:
        return round(self.weight / (self.height ** 2), 4)

    @computed_field
    @property
    def lifestyle_risk(self) -> str:
        if self.smoker and self.bmi > 30:
            return "high"
        elif self.smoker or self.bmi > 27:
            return "medium"
        return "low"

    @computed_field
    @property
    def age_group(self) -> str:
        if self.age < 25:
            return "young"
        elif self.age < 45:
            return "adult"
        elif self.age < 60:
            return "middle_aged"
        return "senior"

    @computed_field
    @property
    def city_tier(self) -> int:
        if self.city in tier_1_cities:
            return 1
        elif self.city in tier_2_cities:
            return 2
        return 3


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "service": "Insurance Premium Predictor", "version": "1.0.0"}


# ── City tier lookup (used by frontend for live badge) ─────────────────────────
@app.get("/city-tier/{city}", tags=["Utilities"])
def get_city_tier(city: str):
    if city in tier_1_cities:
        tier = 1
    elif city in tier_2_cities:
        tier = 2
    else:
        tier = 3
    return {"city": city, "tier": tier}


# ── Prediction endpoint ────────────────────────────────────────────────────────
@app.post("/predict", tags=["ML"])
def predict_premium(data: UserInput):
    input_df = pd.DataFrame([{
        "bmi": data.bmi,
        "age_group": data.age_group,
        "lifestyle_risk": data.lifestyle_risk,
        "city_tier": data.city_tier,
        "income_lpa": data.income_lpa,
        "occupation": data.occupation,
    }])

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    classes = model[-1].classes_  # access final estimator, not Pipeline wrapper

    confidence = float(max(proba))
    class_probabilities = {
        str(cls): round(float(p), 4) for cls, p in zip(classes, proba)
    }

    return JSONResponse(
        status_code=200,
        content={
            "response": {
                "predicted_category": str(prediction),
                "confidence": round(confidence, 4),
                "class_probabilities": class_probabilities,
                "computed": {
                    "bmi": data.bmi,
                    "age_group": data.age_group,
                    "lifestyle_risk": data.lifestyle_risk,
                    "city_tier": data.city_tier,
                },
            }
        },
    )


# ── Serve static frontend (mount LAST so API routes take priority) ─────────────
static_dir = os.path.join(BASE_DIR, "static")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
