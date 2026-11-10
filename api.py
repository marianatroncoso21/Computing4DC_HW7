import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# What each request must contain: the 10 features of one patient
class Patient(BaseModel):
    age: int
    height: float
    weight: float
    aids: int
    cirrhosis: int
    hepatic_failure: int
    immunosuppression: int
    leukemia: int
    lymphoma: int
    solid_tumor_with_metastasis: int


# Create the API
app = FastAPI()


# Endpoint: receives one patient and returns the probability of diabetes
@app.post("/predict")
async def predict(patient: Patient):
    # Parametro: patient
    # Load the trained model
    model = joblib.load("model.joblib")

    # Turn the patient into a table with one row (same format as the training data)
    data = pd.DataFrame([patient.model_dump()])
    # método: patient.model_dump()

    # Probability that this patient has diabetes (class 1)
    probability = model.predict_proba(data)[0][1]
    # método: model.predict_proba(data)

    return {"prediction": float(probability)}
