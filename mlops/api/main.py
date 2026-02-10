from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import numpy as np
import pickle

import mysql.connector

db = mysql.connector.connect(
    host="mlops-mysql",
    user="root",
    password="root", 
    database="mlops"
)

cursor = db.cursor()


with open ("../model/scaler.pkl","rb") as f:
    scaler = pickle.load(f)

import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")



MODEL_URI = "models:/ChurnPredictionModel@production"
model = mlflow.pyfunc.load_model(MODEL_URI)

app = FastAPI(title="Churn Prediction API")

class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: float
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def health():
    return {"status": "API is running"}

@app.post("/predict")
def predict(data: CustomerData):
    try:
       
        X = np.array([[
            data.gender,
            data.SeniorCitizen,
            data.Partner,
            data.Dependents,
            data.tenure,
            data.PhoneService,
            data.MultipleLines,
            data.InternetService,
            data.OnlineSecurity,
            data.OnlineBackup,
            data.DeviceProtection,
            data.TechSupport,
            data.StreamingTV,
            data.StreamingMovies,
            data.Contract,
            data.PaperlessBilling,
            data.PaymentMethod,
            data.MonthlyCharges,
            data.TotalCharges
        ]])



        
        print("Scaler expects:", scaler.n_features_in_)

        
        X_scaled = scaler.transform(X)

        
        prediction = model.predict(X_scaled)
        print("Prediction done")

        
        probability = None
        try:
            proba = model.predict_proba(X_scaled)
            probability = float(proba[0][1])
        except Exception as e:
            print("Probability not available:", e)

        cursor.execute("""
        INSERT INTO prediction_logs (gender, tenure, monthly_charges, total_charges, prediction, probability)
        VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            data.gender,
            data.tenure,
            data.MonthlyCharges,
            data.TotalCharges,
            int(prediction[0]),
            probability
        ))
        db.commit()


        return {
            "churn_prediction": int(prediction[0]),
            "churn_probability": probability
        }

    except Exception as e:
        print("FULL ERROR:", str(e))
        return {"error": str(e)}
