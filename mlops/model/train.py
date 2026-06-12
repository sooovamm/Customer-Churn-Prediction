import pandas as pd
import numpy as np
import pickle

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("churn prediction")


df = pd.read_csv("../data/raw/telco_churn.csv")

print("Dataset loaded successfully")
print("Shape:", df.shape)



if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])


df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")


df = df.fillna(df.median(numeric_only=True))


df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})



categorical_cols = df.select_dtypes(include=["object"]).columns

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])


X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

with mlflow.start_run():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("model trained")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print ("accuracy:", round(acc, 4))
    print(classification_report(y_test, y_pred))

    mlflow.log_param("model","Logistic regression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("accuracy", acc)
    

    result = mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    registered_model_name="ChurnPredictionModel"
    )


   

    print("logged to mlflow")



with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel and scaler saved successfully")
