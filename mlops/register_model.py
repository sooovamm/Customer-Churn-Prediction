import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")

client = MlflowClient()

# Check what's already registered
print("=== Registered Models ===")
for mv in client.search_model_versions("name='ChurnPredictionModel'"):
    print(f"  Version: {mv.version} | Stage: {mv.current_stage} | Run ID: {mv.run_id}")

# Set the production alias on version 1
client.set_registered_model_alias("ChurnPredictionModel", "production", version=1)
print("\n✅ Alias 'production' set on ChurnPredictionModel v1")