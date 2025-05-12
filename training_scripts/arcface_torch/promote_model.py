import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://129.114.27.48:8000/") 
client = MlflowClient()

experiment = client.get_experiment_by_name("250511_2312_training_10k")

runs = client.search_runs(experiment_ids=[experiment.experiment_id], 
    order_by=["metrics.avg_train_loss"], 
    max_results=2)

best_run = runs[0]  # The first run is the best due to sorting
best_run_id = best_run.info.run_id
best_test_accuracy = best_run.data.metrics["avg_train_loss"]
model_uri = f"runs:/{best_run_id}/model"

print(f"Best Run ID: {best_run_id}")
print(f"Test Accuracy: {best_test_accuracy}")
print(f"Model URI: {model_uri}")

model_name = "staging__2"
registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
print(f"Model registered as '{model_name}', version {registered_model.version}")
