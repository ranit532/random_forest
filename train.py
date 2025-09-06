import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import joblib
import os

# Define paths
DATA_PATH = "data/generated_data.csv"
MODEL_DIR = "models"
PLOT_DIR = "plots"

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Set MLflow experiment
mlflow.set_experiment("Random_Forest_PoC")

# 1. Load Data
data = pd.read_csv(DATA_PATH)
X = data.drop("target", axis=1)
y = data["target"]

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start an MLflow run
with mlflow.start_run():
    # 3. Define Model and Hyperparameters
    n_estimators = 100
    max_depth = 10
    random_state = 42

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "random_state": random_state,
    }

    # Log hyperparameters
    mlflow.log_params(params)

    # 4. Train Model
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    rf_model.fit(X_train, y_train)

    # 5. Evaluate Model
    y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    # Log metrics
    mlflow.log_metrics(metrics)
    print(f"Metrics: {metrics}")

    # 6. Create and Save Plots
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    cm_path = os.path.join(PLOT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # Feature Importance
    importances = rf_model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    feature_importance_df = feature_importance_df.sort_values(by="importance", ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x="importance", y="feature", data=feature_importance_df)
    plt.title("Feature Importance")
    feature_importance_path = os.path.join(PLOT_DIR, "feature_importance.png")
    plt.savefig(feature_importance_path)
    plt.close()

    # Log plots as artifacts
    mlflow.log_artifact(cm_path, "plots")
    mlflow.log_artifact(feature_importance_path, "plots")

    # 7. Log Model
    mlflow.sklearn.log_model(rf_model, "random-forest-model")

    # 8. Save the model locally for the FastAPI app
    model_path = os.path.join(MODEL_DIR, "random_forest_model.joblib")
    joblib.dump(rf_model, model_path)
    print(f"Model saved locally at {model_path}")

print("Training script finished successfully.")