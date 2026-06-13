import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import os, joblib

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("iris-classification")

df = pd.read_csv("data/raw/iris.csv")
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

best_acc, best_name, best_model = 0, None, None

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")
        print(f"{name}: acc={acc:.3f} f1={f1:.3f}")

        if acc > best_acc:
            best_acc, best_name, best_model = acc, name, model

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")
print(f"\nBest model: {best_name} ({best_acc:.3f})")

# Register in MLflow Model Registry
with mlflow.start_run(run_name="best_model_registration"):
    mlflow.sklearn.log_model(
        best_model, "model",
        registered_model_name="iris-classifier"
    )