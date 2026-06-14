from fastapi import FastAPI, Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest
import joblib
import logging
import sqlite3
import json
import time
from datetime import datetime

app = FastAPI(title="Iris Classifier API")
model = joblib.load("models/best_model.pkl")

logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s"
)


class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, le=10)
    sepal_width: float = Field(..., gt=0, le=10)
    petal_length: float = Field(..., gt=0, le=10)
    petal_width: float = Field(..., gt=0, le=10)


class PredictionOut(BaseModel):
    prediction: int
    label: str


LABELS = {0: "setosa", 1: "versicolor", 2: "virginica"}

REQUEST_COUNT = Counter(
    "prediction_requests_total", "Total prediction requests",
    ["label"])
REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds", "Prediction latency")


def log_to_db(inp, pred):
    con = sqlite3.connect("logs/predictions.db")
    con.execute("""CREATE TABLE IF NOT EXISTS predictions
                   (id INTEGER PRIMARY KEY, ts TEXT, input TEXT, prediction INTEGER)""")
    con.execute(
        "INSERT INTO predictions (ts,input,prediction) VALUES (?,?,?)",
        (datetime.utcnow().isoformat(), json.dumps(inp), pred),
    )
    con.commit()
    con.close()


@app.post("/predict", response_model=PredictionOut)
def predict(data: IrisInput):
    features = [[data.sepal_length, data.sepal_width,
                 data.petal_length, data.petal_width]]
    start = time.time()
    pred = int(model.predict(features)[0])
    REQUEST_LATENCY.observe(time.time() - start)
    label = LABELS[pred]
    REQUEST_COUNT.labels(label=label).inc()
    logging.info(f"input={features[0]} prediction={pred}")
    log_to_db(data.dict(), pred)
    return {"prediction": pred, "label": label}


@app.get("/health")
def health(): return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")


@app.get("/logs")
def get_logs(limit: int = 10):
    con = sqlite3.connect("logs/predictions.db")
    rows = con.execute(
        "SELECT ts, input, prediction FROM predictions "
        "ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    con.close()
    return [{"ts": r[0], "input": json.loads(r[1]),
             "prediction": r[2]} for r in rows]
