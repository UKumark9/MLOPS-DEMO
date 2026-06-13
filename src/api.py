from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib, logging, sqlite3, json
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

def log_to_db(inp, pred):
    con = sqlite3.connect("logs/predictions.db")
    con.execute("""CREATE TABLE IF NOT EXISTS predictions
        (id INTEGER PRIMARY KEY, ts TEXT, input TEXT, prediction INTEGER)""")
    con.execute("INSERT INTO predictions (ts,input,prediction) VALUES (?,?,?)",
        (datetime.utcnow().isoformat(), json.dumps(inp), pred))
    con.commit(); con.close()

@app.post("/predict", response_model=PredictionOut)
def predict(data: IrisInput):
    features = [[data.sepal_length, data.sepal_width,
                  data.petal_length, data.petal_width]]
    pred = int(model.predict(features)[0])
    label = LABELS[pred]
    logging.info(f"input={features[0]} prediction={pred}")
    log_to_db(data.dict(), pred)
    return {"prediction": pred, "label": label}

@app.get("/health")
def health(): return {"status": "ok"}