import os
import joblib
from sklearn.linear_model import LogisticRegression

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

_X = [[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3], [5.9, 3.0, 4.2, 1.5]]
_y = [0, 2, 1]
_model = LogisticRegression()
_model.fit(_X, _y)
joblib.dump(_model, "models/best_model.pkl")
