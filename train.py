# train.py
import json
import pickle
import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── Load & prepare data ──────────────────────────────────────────────────────
data = load_wine()
X, y = data.data, data.target.astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Train model ──────────────────────────────────────────────────────────────
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f"MSE : {mse:.4f}")
print(f"R²  : {r2:.4f}")

# ── Save model ───────────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler}, f)

# ── Save metrics ─────────────────────────────────────────────────────────────
metrics = {"mse": round(mse, 4), "r2": round(r2, 4)}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved: model.pkl, metrics.json")