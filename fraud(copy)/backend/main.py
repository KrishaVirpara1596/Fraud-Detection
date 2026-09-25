import json, os, joblib, pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

D = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(f"{D}/model.joblib")
info = json.load(open(f"{D}/metrics.json"))
app = FastAPI(title="Vehicle Insurance Fraud API")
# Set ALLOWED_ORIGINS on Render to your Vercel URL to lock this down (comma separated).
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root(): return {"status": "ok"}

@app.get("/info")
def get_info(): return info

@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame([{c: (payload.get(c) if payload.get(c) not in (None, "") else info["defaults"][c]) for c in info["features"]}])
    for c in info["numeric"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    prob = float(model.predict_proba(df)[0, 1]); t = info["threshold"]
    return {"probability": round(prob * 100, 1), "flagged": prob >= t,
            "risk": "High" if prob >= t else ("Medium" if prob >= t * 0.6 else "Low")}
