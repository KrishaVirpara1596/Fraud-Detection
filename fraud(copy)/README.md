# ClaimGuard – Vehicle Insurance Fraud Detection

- `backend/`  FastAPI + trained scikit-learn model (`model.joblib`). Retrain any time: `python train.py`
- `frontend/` React (Vite) app: Home, Check a claim, Model, Data insights, Disclaimer
- `render.yaml` one-click backend deploy config for Render

## Run locally
    cd backend && pip install -r requirements.txt && uvicorn main:app --reload
    cd frontend && cp .env.example .env && npm install && npm run dev      (opens http://localhost:5173)

## Deploy (all free)
1. Create a GitHub repo and push this whole folder to it.
2. **Backend on Render:** render.com > New > Blueprint > pick your repo > Apply. Wait for "Live" and copy the URL (https://fraud-api-xxxx.onrender.com).
   (Manual alternative: New > Web Service, Root Directory `backend`, Build `pip install -r requirements.txt`, Start `uvicorn main:app --host 0.0.0.0 --port $PORT`.)
3. **Frontend on Vercel:** vercel.com > Add New > Project > import the repo > set Root Directory to `frontend` > add environment variable `VITE_API_URL` = your Render URL (no trailing slash) > Deploy.
4. Open your `*.vercel.app` link. The first request after idle can take ~1 minute while Render wakes up.
5. Optional: on Render add env var `ALLOWED_ORIGINS` = your Vercel URL to restrict CORS.
