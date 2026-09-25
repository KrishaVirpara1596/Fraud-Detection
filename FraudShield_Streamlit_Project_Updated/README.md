# 🛡️ FraudShield AI — Streamlit ML Project

A deployment-ready academic Machine Learning project for **Vehicle Insurance Fraud Detection** using the supplied insurance fraud dataset.

## What is included
- `app.py` — complete Streamlit frontend + prediction backend
- `model.joblib` — trained HistGradientBoosting model + preprocessing pipeline
- `train_model.py` — retraining script
- `metadata.json` — dataset/model metadata
- `data/insurance_fraud_data.csv` — supplied dataset
- `requirements.txt` — deployment dependencies
- `.streamlit/config.toml` — Streamlit theme

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## If you want to retrain
Replace the CSV in `data/` and run:

```bash
python train_model.py
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload the **contents of this folder**.
3. Go to Streamlit Community Cloud.
4. Create a new app.
5. Select your repository and `main` branch.
6. Set the main file to `app.py`.
7. Deploy.

The repository already contains `requirements.txt`, the model file and dataset, so the deployed app can start directly.

## Important academic note
The department SOP describes a Flask interface in the common-project phase. This implementation intentionally uses **Streamlit throughout**, because the project requirement for this build is Streamlit. The SOP also requires at least one algorithm to be implemented without a library. The project therefore includes a scratch NumPy logistic-regression implementation in the development/training evidence; the deployed primary model is a scikit-learn HistGradientBoostingClassifier.

## Dataset / target
Target: `fraud reported` (`Y` = fraud, `N` = not fraud).

Identifier/date fields excluded from the model:
- `claim_number`
- `claim_date`
- `zip_code`

## Viva points
**Frontend:** Streamlit UI with navigation, forms, charts and prediction result cards.

**Backend:** The Streamlit app loads a saved ML pipeline, accepts user input, converts it into a DataFrame, calls `predict_proba`, and displays a risk probability.

**ML pipeline:** missing-value handling → categorical encoding → HistGradientBoostingClassifier.

**Deployment:** GitHub → Streamlit Community Cloud → public `streamlit.app` URL.
