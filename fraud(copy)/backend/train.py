import json, joblib, datetime as dt, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

raw = pd.read_csv("insurance_fraud_data.csv"); n_raw = len(raw)
df = raw.rename(columns=lambda c: c.replace(" ", "_"))
CAT = ["gender", "property_status", "accident_site", "channel", "vehicle_category"]
NUM = ["age_of_driver", "marital_status", "safety_rating", "annual_income", "high_education", "address_change",
       "past_num_of_claims", "witness_present", "liab_prct", "police_report", "age_of_vehicle", "vehicle_price",
       "total_claim", "injury_claim", "policy_deductible", "annual_premium", "days_open", "form_defects"]
df[CAT] = df[CAT].replace("*", np.nan)
for c in NUM: df[c] = pd.to_numeric(df[c], errors="coerce")       # '*' -> NaN
df = df.dropna(subset=["fraud_reported"]); df = df[df.age_of_driver <= 100]   # missing target, impossible ages
y = (df.fraud_reported == "Y").astype(int); X = df[CAT + NUM]

eda = {c: {str(k): round(v * 100, 1) for k, v in y.groupby(df[c]).mean().items()}
       for c in ["accident_site", "channel", "past_num_of_claims", "witness_present", "police_report"]}

pre = ColumnTransformer([
    ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                    ("o", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), CAT),
    ("n", SimpleImputer(strategy="median"), NUM)])
params = dict(n_estimators=200, learning_rate=0.05, max_depth=3, min_samples_leaf=5, random_state=42)
pipe = Pipeline([("pre", pre), ("gb", GradientBoostingClassifier(**params))])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
cv_auc = cross_val_score(pipe, Xtr, ytr, cv=5, scoring="roc_auc")
oof = cross_val_predict(pipe, Xtr, ytr, cv=5, method="predict_proba")[:, 1]
# threshold maximising balanced accuracy on out-of-fold predictions
grid = np.arange(0.1, 0.7, 0.01); thr = float(grid[np.argmax([balanced_accuracy_score(ytr, oof >= t) for t in grid])])
pipe.fit(Xtr, ytr)
p = pipe.predict_proba(Xte)[:, 1]; pred = p >= thr
imp = sorted(zip(CAT + NUM, pipe.named_steps["gb"].feature_importances_), key=lambda t: -t[1])[:8]
info = dict(algorithm="GradientBoostingClassifier", library="scikit-learn", params=params, threshold=round(thr, 2),
    trained_at=dt.datetime.now().strftime("%d %b %Y"), features=CAT + NUM, numeric=NUM, n_features=len(CAT + NUM),
    accuracy=round(accuracy_score(yte, pred) * 100, 1), precision=round(precision_score(yte, pred) * 100, 1),
    recall=round(recall_score(yte, pred) * 100, 1), f1=round(f1_score(yte, pred) * 100, 1),
    roc_auc=round(roc_auc_score(yte, p) * 100, 1), train_auc=round(roc_auc_score(ytr, pipe.predict_proba(Xtr)[:, 1]) * 100, 1),
    cv_auc_mean=float(round(cv_auc.mean() * 100, 1)), cv_auc_std=float(round(cv_auc.std() * 100, 1)),
    rows_raw=n_raw, rows_removed=n_raw - len(df), rows_final=len(df), fraud_rate=float(round(y.mean() * 100, 1)),
    importance=[[k, float(round(v * 100, 1))] for k, v in imp], eda=eda)
info["defaults"] = {**{c: X[c].mode()[0] for c in CAT}, **{c: float(X[c].median()) for c in NUM}}
joblib.dump(pipe, "model.joblib"); json.dump(info, open("metrics.json", "w"), indent=1)
print({k: v for k, v in info.items() if k not in ("features", "numeric", "eda")})
