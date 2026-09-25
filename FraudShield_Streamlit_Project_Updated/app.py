
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "insurance_fraud_data.csv"
MODEL_PATH = BASE / "model.joblib"
META_PATH = BASE / "metadata.json"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_meta():
    return json.loads(META_PATH.read_text())

df = load_data()
model = load_model()
meta = load_meta()

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 10% 0%, #eef5ff 0%, #f8fafc 32%, #ffffff 75%);
    color: #1f2937 !important;
}

/* Main page text */
[data-testid="stAppViewContainer"] .stMarkdown {
    color: #1f2937 !important;
}

/* Headings and normal text */
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] p {
    color: #1f2937;
}

/* Form field labels */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] * {
    color: #1f2937 !important;
}

/* Model metric cards */
[data-testid="stMetric"] label {
    color: #475569 !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #111827 !important;
}

[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #475569 !important;
}

/* Information boxes */
[data-testid="stAlert"],
[data-testid="stAlert"] * {
    color: #1f2937 !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0b1220 0%,#111c32 100%);
}
[data-testid="stSidebar"] * { color: #e7eefc !important; }
.hero {
    padding: 2.1rem 2.3rem; border-radius: 24px; margin-bottom: 1.2rem;
    background: linear-gradient(135deg,#0f172a,#1e3a8a 58%,#2563eb);
    color:white; box-shadow: 0 16px 40px rgba(30,58,138,.18);
}
.hero h1 { margin:0; font-size:2.45rem; font-weight:800; letter-spacing:-1px; }
.hero p { margin:.45rem 0 0; color:#dbeafe; font-size:1rem; }
.card {
    background: rgba(255,255,255,.92);
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 1.15rem 1.25rem;
    box-shadow: 0 8px 25px rgba(15,23,42,.06);
    color: #1f2937;
}

.card h3,
.card p,
.card b {
    color: #1f2937;
}

.metric-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 1rem 1.1rem;
    box-shadow: 0 5px 18px rgba(15,23,42,.05);
    color: #1f2937;
}

.metric-card h2 {
    color: #111827 !important;
}

.metric-card .small {
    color: #64748b !important;
}
.badge {
    display:inline-block; padding:.35rem .7rem; border-radius:999px; font-weight:700;
    font-size:.78rem; background:#dbeafe; color:#1d4ed8;
}
.result-safe {background:#ecfdf5;border:1px solid #a7f3d0;border-radius:20px;padding:1.4rem;}
.result-risk {background:#fff1f2;border:1px solid #fecdd3;border-radius:20px;padding:1.4rem;}
.small {color:#64748b;font-size:.86rem;}
/* ===== FINAL TEXT VISIBILITY FIX ===== */

/* Hero headings on every page */
.hero h1,
.hero h2,
.hero h3 {
    color: #ffffff !important;
}

/* Hero normal text */
.hero p {
    color: #ffffff !important;
}

/* Streamlit expander header */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {
    color: #ffffff !important;
}

/* Keep expander content readable */
[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] p {
    color: #1f2937 !important;
}
/* ===== ADVANCED DETAILS - SAME COLOR IN ALL STATES ===== */

[data-testid="stExpander"] details > summary,
[data-testid="stExpander"] details > summary:hover,
[data-testid="stExpander"] details > summary:focus,
[data-testid="stExpander"] details > summary:active {
    background-color: #191c24 !important;
    color: #ffffff !important;
}

/* Text inside the expander header */
[data-testid="stExpander"] details > summary p,
[data-testid="stExpander"] details > summary span,
[data-testid="stExpander"] details > summary div {
    color: #ffffff !important;
}

/* Arrow/icon */
[data-testid="stExpander"] details > summary svg {
    color: #ffffff !important;
    fill: #ffffff !important;
}

/* Remove hover color change */
[data-testid="stExpander"] details > summary:hover p,
[data-testid="stExpander"] details > summary:hover span,
[data-testid="stExpander"] details > summary:hover div {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## 🛡️ FraudShield AI")
st.sidebar.caption("Vehicle Insurance Fraud Detection")
page = st.sidebar.radio("Navigate", ["🏠 Overview", "🔍 Predict Fraud", "📊 EDA", "🧠 Model Details"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown("**Project stack**")
st.sidebar.write("• Python + Streamlit")
st.sidebar.write("• scikit-learn")
st.sidebar.write("• NumPy / Pandas")
st.sidebar.write("• Saved ML pipeline")
st.sidebar.markdown("---")
st.sidebar.caption("Academic ML Project • Streamlit deployment")

if page == "🏠 Overview":
    st.markdown("""<div class="hero">
    <div class="badge">MACHINE LEARNING • DEPLOYMENT READY</div>
    <h1>FraudShield AI</h1>
    <p>Vehicle insurance claim fraud screening with an interactive Streamlit interface.</p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col, title, value, sub in [
        (c1,"Claims analyzed",f"{len(df):,}","dataset records"),
        (c2,"Fraud rate",f"{meta['fraud_rate']*100:.1f}%","positive target rate"),
        (c3,"Input features",str(meta["features"]),"after ID/date exclusions"),
        (c4,"Test ROC-AUC",f"{meta['metrics_at_default_0.5']['roc_auc']:.3f}","held-out test set")]:
        col.markdown(f'<div class="metric-card"><div class="small">{title}</div><h2 style="margin:.25rem 0">{value}</h2><div class="small">{sub}</div></div>',unsafe_allow_html=True)

    st.write("")
    left,right=st.columns([1.35,1])
    with left:
        st.markdown('<div class="card"><h3>🎯 What this project does</h3><p>The app takes vehicle-insurance claim details, applies the same preprocessing used during training, and returns a fraud probability plus a screening decision.</p><p><b>Model:</b> HistGradientBoostingClassifier<br><b>Decision threshold:</b> 25% for fraud-sensitive screening<br><b>Scratch requirement:</b> NumPy logistic regression implementation is included in the project.</p></div>',unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card"><h3>📌 Project flow</h3><p>1. Data cleaning & preprocessing</p><p>2. EDA & feature preparation</p><p>3. Library model training</p><p>4. Scratch algorithm implementation</p><p>5. Evaluation & visualization</p><p>6. Streamlit UI + deployment</p></div>',unsafe_allow_html=True)

elif page == "🔍 Predict Fraud":
    st.markdown("""<div class="hero"><div class="badge">LIVE PREDICTION</div><h1>Check a Claim</h1><p>Quick screening with the most useful claim details. Advanced fields are filled automatically from the training data.</p></div>""", unsafe_allow_html=True)

    # Keep the prediction screen simple while still supplying every feature
    # expected by the trained model. Less commonly entered fields use dataset
    # medians/modes automatically and can be adjusted in Advanced Details.
    def mode_value(col):
        return df[col].dropna().mode().iloc[0]

    with st.form("prediction_form"):
        st.markdown("### 📝 Quick Claim Check")
        st.caption("Only 10 key details are required. The remaining model inputs are automatically filled with typical dataset values.")

        a, b, c, d, e = st.columns(5)
        age = a.number_input("Driver age", min_value=16, max_value=100, value=int(df["age_of_driver"].median()))
        gender = b.selectbox("Gender", sorted(df["gender"].dropna().unique()))
        safety = c.number_input("Safety rating", min_value=int(df["safety_rating"].min()), max_value=int(df["safety_rating"].max()), value=int(df["safety_rating"].median()))
        past = d.number_input("Past claims", min_value=0, max_value=int(df["past_num_of_claims"].max()), value=int(df["past_num_of_claims"].median()))
        total_claim = e.number_input("Total claim", min_value=0.0, value=float(df["total_claim"].median()), step=500.0)

        a, b, c, d, e = st.columns(5)
        site = a.selectbox("Accident site", sorted(df["accident_site"].dropna().unique()))
        witness = b.selectbox("Witness present", sorted(df["witness_present"].dropna().unique()))
        liab = c.number_input("Liability %", min_value=0, max_value=100, value=int(df["liab_prct"].median()))
        police = d.selectbox("Police report", sorted(df["police_report"].dropna().unique()))
        vehicle_age = e.selectbox("Vehicle age", sorted(df["age_of_vehicle"].dropna().unique()))

        with st.expander("⚙️ Advanced Details (optional)"):
            st.caption("You normally don't need to change these. They are included for detailed testing/demo purposes.")
            a, b, c, d = st.columns(4)
            marital = a.selectbox("Marital status", sorted(df["marital_status"].dropna().unique()))
            income = b.number_input("Annual income", min_value=0.0, value=float(df["annual_income"].median()), step=1000.0)
            education = c.selectbox("Higher education", sorted(df["high_education"].dropna().unique()))
            property_status = d.selectbox("Property status", sorted(df["property_status"].dropna().unique()))
            a, b, c, d = st.columns(4)
            day = a.selectbox("Claim day", sorted(df["claim_day_of_week"].dropna().unique()))
            channel = b.selectbox("Claim channel", sorted(df["channel"].dropna().unique()))
            address = c.selectbox("Address changed", sorted(df["address_change"].dropna().unique()))
            injury = d.selectbox("Injury claim", sorted(df["injury_claim"].dropna().unique()))
            a, b, c, d = st.columns(4)
            vehicle_cat = a.selectbox("Vehicle category", sorted(df["vehicle_category"].dropna().unique()))
            vehicle_price = b.number_input("Vehicle price", min_value=0.0, value=float(df["vehicle_price"].median()), step=1000.0)
            vehicle_color = c.selectbox("Vehicle color", sorted(df["vehicle_color"].dropna().unique()))
            deductible = d.number_input("Policy deductible", min_value=0, value=int(df["policy deductible"].median()), step=100)
            a, b = st.columns(2)
            premium = a.number_input("Annual premium", min_value=0.0, value=float(df["annual premium"].median()), step=50.0)
            days_open = b.number_input("Days open", min_value=0.0, value=float(df["days open"].median()), step=0.5)
            defects = st.number_input("Form defects", min_value=0, max_value=int(df["form defects"].max()), value=int(df["form defects"].median()))

        submitted = st.form_submit_button("🛡️ Analyze Claim", use_container_width=True, type="primary")

    if submitted:
        # Defaults for fields hidden from the quick form.
        row = pd.DataFrame([{
            "age_of_driver": age,
            "gender": gender,
            "marital_status": locals().get("marital", mode_value("marital_status")),
            "safety_rating": safety,
            "annual_income": locals().get("income", float(df["annual_income"].median())),
            "high_education": locals().get("education", mode_value("high_education")),
            "address_change": locals().get("address", mode_value("address_change")),
            "property_status": locals().get("property_status", mode_value("property_status")),
            "claim_day_of_week": locals().get("day", mode_value("claim_day_of_week")),
            "accident_site": site,
            "past_num_of_claims": past,
            "witness_present": witness,
            "liab_prct": liab,
            "channel": locals().get("channel", mode_value("channel")),
            "police_report": police,
            "age_of_vehicle": vehicle_age,
            "vehicle_category": locals().get("vehicle_cat", mode_value("vehicle_category")),
            "vehicle_price": locals().get("vehicle_price", float(df["vehicle_price"].median())),
            "vehicle_color": locals().get("vehicle_color", mode_value("vehicle_color")),
            "total_claim": total_claim,
            "injury_claim": locals().get("injury", mode_value("injury_claim")),
            "policy deductible": locals().get("deductible", int(df["policy deductible"].median())),
            "annual premium": locals().get("premium", float(df["annual premium"].median())),
            "days open": locals().get("days_open", float(df["days open"].median())),
            "form defects": locals().get("defects", int(df["form defects"].median()))
        }])
        prob = float(model.predict_proba(row)[0, 1])
        threshold = float(meta["threshold"])
        is_fraud = prob >= threshold
        if is_fraud:
            st.markdown(f'<div class="result-risk"><h2>⚠️ Fraud Risk Detected</h2><p style="font-size:1.1rem">Estimated fraud probability: <b>{prob*100:.1f}%</b></p><p>Screening threshold: {threshold*100:.0f}%. This is a model-based risk flag, not a final claim decision.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-safe"><h2>✅ Lower Fraud Risk</h2><p style="font-size:1.1rem">Estimated fraud probability: <b>{prob*100:.1f}%</b></p><p>Screening threshold: {threshold*100:.0f}%. This is a model-based risk estimate, not a final claim decision.</p></div>', unsafe_allow_html=True)
        st.progress(min(max(prob, 0), 1), text=f"Fraud probability: {prob*100:.1f}%")

elif page == "📊 EDA":
    st.markdown("""<div class="hero"><div class="badge">EXPLORATORY DATA ANALYSIS</div><h1>Dataset Insights</h1><p>Quick visual exploration of the insurance fraud dataset used by the project.</p></div>""", unsafe_allow_html=True)
    import matplotlib.pyplot as plt
    c1,c2=st.columns(2)
    with c1:
        st.markdown("#### Target distribution")
        counts=df["fraud reported"].value_counts().rename(index={"N":"Not Fraud","Y":"Fraud"})
        fig,ax=plt.subplots(figsize=(6,3.4)); counts.plot(kind="bar",ax=ax); ax.set_ylabel("Claims"); ax.set_xlabel(""); ax.tick_params(axis="x",rotation=0); st.pyplot(fig, use_container_width=True); plt.close(fig)
    with c2:
        st.markdown("#### Claim amount distribution")
        fig,ax=plt.subplots(figsize=(6,3.4)); df["total_claim"].plot(kind="hist",bins=30,ax=ax); ax.set_xlabel("Total claim"); st.pyplot(fig, use_container_width=True); plt.close(fig)
    c1,c2=st.columns(2)
    with c1:
        st.markdown("#### Fraud rate by accident site")
        rate=df.groupby("accident_site")["fraud reported"].apply(lambda s:(s=="Y").mean()).sort_values(ascending=False)
        fig,ax=plt.subplots(figsize=(6,3.4)); rate.plot(kind="bar",ax=ax); ax.set_ylabel("Fraud rate"); ax.set_xlabel(""); ax.tick_params(axis="x",rotation=35); st.pyplot(fig, use_container_width=True); plt.close(fig)
    with c2:
        st.markdown("#### Fraud rate by claim channel")
        rate=df.groupby("channel")["fraud reported"].apply(lambda s:(s=="Y").mean()).sort_values(ascending=False)
        fig,ax=plt.subplots(figsize=(6,3.4)); rate.plot(kind="bar",ax=ax); ax.set_ylabel("Fraud rate"); ax.set_xlabel(""); ax.tick_params(axis="x",rotation=30); st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.markdown("#### Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)

elif page == "🧠 Model Details":
    st.markdown("""<div class="hero"><div class="badge">MODEL & EVALUATION</div><h1>How the model works</h1><p>Transparent project documentation for viva, demonstration and deployment.</p></div>""", unsafe_allow_html=True)
    m=meta["metrics_at_default_0.5"]
    c1,c2,c3,c4,c5=st.columns(5)
    for col,(name,val) in zip([c1,c2,c3,c4,c5],[("Accuracy",m["accuracy"]),("Precision",m["precision"]),("Recall",m["recall"]),("F1",m["f1"]),("ROC-AUC",m["roc_auc"])]):
        col.metric(name,f"{val:.3f}")
    st.info("The deployed screening threshold is 25% to make the demo more fraud-sensitive. The metrics above use the standard 50% threshold and are shown for transparent evaluation.")
    left,right=st.columns(2)
    with left:
        st.markdown('<div class="card"><h3>Preprocessing</h3><p>• Median imputation for numeric fields<br>• Most-frequent imputation for categorical fields<br>• Ordinal encoding with unknown-category handling<br>• Identifier/date fields excluded from training<br>• Stratified 80/20 train-test split</p></div>',unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card"><h3>Models</h3><p><b>Primary:</b> HistGradientBoostingClassifier<br><b>Scratch:</b> NumPy logistic regression implemented without an ML model library.<br><b>Target:</b> fraud reported (Y/N)</p></div>',unsafe_allow_html=True)
    st.markdown("### Scratch implementation result")
    sm=meta["scratch_logistic_metrics"]
    st.write({k:round(v,3) for k,v in sm.items()})
    st.markdown("### Project notes")
    st.caption("This application is intended for academic demonstration. A prediction is a risk flag and should not be treated as an automated real-world claim decision.")

