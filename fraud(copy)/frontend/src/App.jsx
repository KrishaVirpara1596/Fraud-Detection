import { useEffect, useState } from "react";
const API = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");
const n = (k, l, min, max, d) => ({ k, l, min, max, d });
const s = (k, l, o, d) => ({ k, l, o, d });
const YN = [["1", "Yes"], ["0", "No"]];
const FIELDS = {
  "Driver": [n("age_of_driver", "Age (years)", 18, 100, 40), s("gender", "Gender", [["M", "Male"], ["F", "Female"]], "M"),
    s("marital_status", "Married", YN, "1"), n("safety_rating", "Safety rating (0-100)", 0, 100, 75),
    n("annual_income", "Annual income", 0, 300000, 60000), s("high_education", "Higher education", YN, "1"),
    s("address_change", "Changed address recently", YN, "0"), s("property_status", "Home", [["Own", "Owns"], ["Rent", "Rents"]], "Own"),
    n("past_num_of_claims", "Past claims", 0, 10, 0)],
  "Claim": [s("accident_site", "Accident site", [["Local", "Local road"], ["Highway", "Highway"], ["Parking Lot", "Parking lot"]], "Local"),
    s("witness_present", "Witness present", YN, "0"), s("police_report", "Police report filed", YN, "1"),
    n("liab_prct", "Liability (%)", 0, 100, 50), s("channel", "Filed through", [["Broker", "Broker"], ["Phone", "Phone"], ["Online", "Online"]], "Broker"),
    n("total_claim", "Total claim amount", 0, 150000, 20000), n("injury_claim", "Injury claim amount", 0, 100000, 5000),
    n("days_open", "Days claim open", 0, 30, 9), n("form_defects", "Defects on claim form", 0, 15, 4)],
  "Vehicle": [n("age_of_vehicle", "Vehicle age (years)", 0, 30, 5), s("vehicle_category", "Vehicle size", [["Compact", "Compact"], ["Medium", "Medium"], ["Large", "Large"]], "Medium"),
    n("vehicle_price", "Vehicle price", 0, 150000, 23000)],
  "Policy": [s("policy_deductible", "Deductible", [["500", "500"], ["1000", "1,000"], ["2000", "2,000"]], "1000"),
    n("annual_premium", "Annual premium", 0, 3000, 1270)],
};
const PAGES = [["home", "Home"], ["predict", "Check a claim"], ["model", "Model"], ["data", "Data insights"], ["notice", "Disclaimer"]];
const label = (k) => k.replace(/_/g, " ");

function Bar({ name, v, max = 100, tone }) {
  return <div className="bar"><div className="bar-h"><span>{name}</span><b>{v}%</b></div>
    <div className="track"><i className={tone} style={{ width: `${Math.min(100, (v / max) * 100)}%` }} /></div></div>;
}

function Home({ info, go }) {
  return <section className="hero">
    <div><h1>Is this claim genuine?</h1>
      <p className="lead">Enter the details of a vehicle insurance claim and get a fraud risk score in seconds, based on {info ? info.rows_final.toLocaleString() : "thousands of"} past claims.</p>
      <div className="row"><button className="btn" onClick={() => go("predict")}>Check a claim</button>
        <button className="btn ghost" onClick={() => go("model")}>How the model works</button></div></div>
    <aside className="slip"><h3>Model on record</h3>
      {info ? <dl><dt>ROC AUC</dt><dd>{info.roc_auc}%</dd><dt>Accuracy</dt><dd>{info.accuracy}%</dd><dt>Recall (fraud caught)</dt><dd>{info.recall}%</dd>
        <dt>Claims analysed</dt><dd>{info.rows_final.toLocaleString()}</dd></dl>
        : <p className="muted">Waking the server. The first load can take up to a minute on free hosting.</p>}</aside>
  </section>;
}

function Predict() {
  const KEYS = ["annual_income", "age_of_driver", "total_claim", "injury_claim", "past_num_of_claims", "accident_site", "police_report", "witness_present"];
  const all = Object.values(FIELDS).flat();
  const basic = KEYS.map((k) => all.find((x) => x.k === k));
  const more = Object.entries(FIELDS).map(([g, fs]) => [g, fs.filter((x) => !KEYS.includes(x.k))]).filter(([, fs]) => fs.length);
  const init = Object.fromEntries(all.map((x) => [x.k, KEYS.includes(x.k) ? String(x.d) : ""]));
  const [f, setF] = useState(init), [res, setRes] = useState(null), [busy, setBusy] = useState(false), [err, setErr] = useState("");
  const submit = async () => {
    setBusy(true); setErr(""); setRes(null);
    const body = Object.fromEntries(Object.entries(f).filter(([, v]) => v !== ""));
    try {
      const r = await fetch(`${API}/predict`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (!r.ok) throw new Error(); setRes(await r.json());
    } catch { setErr("Could not reach the prediction server. Check your connection and try again in a moment."); }
    setBusy(false);
  };
  const box = (x, opt) => <label key={x.k}>{x.l}
    {x.o ? <select value={f[x.k]} onChange={(e) => setF({ ...f, [x.k]: e.target.value })}>{opt && <option value="">Not sure</option>}{x.o.map(([v, t]) => <option key={v} value={v}>{t}</option>)}</select>
      : <input type="number" min={x.min} max={x.max} placeholder={opt ? "Typical value" : undefined} value={f[x.k]} onChange={(e) => setF({ ...f, [x.k]: e.target.value })} />}</label>;
  return <section><h2>Check a claim</h2><p className="muted">Fill in these 8 details. Anything you skip uses a typical value.</p>
    <fieldset><legend>Essentials</legend><div className="grid">{basic.map((x) => box(x, false))}</div></fieldset>
    <details><summary>Add more details for a sharper score (optional)</summary>
      {more.map(([g, fs]) => <fieldset key={g}><legend>{g}</legend><div className="grid">{fs.map((x) => box(x, true))}</div></fieldset>)}</details>
    <button className="btn" disabled={busy} onClick={submit}>{busy ? "Checking…" : "Check this claim"}</button>
    {err && <p className="err" role="alert">{err}</p>}
    {res && <div className={`result ${res.flagged ? "hot" : "ok"}`} role="status"><div className="pct">{res.probability}%</div>
      <div><h3>{res.risk} fraud risk</h3><p>{res.flagged ? "Refer this claim to an investigator before paying out." : "No strong fraud signals. Process the claim as normal."}</p></div></div>}
  </section>;
}

function Model({ info }) {
  if (!info) return <p className="muted">Loading model details…</p>;
  return <section><h2>{info.algorithm}</h2><p className="muted">Trained with {info.library} on {info.trained_at} using {info.n_features} claim features.</p>
    <div className="cols">
      <div className="panel"><h3>Hyperparameters</h3><dl>{Object.entries(info.params).map(([k, v]) => <><dt>{label(k)}</dt><dd>{v}</dd></>)}
        <dt>decision threshold</dt><dd>{info.threshold}</dd></dl></div>
      <div className="panel"><h3>Test performance</h3>{[["Accuracy", info.accuracy], ["Precision", info.precision], ["Recall", info.recall], ["F1 score", info.f1], ["ROC AUC", info.roc_auc]].map(([a, b]) => <Bar key={a} name={a} v={b} />)}
        <p className="muted small">Train AUC {info.train_auc}% vs test AUC {info.roc_auc}%. 5-fold CV AUC {info.cv_auc_mean}% ± {info.cv_auc_std}%.</p></div>
    </div>
    <div className="panel"><h3>What drives the score</h3>{info.importance.map(([k, v]) => <Bar key={k} name={label(k)} v={v} max={info.importance[0][1]} tone="red" />)}</div></section>;
}

function Data({ info }) {
  if (!info) return <p className="muted">Loading data insights…</p>;
  const ttl = { accident_site: "Accident site", channel: "Filing channel", past_num_of_claims: "Past claims", witness_present: "Witness present (0 = no)", police_report: "Police report (0 = no)" };
  return <section><h2>Data insights</h2><p className="muted">Vehicle insurance fraud dataset, {info.fraud_rate}% of claims flagged as fraud.</p>
    <div className="stats"><div><span>Raw records</span><b>{info.rows_raw.toLocaleString()}</b></div><div><span>Rows removed</span><b>{info.rows_removed}</b></div><div><span>Used for training</span><b>{info.rows_final.toLocaleString()}</b></div></div>
    <p className="muted small">Removed rows had no fraud label or an impossible driver age. Other missing values were filled with the median or most common value.</p>
    <div className="cols">{Object.entries(info.eda).map(([k, v]) => <div className="panel" key={k}><h3>Fraud rate by {ttl[k].toLowerCase()}</h3>
      {Object.entries(v).map(([c, r]) => <Bar key={c} name={c} v={r} max={Math.max(40, ...Object.values(v))} tone="red" />)}</div>)}</div></section>;
}

const Notice = () => <section><h2>Disclaimer</h2><p className="lead">This tool estimates fraud risk from patterns in historical claims. It is built for learning and demonstration.</p>
  <p>A high score is a reason to look closer, not proof of fraud. Do not deny or delay a real claim based on this score alone. Every decision needs a human review.</p></section>;

export default function App() {
  const [page, setPage] = useState("home"), [info, setInfo] = useState(null), [down, setDown] = useState(false);
  useEffect(() => { fetch(`${API}/info`).then((r) => r.json()).then(setInfo).catch(() => setDown(true)); }, []);
  const go = (p) => { setPage(p); window.scrollTo(0, 0); };
  return <><header><b className="logo" onClick={() => go("home")}>ClaimGuard</b>
    <nav>{PAGES.map(([k, t]) => <button key={k} className={page === k ? "on" : ""} onClick={() => go(k)}>{t}</button>)}</nav></header>
    <main>{down && <p className="err">The server is not responding yet. Free hosting sleeps when idle; reload in a minute.</p>}
      {page === "home" && <Home info={info} go={go} />}{page === "predict" && <Predict />}
      {page === "model" && <Model info={info} />}{page === "data" && <Data info={info} />}{page === "notice" && <Notice />}</main></>;
}
