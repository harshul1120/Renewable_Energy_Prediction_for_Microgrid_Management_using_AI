# =============================================================================
# Streamlit Dashboard — Renewable Energy Prediction
# Run with: streamlit run app.py
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Renewable Energy Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.8rem; color: #94a3b8; margin-top: 4px; }
    .section-header {
        font-size: 1.3rem; font-weight: 700;
        color: #e2e8f0; padding: 0.5rem 0;
        border-bottom: 2px solid #38bdf8;
        margin-bottom: 1rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    .stSelectbox label, .stSlider label { color: #94a3b8 !important; }
    h1, h2, h3 { color: #e2e8f0 !important; }
    p, li { color: #cbd5e1; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background: linear-gradient(90deg, #0ea5e9, #6366f1);
     padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;'>
  <h1 style='color:white; margin:0; font-size:1.9rem;'>⚡ Renewable Energy Prediction Dashboard</h1>
  <p style='color:#e0f2fe; margin:0.3rem 0 0; font-size:0.95rem;'>
    Intelligent Microgrid Management · NIT Delhi · GradientBoosting · RandomForest · XGBoost · SHAP
  </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/b/b6/NIT_Delhi_logo.png/200px-NIT_Delhi_logo.png",
             width=80)
    st.markdown("### ⚙️ Configuration")
    target = st.selectbox("Prediction Target", ["Solar (MW)", "Wind (MW)"])
    model_choice = st.selectbox("ML Model", ["XGBoost", "RandomForest", "GradientBoosting"])
    n_estimators = st.slider("Number of Trees", 50, 300, 200, 50)
    test_size = st.slider("Test Set Size (%)", 10, 30, 20, 5)
    show_shap = st.checkbox("Show SHAP Analysis", value=True)
    st.markdown("---")
    st.markdown("**Project Team**")
    st.markdown("Aman Dagar · Harshul Gupta  \nMayank Chaudhary · Nidhi Chauhan")
    st.markdown("*Supervisor: Dr. Amit Kumar Singh*")

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("your_dataset.xlsx")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    for col in ["Demand (MW)", "Wind (MW)", "Solar (MW)", "Total Generation (MW)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    df["Hour"]      = df["Timestamp"].dt.hour
    df["Month"]     = df["Timestamp"].dt.month
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["Season"]    = df["Month"].map({12:0,1:0,2:0,3:1,4:1,5:1,
                                        6:2,7:2,8:2,9:3,10:3,11:3})
    df["Hour_sin"]  = np.sin(2*np.pi*df["Hour"]/24)
    df["Hour_cos"]  = np.cos(2*np.pi*df["Hour"]/24)
    df["Month_sin"] = np.sin(2*np.pi*df["Month"]/12)
    df["Month_cos"] = np.cos(2*np.pi*df["Month"]/12)
    return df

df = load_data()

# ── Dataset Overview ──────────────────────────────────────────────────────────
with st.expander("📊 Dataset Overview", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Date Range", f"{df['Timestamp'].min().strftime('%b %Y')} – {df['Timestamp'].max().strftime('%b %Y')}")
    c3.metric("Avg Solar (MW)", f"{df['Solar (MW)'].mean():,.0f}")
    c4.metric("Avg Wind (MW)",  f"{df['Wind (MW)'].mean():,.0f}")
    st.dataframe(df[["Timestamp","Demand (MW)","Wind (MW)","Solar (MW)",
                      "Total Generation (MW)"]].tail(10), use_container_width=True)

# ── Train Model ───────────────────────────────────────────────────────────────
SOLAR_FEAT = ["Wind (MW)","Demand (MW)","Hour","Month","DayOfWeek",
              "IsWeekend","Season","Hour_sin","Hour_cos","Month_sin","Month_cos"]
WIND_FEAT  = ["Solar (MW)","Demand (MW)","Hour","Month","DayOfWeek",
              "IsWeekend","Season","Hour_sin","Hour_cos","Month_sin","Month_cos"]

features = SOLAR_FEAT if target == "Solar (MW)" else WIND_FEAT
X = df[features]
y = df[target]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size/100, random_state=42)

@st.cache_resource
def train_model(model_name, n_est, feat_tuple, target_name, ts):
    features = list(feat_tuple)
    X = df[features]; y = df[target_name]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=ts/100, random_state=42)
    if model_name == "XGBoost":
        m = xgb.XGBRegressor(n_estimators=n_est, learning_rate=0.05, max_depth=6,
                              subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    elif model_name == "RandomForest":
        m = RandomForestRegressor(n_estimators=n_est, max_depth=10, n_jobs=-1, random_state=42)
    else:
        m = GradientBoostingRegressor(n_estimators=n_est, learning_rate=0.05,
                                       max_depth=5, random_state=42)
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    return m, pred, y_te

with st.spinner(f"Training {model_choice} for {target}..."):
    model, pred, y_test = train_model(model_choice, n_estimators,
                                       tuple(features), target, test_size)

mae  = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2   = r2_score(y_test, pred)

# ── Metrics Row ───────────────────────────────────────────────────────────────
st.markdown(f"<div class='section-header'>📈 {model_choice} Model Results — {target}</div>",
            unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("R² Score",  f"{r2:.4f}",  delta=f"{'Excellent' if r2>0.95 else 'Good' if r2>0.85 else 'Fair'}")
c2.metric("MAE",       f"{mae:,.1f} MW")
c3.metric("RMSE",      f"{rmse:,.1f} MW")
c4.metric("Test Points", f"{len(y_test):,}")

# ── Actual vs Predicted ───────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Time Series — Actual vs Predicted**")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#0f172a"); ax.set_facecolor("#1e293b")
    ax.tick_params(colors="white"); ax.grid(alpha=0.15, color="white")
    [ax.spines[s].set_color("#334155") for s in ax.spines]
    n = 120
    ax.plot(y_test.values[:n], color="#38bdf8", lw=1.5, label="Actual")
    ax.plot(pred[:n], color="#f97316", lw=1.5, ls="--", label="Predicted")
    ax.set_ylabel(target, color="white"); ax.set_xlabel("Sample", color="white")
    ax.legend(facecolor="#334155", labelcolor="white", fontsize=9)
    st.pyplot(fig, use_container_width=True)

with col2:
    st.markdown("**Scatter — Actual vs Predicted**")
    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    fig2.patch.set_facecolor("#0f172a"); ax2.set_facecolor("#1e293b")
    ax2.tick_params(colors="white"); ax2.grid(alpha=0.15, color="white")
    [ax2.spines[s].set_color("#334155") for s in ax2.spines]
    ax2.scatter(y_test[:600], pred[:600], alpha=0.3, color="#34d399", s=8)
    mn, mx = y_test.min(), y_test.max()
    ax2.plot([mn,mx],[mn,mx], color="#f87171", lw=2, label="Perfect Fit")
    ax2.set_xlabel(f"Actual {target}", color="white")
    ax2.set_ylabel(f"Predicted {target}", color="white")
    ax2.legend(facecolor="#334155", labelcolor="white")
    st.pyplot(fig2, use_container_width=True)

# ── Model Comparison Table ────────────────────────────────────────────────────
st.markdown(f"<div class='section-header'>🏆 All-Model Comparison — {target}</div>",
            unsafe_allow_html=True)

@st.cache_resource
def compare_all(target_name, feat_tuple, ts):
    features = list(feat_tuple)
    X = df[features]; y = df[target_name]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=ts/100, random_state=42)
    rows = []
    for name, m in [
        ("GradientBoosting", GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)),
        ("RandomForest",     RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)),
        ("XGBoost",          xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, verbosity=0, random_state=42))
    ]:
        m.fit(X_tr, y_tr); p = m.predict(X_te)
        rows.append({"Model": name,
                     "MAE":  round(mean_absolute_error(y_te,p),2),
                     "RMSE": round(np.sqrt(mean_squared_error(y_te,p)),2),
                     "R²":   round(r2_score(y_te,p),4)})
    return pd.DataFrame(rows)

with st.spinner("Running all 3 models for comparison..."):
    comp_df = compare_all(target, tuple(features), test_size)

best_r2 = comp_df["R²"].max()
def highlight_best(row):
    return ["background-color: #14532d; color: white" if row["R²"]==best_r2
            else "background-color: #1e293b; color: #cbd5e1"]*len(row)

st.dataframe(comp_df.style.apply(highlight_best, axis=1), use_container_width=True)

# ── SHAP Analysis ─────────────────────────────────────────────────────────────
if show_shap:
    st.markdown("<div class='section-header'>🔍 SHAP Explainability (XGBoost)</div>",
                unsafe_allow_html=True)
    st.info("SHAP (SHapley Additive exPlanations) shows which features drive each prediction. "
            "Red = high feature value pushes prediction up. Blue = low value pulls it down.")

    @st.cache_resource
    def get_shap(feat_tuple, target_name, ts):
        features = list(feat_tuple)
        X = df[features]; y = df[target_name]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=ts/100, random_state=42)
        m = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, verbosity=0, random_state=42)
        m.fit(X_tr, y_tr)
        exp = shap.TreeExplainer(m)
        sample = X_te.sample(min(400, len(X_te)), random_state=42)
        sv = exp.shap_values(sample)
        return sv, sample, features

    with st.spinner("Computing SHAP values..."):
        shap_vals, shap_sample, shap_feats = get_shap(tuple(features), target, test_size)

    sc1, sc2 = st.columns(2)
    with sc1:
        fig_s1, _ = plt.subplots(figsize=(7, 5))
        shap.summary_plot(shap_vals, shap_sample, feature_names=shap_feats,
                          plot_type="bar", show=False)
        plt.title("Mean |SHAP| — Feature Ranking", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_s1, use_container_width=True)

    with sc2:
        fig_s2, _ = plt.subplots(figsize=(7, 5))
        shap.summary_plot(shap_vals, shap_sample, feature_names=shap_feats, show=False)
        plt.title("SHAP Beeswarm — Feature Impact", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_s2, use_container_width=True)

# ── Live Predictor ────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>🎯 Live Prediction</div>", unsafe_allow_html=True)
st.markdown("Enter values to get a real-time energy prediction from the trained model:")

lc1, lc2, lc3, lc4 = st.columns(4)
hour     = lc1.slider("Hour of Day", 0, 23, 12)
month    = lc2.slider("Month", 1, 12, 6)
demand   = lc3.number_input("Demand (MW)", 100000.0, 250000.0, 190000.0, 1000.0)
other_mw = lc4.number_input("Wind (MW)" if target=="Solar (MW)" else "Solar (MW)",
                              0.0, 70000.0, 10000.0, 500.0)

input_dict = {
    "Hour": hour, "Month": month,
    "DayOfWeek": 2, "IsWeekend": 0,
    "Season": {1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3,12:0}[month],
    "Hour_sin": np.sin(2*np.pi*hour/24),
    "Hour_cos": np.cos(2*np.pi*hour/24),
    "Month_sin": np.sin(2*np.pi*month/12),
    "Month_cos": np.cos(2*np.pi*month/12),
    "Demand (MW)": demand
}
if target == "Solar (MW)":
    input_dict["Wind (MW)"] = other_mw
else:
    input_dict["Solar (MW)"] = other_mw

input_df = pd.DataFrame([{f: input_dict[f] for f in features}])
live_pred = model.predict(input_df)[0]

st.markdown(f"""
<div style='background: linear-gradient(135deg, #0ea5e9, #6366f1);
     padding: 1.5rem 2rem; border-radius: 12px; margin-top: 1rem; text-align: center;'>
  <div style='color: #e0f2fe; font-size: 0.9rem;'>Predicted {target}</div>
  <div style='color: white; font-size: 3rem; font-weight: 800;'>{live_pred:,.1f} MW</div>
  <div style='color: #bae6fd; font-size: 0.85rem;'>Model: {model_choice} | Hour: {hour}:00 | Month: {month}</div>
</div>
""", unsafe_allow_html=True)

# ── Download ──────────────────────────────────────────────────────────────────
results_out = pd.DataFrame({"Actual": y_test.values, "Predicted": pred})
st.download_button("⬇️ Download Predictions CSV", results_out.to_csv(index=False),
                   "predictions.csv", "text/csv")
