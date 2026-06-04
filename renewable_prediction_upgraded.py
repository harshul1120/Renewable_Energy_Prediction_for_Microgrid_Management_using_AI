# =============================================================================
# Intelligent Prediction of Renewable Energy — UPGRADED ML Pipeline
# NIT Delhi | Electrical Engineering | 6th Semester Project
# Additions: XGBoost, RandomForest, SHAP Feature Importance
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: Load and Clean Data
# ──────────────────────────────────────────────────────────────────────────────
df = pd.read_excel("your_dataset.xlsx")
df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["Timestamp"])

for col in ["Demand (MW)", "Wind (MW)", "Solar (MW)", "Total Generation (MW)"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna()
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: Feature Engineering (NEW — cyclic time encoding + season)
# ──────────────────────────────────────────────────────────────────────────────
df["Hour"]       = df["Timestamp"].dt.hour
df["Month"]      = df["Timestamp"].dt.month
df["DayOfWeek"]  = df["Timestamp"].dt.dayofweek
df["IsWeekend"]  = (df["DayOfWeek"] >= 5).astype(int)
df["Season"]     = df["Month"].map({12:0,1:0,2:0, 3:1,4:1,5:1,
                                     6:2,7:2,8:2, 9:3,10:3,11:3})
# Cyclic encoding prevents model from seeing Hour 23 and Hour 0 as far apart
df["Hour_sin"]   = np.sin(2 * np.pi * df["Hour"] / 24)
df["Hour_cos"]   = np.cos(2 * np.pi * df["Hour"] / 24)
df["Month_sin"]  = np.sin(2 * np.pi * df["Month"] / 12)
df["Month_cos"]  = np.cos(2 * np.pi * df["Month"] / 12)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: Prepare Features and Targets
# ──────────────────────────────────────────────────────────────────────────────
SOLAR_FEATURES = ["Wind (MW)", "Demand (MW)", "Hour", "Month", "DayOfWeek",
                  "IsWeekend", "Season", "Hour_sin", "Hour_cos",
                  "Month_sin", "Month_cos"]

WIND_FEATURES  = ["Solar (MW)", "Demand (MW)", "Hour", "Month", "DayOfWeek",
                  "IsWeekend", "Season", "Hour_sin", "Hour_cos",
                  "Month_sin", "Month_cos"]

X_solar, y_solar = df[SOLAR_FEATURES], df["Solar (MW)"]
X_wind,  y_wind  = df[WIND_FEATURES],  df["Wind (MW)"]

Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(X_solar, y_solar, test_size=0.2, random_state=42)
Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(X_wind,  y_wind,  test_size=0.2, random_state=42)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: Train Three Models for Solar Prediction (Model Comparison)
# ──────────────────────────────────────────────────────────────────────────────
solar_models = {
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                   max_depth=5, random_state=42),
    "RandomForest":     RandomForestRegressor(n_estimators=200, max_depth=10,
                                              n_jobs=-1, random_state=42),
    "XGBoost":          xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
                                          max_depth=6, subsample=0.8,
                                          colsample_bytree=0.8,
                                          random_state=42, verbosity=0)
}

solar_results = {}
print("\n── Solar Power Model Comparison ──────────────────────────")
for name, model in solar_models.items():
    model.fit(Xs_tr, ys_tr)
    pred = model.predict(Xs_te)
    solar_results[name] = {
        "model": model, "pred": pred,
        "MAE":  mean_absolute_error(ys_te, pred),
        "RMSE": np.sqrt(mean_squared_error(ys_te, pred)),
        "R2":   r2_score(ys_te, pred)
    }
    print(f"  {name:20s} | MAE: {solar_results[name]['MAE']:8.2f} | "
          f"RMSE: {solar_results[name]['RMSE']:8.2f} | R²: {solar_results[name]['R2']:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: Train XGBoost for Wind
# ──────────────────────────────────────────────────────────────────────────────
print("\n── Wind Power (XGBoost) ───────────────────────────────────")
wind_xgb = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=42, verbosity=0)
wind_xgb.fit(Xw_tr, yw_tr)
wind_pred = wind_xgb.predict(Xw_te)
print(f"  XGBoost  | MAE: {mean_absolute_error(yw_te,wind_pred):8.2f} | "
      f"RMSE: {np.sqrt(mean_squared_error(yw_te,wind_pred)):8.2f} | "
      f"R²: {r2_score(yw_te,wind_pred):.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 6: SHAP Feature Importance (NEW — Explainability)
# ──────────────────────────────────────────────────────────────────────────────
print("\nComputing SHAP values (this takes ~30 seconds)...")
best_solar_model = solar_results["XGBoost"]["model"]
explainer   = shap.TreeExplainer(best_solar_model)
sample_500  = Xs_te.sample(500, random_state=42)
shap_values = explainer.shap_values(sample_500)

# Plot 1 — SHAP Summary (Beeswarm)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, sample_500, feature_names=SOLAR_FEATURES, show=False)
plt.title("SHAP Feature Importance — Solar Power Prediction (XGBoost)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("shap_summary_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_summary_plot.png")

# Plot 2 — SHAP Bar Plot (mean |SHAP|)
plt.figure(figsize=(9, 5))
shap.summary_plot(shap_values, sample_500, feature_names=SOLAR_FEATURES,
                  plot_type="bar", show=False)
plt.title("Mean |SHAP| — Feature Importance Ranking", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("shap_bar_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_bar_plot.png")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 7: Model Comparison Bar Chart (NEW)
# ──────────────────────────────────────────────────────────────────────────────
BG   = "#0f172a"
CARD = "#1e293b"
BLUE = "#38bdf8"; VIOLET = "#818cf8"; GREEN = "#34d399"
model_names = list(solar_results.keys())

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.patch.set_facecolor(BG)
for ax, metric, color in zip(axes, ["MAE","RMSE","R2"], [BLUE, VIOLET, GREEN]):
    ax.set_facecolor(CARD)
    vals = [solar_results[m][metric] for m in model_names]
    bars = ax.bar(model_names, vals, color=color, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=9, color="white", fontweight="bold")
    ax.set_title(metric, color="white", fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors="white", labelsize=8)
    [ax.spines[s].set_visible(False) for s in ["top","right"]]
    [ax.spines[s].set_color("#334155") for s in ["bottom","left"]]
    [lbl.set_color("white") for lbl in ax.get_xticklabels()]
fig.suptitle("Solar Power Prediction — GradientBoosting vs RandomForest vs XGBoost",
             color="white", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved: model_comparison.png")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 8: Actual vs Predicted Plots (both Solar and Wind)
# ──────────────────────────────────────────────────────────────────────────────
best_pred = solar_results["XGBoost"]["pred"]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.patch.set_facecolor(BG)
fig.suptitle("Actual vs Predicted — Solar & Wind Power", color="white",
             fontsize=15, fontweight="bold")

for ax in axes.flat:
    ax.set_facecolor(CARD)
    ax.tick_params(colors="white")
    [ax.spines[s].set_color("#334155") for s in ax.spines]
    ax.grid(alpha=0.15, color="white")

n = 150
# Solar time-series
axes[0,0].plot(ys_te.values[:n], color=BLUE, lw=1.5, label="Actual")
axes[0,0].plot(best_pred[:n], color="#f97316", lw=1.5, ls="--", label="XGBoost")
axes[0,0].set_title("Solar — Time Series (XGBoost)", color="white", fontweight="bold")
axes[0,0].set_ylabel("Solar (MW)", color="white")
axes[0,0].legend(facecolor="#334155", labelcolor="white", fontsize=9)

# Solar scatter
axes[0,1].scatter(ys_te[:800], best_pred[:800], alpha=0.3, color=GREEN, s=8)
mn,mx = ys_te.min(), ys_te.max()
axes[0,1].plot([mn,mx],[mn,mx], color="#f87171", lw=2, label="Perfect Fit")
axes[0,1].set_title(f"Solar — Scatter (R²={solar_results['XGBoost']['R2']:.4f})",
                    color="white", fontweight="bold")
axes[0,1].set_xlabel("Actual (MW)", color="white")
axes[0,1].set_ylabel("Predicted (MW)", color="white")
axes[0,1].legend(facecolor="#334155", labelcolor="white")

# Wind time-series
axes[1,0].plot(yw_te.values[:n], color=BLUE, lw=1.5, label="Actual")
axes[1,0].plot(wind_pred[:n], color="#f97316", lw=1.5, ls="--", label="XGBoost")
axes[1,0].set_title("Wind — Time Series (XGBoost)", color="white", fontweight="bold")
axes[1,0].set_ylabel("Wind (MW)", color="white")
axes[1,0].legend(facecolor="#334155", labelcolor="white", fontsize=9)

# Wind scatter
axes[1,1].scatter(yw_te[:800], wind_pred[:800], alpha=0.3, color=VIOLET, s=8)
mn,mx = yw_te.min(), yw_te.max()
axes[1,1].plot([mn,mx],[mn,mx], color="#f87171", lw=2, label="Perfect Fit")
axes[1,1].set_title(f"Wind — Scatter (R²={r2_score(yw_te,wind_pred):.4f})",
                    color="white", fontweight="bold")
axes[1,1].set_xlabel("Actual (MW)", color="white")
axes[1,1].set_ylabel("Predicted (MW)", color="white")
axes[1,1].legend(facecolor="#334155", labelcolor="white")

plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved: actual_vs_predicted.png")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 9: Save Predictions CSV
# ──────────────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame({
    "Actual_Solar":    ys_te.values,
    "Predicted_Solar_GBM": solar_results["GradientBoosting"]["pred"],
    "Predicted_Solar_RF":  solar_results["RandomForest"]["pred"],
    "Predicted_Solar_XGB": solar_results["XGBoost"]["pred"],
    "Actual_Wind":     yw_te.values,
    "Predicted_Wind_XGB":  wind_pred
})
results_df.to_csv("renewable_predictions_upgraded.csv", index=False)
print("Saved: renewable_predictions_upgraded.csv")
print("\n✅ All done! Check the .png files and CSV.")
