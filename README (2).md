# ⚡ Renewable Energy Prediction for Microgrid Management

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-0.96_R²-FF6600?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-00C49F?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A machine learning system that forecasts solar and wind power output to enable smarter microgrid energy management. Trained on 12,984 hourly records spanning 18 months, the best model achieves **R² = 0.9598** on solar prediction using XGBoost with engineered time features.

---

## The Problem

Renewable sources like solar and wind are inherently variable — output shifts hour by hour based on weather, season, and time of day. Grid operators managing microgrids need accurate short-term forecasts to:
- Pre-charge or discharge battery storage at the right time
- Decide when to draw from or export to the main grid
- Avoid blackouts during sudden generation drops

This project builds and compares three ensemble ML models to make those forecasts reliable.

---

## Results

### Solar Power Prediction

| Model | R² Score | MAE (MW) | RMSE (MW) |
|---|---|---|---|
| Gradient Boosting | 0.9495 | 2,353 | 4,417 |
| Random Forest | 0.9584 | 1,924 | 4,007 |
| **XGBoost** ✅ | **0.9598** | **2,017** | **3,939** |

### Wind Power Prediction

| Model | R² Score | MAE (MW) | RMSE (MW) |
|---|---|---|---|
| XGBoost | 0.6887 | 2,332 | 3,239 |

> Wind R² is lower because wind speed follows less predictable daily patterns compared to solar irradiance. The model captures seasonal and demand-based trends well but struggles with sudden wind shifts — a known challenge in wind forecasting literature.

---

## What Makes This Different from Basic Forecasting

**Cyclic time encoding** — Instead of feeding `Hour=23` and `Hour=0` as numerically distant values, the model uses:
```python
Hour_sin = sin(2π × Hour / 24)
Hour_cos = cos(2π × Hour / 24)
```
This is standard practice in time-series ML and measurably improves accuracy at day boundaries.

**SHAP explainability** — Beyond just accuracy metrics, SHAP (SHapley Additive exPlanations) reveals *why* each prediction was made. The analysis confirmed that Wind Speed has a 0.92 correlation with total energy output, which aligned exactly with the correlation heatmap from data exploration.

**Three-model comparison** — Rather than picking one algorithm, the pipeline trains Gradient Boosting, Random Forest, and XGBoost side by side, evaluating all three on identical train/test splits for a fair comparison.

---

## Project Structure

```
├── renewable_prediction_upgraded.py   # Full ML pipeline — preprocessing, training, SHAP, plots
├── app.py                             # Streamlit dashboard with live prediction
├── renewable_prediction.ipynb         # Original exploratory notebook
├── your_dataset.xlsx                  # Hourly energy dataset (Jan 2024 – Jun 2025)
├── Plant_1_Generation_Data.csv        # Raw generation records
├── Plant_1_Weather_Sensor_Data.csv    # Weather sensor readings
├── fig1_model_comparison.png          # GBM vs RF vs XGBoost bar chart
├── fig2_shap_summary.png              # SHAP beeswarm plot
├── fig3_actual_vs_pred.png            # Actual vs predicted time series
└── requirements.txt
```

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/renewable-energy-prediction-microgrid.git
cd renewable-energy-prediction-microgrid
pip install -r requirements.txt

# Run the full ML pipeline (generates plots + CSV)
python renewable_prediction_upgraded.py

# Launch the interactive dashboard
streamlit run app.py
```

---

## Dashboard Features

The Streamlit dashboard (`app.py`) lets you:

- Switch between Solar and Wind prediction targets
- Select any of the three models from a dropdown
- Adjust number of estimators and test split size with sliders
- See live SHAP beeswarm and bar plots update in real time
- Enter custom input values (hour, month, demand) and get an instant MW prediction
- Download predictions as CSV

---

## Key Features Used

| Feature | Type | Why It Matters |
|---|---|---|
| Wind (MW) | Raw | Strong 0.92 correlation with total output |
| Demand (MW) | Raw | Captures load-driven generation patterns |
| Hour_sin / Hour_cos | Engineered | Cyclic encoding — no false distance between Hour 23 and Hour 0 |
| Month_sin / Month_cos | Engineered | Captures seasonal variation continuously |
| Season | Derived | Explicit seasonal grouping (0=Winter … 3=Autumn) |
| IsWeekend | Derived | Demand patterns shift on weekends |
| DayOfWeek | Raw | Weekly consumption rhythm |

---

## System Architecture

The microgrid model (built in MATLAB Simulink) integrates:
- Solar PV and wind turbine sources connected via DC/DC converters
- Bidirectional DC bus with battery storage
- Pulse-mounted transformer for grid connection
- A management system layer that receives ML predictions and switches energy sources accordingly

The ML predictions feed into this management layer to pre-emptively switch between renewable sources, battery, and grid based on forecasted generation.

---

## Tech Stack

- **Python 3.10+**
- **scikit-learn** — GradientBoostingRegressor, RandomForestRegressor
- **XGBoost** — Primary model
- **SHAP** — Feature explainability
- **Streamlit** — Interactive dashboard
- **pandas / numpy / matplotlib / seaborn**
- **MATLAB Simulink** — Microgrid simulation

---

## Dataset

- 12,984 hourly records, January 2024 – June 2025
- Features: Timestamp, Demand (MW), Wind (MW), Solar (MW), Total Generation (MW)
- Engineered features: Hour, Month, DayOfWeek, IsWeekend, Season, cyclic encodings
- 80/20 train-test split, `random_state=42` for reproducibility

