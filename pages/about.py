import streamlit as st
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.nav import render_nav, get_theme
render_nav("pages/about.py")
t = get_theme()

st.markdown("<div style='font-size:2.2rem;font-weight:800;line-height:1.2;margin-bottom:0.2rem'>About</div>", unsafe_allow_html=True)
st.markdown(f"<hr style='border:none;border-top:1px solid {t['border']};margin:0.8rem 0 1rem 0'>", unsafe_allow_html=True)

st.markdown(f"""
<div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.3);
            border-radius:12px;padding:1.2rem 1.5rem;margin:0.8rem 0 1.8rem 0;">
  <div style="color:{t["text_muted"]};font-size:0.88rem;margin-bottom:0.2rem">Developed by</div>
  <div style="font-size:1.1rem;font-weight:700;color:{t["text_primary"]};margin-bottom:0.3rem">
    Tavishi Sharma
  </div>
  <div style="color:{t["text_muted"]};font-size:0.92rem;">
    Honors Thesis · Department of Computer Science
  </div>
  <div style="margin-top:0.8rem;display:flex;gap:1rem;flex-wrap:wrap">
    <a href="https://www.linkedin.com/in/tavishi-sharma05/" target="_blank"
       style="display:inline-flex;align-items:center;gap:0.4rem;padding:0.3rem 0.85rem;
              border-radius:8px;background:{t["link_btn_bg"]};border:1px solid {t["link_btn_border"]};
              color:{t["text_primary"]};text-decoration:none;font-size:0.85rem;font-weight:500;">
      🔗 LinkedIn
    </a>
    <a href="https://github.com/Sharma-Tavishi" target="_blank"
       style="display:inline-flex;align-items:center;gap:0.4rem;padding:0.3rem 0.85rem;
              border-radius:8px;background:{t["link_btn_bg"]};border:1px solid {t["link_btn_border"]};
              color:{t["text_primary"]};text-decoration:none;font-size:0.85rem;font-weight:500;">
      💻 GitHub
    </a>
    <a href="https://sharma-tavishi.github.io" target="_blank"
       style="display:inline-flex;align-items:center;gap:0.4rem;padding:0.3rem 0.85rem;
              border-radius:8px;background:{t["link_btn_bg"]};border:1px solid {t["link_btn_border"]};
              color:{t["text_primary"]};text-decoration:none;font-size:0.85rem;font-weight:500;">
      🌐 Website
    </a>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("## Overview")
st.markdown("""
Air travel delays affect nearly **one in five US domestic flights**, costing passengers time and airlines
billions annually. This thesis builds a complete end-to-end ML system that predicts whether a flight
will be on-time, mildly delayed, or significantly delayed, and makes those predictions accessible
through natural language.

Instead of filling out a technical form, a user can simply ask:
*"Will my Delta flight from Atlanta to JFK tomorrow morning be delayed?"*
and receive a data-driven, plain-English answer powered by a LightGBM model and the Anthropic Claude API.
""")

st.divider()

st.markdown("## How It Works")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 1. Parse")
    st.markdown("Your natural language question is sent to **Claude** (Anthropic), which extracts structured flight details: origin, destination, airline, departure time, and date.")
with col2:
    st.markdown("### 2. Predict")
    st.markdown("The extracted details feed into a **LightGBM** model trained on ~987,000 US domestic flights. It outputs a delay class and estimated delay in minutes.")
with col3:
    st.markdown("### 3. Explain")
    st.markdown("Claude generates a friendly, plain-English explanation factoring in weather conditions, historical route performance, and prediction confidence.")

st.divider()

st.markdown("## Machine Learning Model")

mc1, mc2 = st.columns(2)
with mc1:
    st.markdown("**Algorithm**")
    st.markdown("""
- **Classifier:** LightGBM (`LGBMClassifier`)
  Predicts delay class: On-time / Minor / Major
- **Regressor:** LightGBM (`LGBMRegressor`)
  Estimates exact delay in minutes
- **Encoder:** Ordinal Encoder for categorical features
- **Balancing:** Class weights to handle imbalanced delay classes
""")
with mc2:
    st.markdown("**Why LightGBM?**")
    st.markdown("""
- Outperforms Logistic Regression and Random Forest on tabular aviation data
- Handles mixed feature types (categorical + numeric) natively
- Built-in support for missing values
- Faster training than Random Forest on large datasets
- State-of-the-art on structured/tabular benchmarks
""")

st.markdown("**Features used for prediction (14 total)**")
feat_col1, feat_col2 = st.columns(2)
with feat_col1:
    st.markdown("""
| Feature | Description |
|---|---|
| `dep_hour` | Scheduled departure hour (0–23) |
| `arr_hour` | Scheduled arrival hour (0–23) |
| `month` | Month of flight (1–12) |
| `dayofweek` | Day of week (1=Mon, 7=Sun) |
| `distance` | Route distance in miles |
| `carrier` | Airline IATA code |
| `origin_topK` | Origin airport (top 50 or OTHER) |
""")
with feat_col2:
    st.markdown("""
| Feature | Description |
|---|---|
| `dest_topK` | Destination airport (top 50 or OTHER) |
| `route_avg_delay` | Historical avg delay on this route |
| `temp` | Temperature at origin (°C) |
| `wspd` | Wind speed at origin (km/h) |
| `prcp` | Precipitation (mm) |
| `snow` | Snowfall (mm) |
| `coco` | Weather condition code |
""")

st.markdown("**Output classes**")
st.markdown("""
| Class | Label | Definition |
|---|---|---|
| 0 | On-time | Arrives ≤ 15 minutes late |
| 1 | Minor Delay | Arrives 16–59 minutes late |
| 2 | Major Delay | Arrives 60+ minutes late |
""")

st.divider()

st.markdown("## Model Comparison")
st.markdown("Three models were evaluated on the same dataset to justify the LightGBM choice:")
st.markdown("""
| Model | Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 41.1% | 0.294 | 0.557 |
| Random Forest | 71.5% | 0.339 | 0.569 |
| **LightGBM** | **66.5%** | **0.374** | **0.600** |
""")
st.info("Accuracy alone is misleading on imbalanced datasets. LightGBM wins on Macro F1 and ROC-AUC, the metrics that matter for minority class (delay) detection.")

st.divider()

st.markdown("## Training Data")
dc1, dc2 = st.columns(2)
with dc1:
    st.markdown("**Flight Data: Bureau of Transportation Statistics**")
    st.markdown("""
- Source: [BTS On-Time Reporting](https://www.transtats.bts.gov/)
- Coverage: January 2022 – November 2025 (47 monthly files)
- Sampling: 21,000 random rows per file → ~987,000 flights total
- Airlines: 9 US domestic carriers
- Routes: 50 top origin + 50 top destination airports
""")
with dc2:
    st.markdown("**Weather Data: Open-Meteo**")
    st.markdown("""
- Source: [Open-Meteo Historical API](https://open-meteo.com/)
- Coverage: Hourly historical weather for 51 major US airports
- Variables: Temperature, wind speed, precipitation, snowfall, weather code
- Merge: Joined to each flight by airport + date + departure hour
- Free, no API key required
""")

st.divider()

st.markdown("## App Features")
fc1, fc2, fc3, fc4 = st.columns(4)
with fc1:
    st.markdown("**✈️ Predict**")
    st.markdown("Natural language queries or manual input. Predicts delay class + estimated minutes with AI-generated explanation.")
with fc2:
    st.markdown("**🔍 Explain**")
    st.markdown("SHAP-based explainability showing which features drove the prediction for a specific flight.")
with fc3:
    st.markdown("**📊 Insights**")
    st.markdown("Airline reliability rankings and US route delay map, computed from the full 987K training flights.")
with fc4:
    st.markdown("**🤖 AI Assistant**")
    st.markdown("Powered by Anthropic Claude (`claude-sonnet-4-6`) for natural language parsing and conversational responses.")

st.divider()

st.markdown("## Tech Stack")
tc1, tc2, tc3 = st.columns(3)
with tc1:
    st.markdown("**ML & Data**")
    st.markdown("""
- Python 3.12
- LightGBM
- scikit-learn
- SHAP
- pandas / NumPy
- joblib
""")
with tc2:
    st.markdown("**AI & NLP**")
    st.markdown("""
- Anthropic Claude API
- Model: `claude-sonnet-4-6`
- Used for:
  - NL → structured JSON parsing
  - Plain-English explanations
""")
with tc3:
    st.markdown("**Web App & Data**")
    st.markdown("""
- Streamlit
- Plotly (maps & charts)
- Open-Meteo API (weather)
- AeroDataBox API (live flight lookup)
- python-dotenv
- Dark / Light theme
""")

st.divider()

st.markdown("""
<div style="text-align:center;padding:0.5rem 0 1rem 0">
  <p style="color:#64748b;font-size:0.82rem;margin:0">
    Built by
    <a href="https://sharma-tavishi.github.io" target="_blank"
       style="color:#a5b4fc;text-decoration:none;font-weight:500;">Tavishi Sharma</a>
    &nbsp;·&nbsp; Honors Thesis 2026 &nbsp;·&nbsp;
    Data: BTS + Open-Meteo &nbsp;·&nbsp; AI: Anthropic Claude
  </p>
</div>
""", unsafe_allow_html=True)
