# Flight Delay Predictor

An end-to-end machine learning system that predicts US domestic flight delays and explains them in plain English. Built as an Honors Thesis project.

## What it does

- Predicts whether a flight will be **on-time, mildly delayed (16–59 min), or significantly delayed (60+ min)**
- Accepts queries in **natural language** ("Will my Delta flight from Atlanta to JFK tomorrow morning be delayed?") or via a **manual input form**
- Looks up live flight details by **flight number** (AviationStack API)
- Explains each prediction using **SHAP values**
- Shows **airline reliability rankings** and a **US route delay map**

## Project Structure

```
├── app.py               # entry point
├── pages/
│   ├── predict.py       # NL + manual prediction
│   ├── explain.py       # SHAP explainability
│   ├── insights.py      # airline stats + route map
│   └── about.py         # project info
├── utils/nav.py         # shared nav bar
├── notebooks/
│   ├── DataPreprocess.ipynb       # cleans BTS flight data, merges weather, exports parquet
│   ├── ModelComparison.ipynb      # trains and compares Logistic Regression, Random Forest, LightGBM
│   ├── TrainOnColab.ipynb         # full training pipeline optimized for Google Colab
│   ├── LLM_integration.ipynb      # experiments with Claude API for NL parsing
│   └── generate_poster_charts.py  # generates charts used in the thesis poster
├── assets/              # charts and diagrams
└── docs/                # user testing guide, project overview
```

> **Note:** `Data/` and `models/` are excluded from this repo due to size.

## Tech Stack

| Layer | Tools |
|---|---|
| ML | LightGBM, scikit-learn, SHAP |
| NLP | Anthropic Claude (`claude-sonnet-4-6`) |
| Weather | Open-Meteo API, meteostat |
| App | Streamlit, Plotly |
| Data | BTS On-Time Reporting (~987K flights, 2022–2025) |

## Author

**Tavishi Sharma**

[LinkedIn](https://www.linkedin.com/in/tavishi-sharma05/) · [GitHub](https://github.com/Sharma-Tavishi) · [Website](https://sharma-tavishi.github.io)
