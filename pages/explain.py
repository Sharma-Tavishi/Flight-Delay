import math
import json as _json
import urllib.request
import warnings
import calendar
from datetime import date
import joblib
import shap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.nav import render_nav
from utils.constants import AIRLINE_NAMES, MODEL_PATHS
render_nav("pages/explain.py")


@st.cache_resource
def load_artifacts():
    clf        = joblib.load(MODEL_PATHS["classifier"])
    reg        = joblib.load(MODEL_PATHS["regressor"])
    enc        = joblib.load(MODEL_PATHS["encoder"])
    top_orig   = joblib.load(MODEL_PATHS["top_orig"])
    top_dest   = joblib.load(MODEL_PATHS["top_dest"])
    route_data = joblib.load(MODEL_PATHS["route_data"])
    pre_info   = joblib.load(MODEL_PATHS["preprocessor"])
    return clf, reg, enc, top_orig, top_dest, route_data, pre_info

clf, reg, enc, top_orig, top_dest, route_data, pre_info = load_artifacts()

CAT_COLS     = pre_info["cat_cols"]
NUM_COLS     = pre_info["num_cols"]
NUM_MEDIANS  = pre_info["num_medians"]
FEATURE_COLS = pre_info["feature_cols"]
ROUTE_AVG    = route_data["route_avg"]
GLOBAL_AVG   = route_data["global_avg"]


@st.cache_data
def load_airport_coords():
    df = pd.read_csv("Data/airports.csv")
    df.columns = [c.lower() for c in df.columns]
    df["iata"] = df["iata"].astype(str).str.strip().str.upper()
    return df.drop_duplicates("iata").set_index("iata")[["lat","lon"]].to_dict("index")

airport_coords = load_airport_coords()


@st.cache_data(ttl=86400)
def get_historical_weather(iata: str, month: int):
    coords = airport_coords.get(iata)
    if not coords:
        return None, None
    lat, lon = coords["lat"], coords["lon"]
    try:
        year = date.today().year - 1
        last_day = calendar.monthrange(year, month)[1]
        url = (f"https://archive-api.open-meteo.com/v1/archive"
               f"?latitude={lat}&longitude={lon}"
               f"&start_date={year}-{month:02d}-01&end_date={year}-{month:02d}-{last_day:02d}"
               f"&daily=temperature_2m_mean,windspeed_10m_mean&timezone=auto")
        with urllib.request.urlopen(url, timeout=10) as r:
            data = _json.loads(r.read())
        daily = data.get("daily", {})
        temps = [t for t in daily.get("temperature_2m_mean", []) if t is not None]
        winds = [w for w in daily.get("windspeed_10m_mean", []) if w is not None]
        avg_temp = int(round(sum(temps) / len(temps))) if temps else None
        avg_wind = int(round(sum(winds) / len(winds))) if winds else None
        return avg_temp, avg_wind
    except Exception:
        return None, None


if "_explain_temp" not in st.session_state:
    st.session_state["_explain_temp"] = 15
if "_explain_wspd" not in st.session_state:
    st.session_state["_explain_wspd"] = 10
if "_explain_orig_temp" not in st.session_state:
    st.session_state["_explain_orig_temp"] = 15
if "_explain_orig_wspd" not in st.session_state:
    st.session_state["_explain_orig_wspd"] = 10


def haversine_miles(iata1: str, iata2: str) -> float:
    if iata1 not in airport_coords or iata2 not in airport_coords:
        return NUM_MEDIANS.get("distance", 800.0)
    lat1, lon1 = math.radians(airport_coords[iata1]["lat"]), math.radians(airport_coords[iata1]["lon"])
    lat2, lon2 = math.radians(airport_coords[iata2]["lat"]), math.radians(airport_coords[iata2]["lon"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return round(3958.8 * 2 * math.asin(math.sqrt(a)), 1)


FEATURE_LABELS = {
    "dep_hour":        "Departure Hour",
    "arr_hour":        "Arrival Hour",
    "month":           "Month",
    "dayofweek":       "Day of Week",
    "distance":        "Distance (miles)",
    "temp":            "Temperature (°C)",
    "wspd":            "Wind Speed (km/h)",
    "prcp":            "Precipitation (mm)",
    "snow":            "Snowfall (mm)",
    "coco":            "Weather Condition",
    "origin_topK":     "Origin Airport",
    "dest_topK":       "Destination Airport",
    "carrier":         "Airline",
    "route_avg_delay": "Route Avg Delay (min)",
}


@st.cache_resource
def get_explainer():
    return shap.TreeExplainer(clf)

explainer = get_explainer()


def build_row(origin, dest, carrier, dep_hour, arr_hour, month, dayofweek,
              distance, temp, wspd, prcp, snow, coco):
    o = origin  if origin  in top_orig else "OTHER"
    d = dest    if dest    in top_dest  else "OTHER"
    match           = ROUTE_AVG[(ROUTE_AVG.origin_topK == o) & (ROUTE_AVG.dest_topK == d)]
    route_avg_delay = float(match["route_avg_delay"].iloc[0]) if len(match) else GLOBAL_AVG

    row = {
        "dep_hour": dep_hour, "arr_hour": arr_hour, "month": month,
        "dayofweek": dayofweek, "distance": distance,
        "temp": temp, "wspd": wspd, "prcp": prcp, "snow": snow, "coco": coco,
        "origin_topK": o, "dest_topK": d, "carrier": carrier,
        "route_avg_delay": route_avg_delay,
    }
    df_row = pd.DataFrame([row])
    for col in NUM_COLS:
        df_row[col] = pd.to_numeric(df_row[col], errors="coerce").fillna(NUM_MEDIANS.get(col, 0))
    df_row[CAT_COLS] = enc.transform(df_row[CAT_COLS]).astype(int)
    for col in CAT_COLS:
        df_row[col] = pd.Categorical(df_row[col])
    return df_row[FEATURE_COLS], route_avg_delay


def waterfall_chart(shap_vals, feature_vals, base_value, predicted_class):
    labels  = [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS]
    idx     = np.argsort(np.abs(shap_vals))[::-1][:10]
    s_vals  = shap_vals[idx]
    f_names = [f"{labels[i]} = {feature_vals[i]}" for i in idx]
    colors  = ["#dc2626" if v > 0 else "#16a34a" for v in s_vals]

    fig = go.Figure(go.Bar(
        x=s_vals,
        y=f_names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in s_vals],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Why this prediction? (class: {['On-time','Minor Delay','Major Delay'][predicted_class]})",
        xaxis_title="SHAP value (pushes toward delay →  or on-time ←)",
        yaxis=dict(autorange="reversed"),
        height=420,
        margin=dict(l=20, r=80, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e5e7eb",
    )
    return fig


st.markdown("<h1 style='margin-bottom:0'>Explainability Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='margin-top:0.2rem;'>Understand why the model makes each prediction using SHAP values.</p>",
            unsafe_allow_html=True)
st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.08);margin:0.8rem 0 1rem 0'>", unsafe_allow_html=True)

tab_local, = st.tabs(["Single Flight Explanation"])

with tab_local:
    st.markdown("Enter a flight below to see which factors drove the prediction.")

    col1, col2, col3 = st.columns(3)
    with col1:
        origin_e  = st.text_input("Origin",      value="ORD", max_chars=3).upper()
        dest_e    = st.text_input("Destination",  value="JFK", max_chars=3).upper()
        carrier_e = st.selectbox(
            "Airline",
            options=sorted(AIRLINE_NAMES.keys()),
            format_func=lambda x: f"{AIRLINE_NAMES.get(x, x)} ({x})"
        )

    with col2:
        dep_hour_e  = st.slider("Departure hour", 0, 23, 17, format="%d:00")
        month_e     = st.selectbox("Month", options=list(range(1, 13)),
                          format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun",
                                                  "Jul","Aug","Sep","Oct","Nov","Dec"][x-1])
        dayofweek_e = st.selectbox("Day of week",
            options=[1,2,3,4,5,6,7],
            format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x-1])
    with col3:
        pass

    # re-fetch historical weather when origin, destination, or month changes
    origin_changed = st.session_state.get("_explain_prev_origin") != origin_e
    dest_changed   = st.session_state.get("_explain_prev_dest")   != dest_e
    month_changed  = st.session_state.get("_explain_prev_month")  != month_e

    if origin_changed or month_changed:
        if len(origin_e) == 3 and origin_e in airport_coords:
            ot, ow = get_historical_weather(origin_e, month_e)
            if ot is not None:
                st.session_state["_explain_orig_temp"] = ot
                st.session_state["_explain_orig_wspd"] = ow
        st.session_state["_explain_prev_origin"] = origin_e

    if dest_changed or month_changed:
        if len(dest_e) == 3 and dest_e in airport_coords:
            wt, ww = get_historical_weather(dest_e, month_e)
            if wt is not None:
                st.session_state["_explain_temp"] = wt
                st.session_state["_explain_wspd"] = ww
        st.session_state["_explain_prev_dest"] = dest_e

    if month_changed:
        st.session_state["_explain_prev_month"] = month_e

    MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    wcol1, wcol2 = st.columns(2)
    with wcol1:
        st.markdown("<small style='color:#9ca3af'>Origin weather</small>", unsafe_allow_html=True)
        orig_temp_e = st.number_input("Temp (°C)",  -30, 45, key="_explain_orig_temp")
        orig_wspd_e = st.number_input("Wind (km/h)",  0, 150, key="_explain_orig_wspd")
        if origin_e in airport_coords:
            st.caption(f"📍 Avg weather for {MONTH_NAMES[month_e-1]} at {origin_e}")
    with wcol2:
        st.markdown("<small style='color:#9ca3af'>Destination weather</small>", unsafe_allow_html=True)
        temp_e = st.number_input("Temp (°C) ",  -30, 45, key="_explain_temp")
        wspd_e = st.number_input("Wind (km/h) ",  0, 150, key="_explain_wspd")
        if dest_e in airport_coords:
            st.caption(f"📍 Avg weather for {MONTH_NAMES[month_e-1]} at {dest_e}")

    if st.button("Explain this flight", type="primary"):
        with st.spinner("Computing SHAP values..."):
            arr_hour_e = (dep_hour_e + 2) % 24
            df_row, route_avg = build_row(
                origin_e, dest_e, carrier_e, dep_hour_e, arr_hour_e,
                month_e, dayofweek_e, haversine_miles(origin_e, dest_e),
                float(temp_e), float(wspd_e), 0.0, 0.0, 1.0
            )

            pred_class = int(clf.predict(df_row)[0])
            probs      = clf.predict_proba(df_row)[0]
            shap_vals  = explainer.shap_values(df_row)

            if isinstance(shap_vals, list):
                sv = np.array(shap_vals)[:, 0, :]
            else:
                sv = shap_vals[0].T

            base_vals = explainer.expected_value
            if not hasattr(base_vals, "__len__"):
                base_vals = [base_vals] * 3

            CLASS_COLORS = {0: "#16a34a", 1: "#d97706", 2: "#dc2626"}
            CLASS_LABELS = {0: "On-time", 1: "Minor Delay (16–59 min)", 2: "Major Delay (60+ min)"}

            st.markdown(f"""
            <div style="border-left:5px solid {CLASS_COLORS[pred_class]};
                        border-radius:8px;padding:0.8rem 1.2rem;margin-bottom:1rem;">
                <strong style="color:{CLASS_COLORS[pred_class]};font-size:1.1rem;">
                    {CLASS_LABELS[pred_class]}
                </strong>
                &nbsp;&nbsp;On-time {probs[0]*100:.0f}% &nbsp;|&nbsp;
                Minor {probs[1]*100:.0f}% &nbsp;|&nbsp;
                Major {probs[2]*100:.0f}%
            </div>
            """, unsafe_allow_html=True)

            raw_vals = df_row.iloc[0].values
            fig = waterfall_chart(sv[pred_class], raw_vals, base_vals[pred_class], pred_class)
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "🔴 Red bars push toward delay. 🟢 Green bars push toward on-time. "
                "Longer bar = stronger influence."
            )

            with st.expander("All classes SHAP breakdown"):
                cols = st.columns(3)
                for i, (label, color) in enumerate(zip(
                    ["On-time","Minor Delay","Major Delay"],
                    ["#16a34a","#d97706","#dc2626"]
                )):
                    top5_idx = np.argsort(np.abs(sv[i]))[::-1][:5]
                    items = "".join([
                        f"<li>{FEATURE_LABELS.get(FEATURE_COLS[j], FEATURE_COLS[j])}: "
                        f"<strong>{sv[i][j]:+.3f}</strong></li>"
                        for j in top5_idx
                    ])
                    cols[i].markdown(
                        f"<div style='color:{color};font-weight:700'>{label}</div>"
                        f"<ul style='font-size:0.85rem'>{items}</ul>",
                        unsafe_allow_html=True
                    )
