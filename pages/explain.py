import math
import json as _json
import urllib.request
import warnings
import calendar
import requests
from datetime import date, datetime
import joblib
import shap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.nav import render_nav, get_theme
from utils.constants import AIRLINE_NAMES, MODEL_PATHS, AIRPORT_NAMES, airport_label

AVIATION_KEY = os.getenv("AERODATABOX_API_KEY", "").strip()


@st.cache_data(ttl=300, show_spinner=False)
def lookup_flight_explain(flight_iata: str, flight_date: str | None = None) -> dict:
    if not AVIATION_KEY:
        return {}
    try:
        date_str = flight_date or date.today().strftime("%Y-%m-%d")
        url = f"https://aerodatabox.p.rapidapi.com/flights/number/{flight_iata.upper()}/{date_str}"
        resp = requests.get(
            url,
            headers={
                "x-rapidapi-key":  AVIATION_KEY,
                "x-rapidapi-host": "aerodatabox.p.rapidapi.com",
            },
            timeout=8,
        )
        data = resp.json()
        if not data or not isinstance(data, list):
            return {}
        flight   = data[0]
        dep      = flight.get("departure", {})
        arr      = flight.get("arrival",   {})
        al       = flight.get("airline",   {})
        sched    = dep.get("scheduledTime") or {}
        revised  = dep.get("revisedTime")  or {}
        scheduled = (sched.get("local") or sched.get("utc")
                     or revised.get("local") or revised.get("utc") or "")
        dep_hour = None
        fl_date  = None
        if scheduled:
            try:
                cleaned  = scheduled.replace("T", " ").split("+")[0].split("Z")[0].strip()[:16]
                dt       = datetime.strptime(cleaned, "%Y-%m-%d %H:%M")
                dep_hour = dt.hour
                fl_date  = dt.strftime("%Y-%m-%d")
            except Exception:
                pass
        return {
            "origin":      (dep.get("airport") or {}).get("iata") or None,
            "dest":        (arr.get("airport") or {}).get("iata") or None,
            "carrier":     al.get("iata") or None,
            "dep_hour":    dep_hour,
            "flight_date": fl_date or date_str,
        }
    except Exception:
        return {}
render_nav("pages/explain.py")
t = get_theme()


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


def waterfall_chart(shap_vals, feature_vals, base_value, predicted_class, font_color="#e5e7eb"):
    labels  = [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS]
    idx     = np.argsort(np.abs(shap_vals))[::-1][:10]
    s_vals  = shap_vals[idx]

    cat_decode = {
        "origin_topK": enc.categories_[CAT_COLS.index("origin_topK")],
        "dest_topK":   enc.categories_[CAT_COLS.index("dest_topK")],
        "carrier":     enc.categories_[CAT_COLS.index("carrier")],
    }
    int_features = {"dep_hour", "arr_hour", "month", "dayofweek", "coco"}

    def fmt(feat, val):
        if feat in cat_decode:
            try:
                return cat_decode[feat][int(val)]
            except (IndexError, ValueError):
                return str(val)
        if feat == "route_avg_delay":
            return f"{float(val):.1f} min"
        if feat == "distance":
            return f"{float(val):.0f} mi"
        if feat in int_features:
            return str(int(val))
        return f"{float(val):.1f}"

    f_names = [f"{labels[i]} = {fmt(FEATURE_COLS[i], feature_vals[i])}" for i in idx]
    colors  = ["#dc2626" if v > 0 else "#16a34a" for v in s_vals]

    fig = go.Figure(go.Bar(
        x=s_vals,
        y=f_names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in s_vals],
        textposition="auto",
    ))
    fig.update_layout(
        title=dict(
            text=f"Why this prediction? (class: {['On-time','Minor Delay','Major Delay'][predicted_class]})",
            font=dict(color=font_color),
        ),
        xaxis=dict(
            title=dict(text="SHAP value (pushes toward delay →  or on-time ←)", font=dict(color=font_color)),
            tickfont=dict(color=font_color),
        ),
        yaxis=dict(autorange="reversed", tickfont=dict(color=font_color)),
        height=420,
        margin=dict(l=20, r=100, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=font_color),
    )
    return fig


st.markdown("<div style='font-size:2.2rem;font-weight:800;line-height:1.2;margin-bottom:0'>Explainability Dashboard</div>", unsafe_allow_html=True)
st.markdown("<p style='margin-top:0.2rem;'>Understand why the model makes each prediction using SHAP values.</p>",
            unsafe_allow_html=True)
st.markdown(f"<hr style='border:none;border-top:1px solid {t['border']};margin:0.8rem 0 1rem 0'>", unsafe_allow_html=True)

tab_local, = st.tabs(["Single Flight Explanation"])

# session state defaults
for k, v in [("ex_origin","ORD"), ("ex_dest","JFK"), ("ex_carrier","AA"),
             ("ex_dep_hour",17), ("ex_month", date.today().month),
             ("ex_dayofweek", date.today().isoweekday())]:
    if k not in st.session_state:
        st.session_state[k] = v

with tab_local:
    st.markdown("Enter a flight below to see which factors drove the prediction.")

    fn_col, btn_col, _ = st.columns([2, 1, 3])
    with fn_col:
        ex_flight_num = st.text_input(
            "Flight number (optional)",
            placeholder="e.g. AA100, DL400",
            key="ex_flight_num_input"
        ).strip().upper().replace(" ", "")
    with btn_col:
        st.markdown("<div style='margin-top:1.75rem'></div>", unsafe_allow_html=True)
        ex_autofill = st.button("Auto-fill", key="ex_autofill_btn")

    if ex_autofill:
        if not ex_flight_num:
            st.warning("Enter a flight number first.")
        elif not AVIATION_KEY:
            st.warning("Flight lookup not configured.")
        else:
            with st.spinner(f"Looking up {ex_flight_num}..."):
                lk = lookup_flight_explain(ex_flight_num)
            if lk:
                if lk.get("origin"):   st.session_state["ex_origin"]   = lk["origin"]
                if lk.get("dest"):     st.session_state["ex_dest"]     = lk["dest"]
                if lk.get("carrier") and lk["carrier"] in AIRLINE_NAMES:
                                       st.session_state["ex_carrier"]  = lk["carrier"]
                if lk.get("dep_hour") is not None:
                                       st.session_state["ex_dep_hour"] = lk["dep_hour"]
                if lk.get("flight_date"):
                    dt = datetime.strptime(lk["flight_date"], "%Y-%m-%d")
                    st.session_state["ex_month"]     = dt.month
                    st.session_state["ex_dayofweek"] = dt.isoweekday()
                st.session_state["_ex_lookup_ok"] = ex_flight_num
                st.rerun()
            else:
                st.error(f"Couldn't find flight **{ex_flight_num}**. Fill in the fields manually.")

    if st.session_state.get("_ex_lookup_ok"):
        fn_ = st.session_state["_ex_lookup_ok"]
        st.success(
            f"Flight **{fn_}** found: "
            f"{st.session_state.get('ex_origin','?')} → {st.session_state.get('ex_dest','?')}, "
            f"{AIRLINE_NAMES.get(st.session_state.get('ex_carrier',''),'?')}, "
            f"{st.session_state.get('ex_dep_hour',0):02d}:00. "
            "Fields updated — adjust if needed, then click Explain."
        )

    airport_options = sorted(AIRPORT_NAMES.keys())

    col1, col2, col3 = st.columns(3)
    with col1:
        origin_e = st.selectbox(
            "Origin",
            options=airport_options,
            index=airport_options.index(st.session_state["ex_origin"])
                  if st.session_state["ex_origin"] in airport_options else 0,
            format_func=airport_label,
            key="ex_origin_sel"
        )
        dest_e = st.selectbox(
            "Destination",
            options=airport_options,
            index=airport_options.index(st.session_state["ex_dest"])
                  if st.session_state["ex_dest"] in airport_options else 0,
            format_func=airport_label,
            key="ex_dest_sel"
        )
        carrier_options = sorted(AIRLINE_NAMES.keys())
        carrier_e = st.selectbox(
            "Airline",
            options=carrier_options,
            index=carrier_options.index(st.session_state["ex_carrier"])
                  if st.session_state["ex_carrier"] in carrier_options else 0,
            format_func=lambda x: f"{AIRLINE_NAMES.get(x, x)} ({x})"
        )

    with col2:
        dep_hour_e  = st.slider("Departure hour", 0, 23, key="ex_dep_hour", format="%d:00")
        month_e     = st.selectbox("Month", options=list(range(1, 13)),
                          index=st.session_state["ex_month"] - 1,
                          format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun",
                                                  "Jul","Aug","Sep","Oct","Nov","Dec"][x-1])
        dayofweek_e = st.selectbox("Day of week",
            options=[1,2,3,4,5,6,7],
            index=st.session_state["ex_dayofweek"] - 1,
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

            CLASS_BG = {0: "#f0fdf4", 1: "#fffbeb", 2: "#fef2f2"}
            st.markdown(f"""
            <div style="background:{CLASS_BG[pred_class]};border-left:5px solid {CLASS_COLORS[pred_class]};
                        border-radius:12px;padding:1.2rem 1.5rem;margin:1rem 0;">
                <div style="font-size:1.3rem;font-weight:700;color:{CLASS_COLORS[pred_class]};">
                    {CLASS_LABELS[pred_class]}
                </div>
            </div>
            <div style="display:flex;gap:1rem;margin:1rem 0;">
              <div style="flex:1;background:rgba(22,163,74,0.12);border:1px solid #16a34a;
                          border-radius:10px;padding:1rem;text-align:center;">
                <div style="color:#16a34a;font-size:2rem;font-weight:800;">{probs[0]*100:.0f}%</div>
                <div style="color:#6b7280;font-size:0.95rem;font-weight:500;margin-top:0.2rem;">On-time</div>
              </div>
              <div style="flex:1;background:rgba(217,119,6,0.12);border:1px solid #d97706;
                          border-radius:10px;padding:1rem;text-align:center;">
                <div style="color:#d97706;font-size:2rem;font-weight:800;">{probs[1]*100:.0f}%</div>
                <div style="color:#6b7280;font-size:0.95rem;font-weight:500;margin-top:0.2rem;">Minor delay</div>
              </div>
              <div style="flex:1;background:rgba(220,38,38,0.12);border:1px solid #dc2626;
                          border-radius:10px;padding:1rem;text-align:center;">
                <div style="color:#dc2626;font-size:2rem;font-weight:800;">{probs[2]*100:.0f}%</div>
                <div style="color:#6b7280;font-size:0.95rem;font-weight:500;margin-top:0.2rem;">Major delay</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            raw_vals = df_row.iloc[0].values
            fig = waterfall_chart(sv[pred_class], raw_vals, base_vals[pred_class], pred_class,
                                  font_color=get_theme()["font_color"])
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
