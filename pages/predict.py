import os
import re
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import anthropic
import requests
from datetime import datetime, date
from meteostat import Stations, Hourly
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# converts **bold** markdown to html — needed since responses go into raw html divs
def md_to_html(text: str) -> str:
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text, flags=re.DOTALL)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text, flags=re.DOTALL)
    text = text.replace('\n', '<br>')
    return text

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.nav import render_nav
from utils.constants import AIRLINE_NAMES, AIRLINE_CODES, airline_label, MODEL_PATHS
render_nav("pages/predict.py")


@st.cache_resource
def load_models():
    clf        = joblib.load(MODEL_PATHS["classifier"])
    reg        = joblib.load(MODEL_PATHS["regressor"])
    enc        = joblib.load(MODEL_PATHS["encoder"])
    top_orig   = joblib.load(MODEL_PATHS["top_orig"])
    top_dest   = joblib.load(MODEL_PATHS["top_dest"])
    route_data = joblib.load(MODEL_PATHS["route_data"])
    pre_info   = joblib.load(MODEL_PATHS["preprocessor"])
    return clf, reg, enc, top_orig, top_dest, route_data, pre_info

clf, reg, enc, top_orig, top_dest, route_data, pre_info = load_models()

CAT_COLS     = pre_info["cat_cols"]
NUM_COLS     = pre_info["num_cols"]
NUM_MEDIANS  = pre_info["num_medians"]
FEATURE_COLS = pre_info["feature_cols"]
ROUTE_AVG    = route_data["route_avg"]
GLOBAL_AVG   = route_data["global_avg"]
API_KEY      = os.getenv("ANTHROPIC_API_KEY", "")
AVIATION_KEY = os.getenv("AVIATIONSTACK_API_KEY", "").strip()

CLASS_LABELS = {0: "On-time", 1: "Minor Delay (16–59 min)", 2: "Major Delay (60+ min)"}
CLASS_COLORS = {0: "#16a34a", 1: "#d97706", 2: "#dc2626"}
CLASS_BG     = {0: "#f0fdf4", 1: "#fffbeb", 2: "#fef2f2"}
CLASS_BORDER = {0: "#86efac", 1: "#fcd34d", 2: "#fca5a5"}



@st.cache_data(ttl=3600, show_spinner=False)
def get_weather(iata: str, lat: float, lon: float, flight_date: str, hour: int):
    try:
        dt       = datetime.strptime(flight_date, "%Y-%m-%d")
        stations = Stations().nearby(lat, lon).fetch(1)
        if stations.empty:
            return {}
        wx = Hourly(stations.index[0], dt, dt).fetch()
        if wx.empty:
            return {}
        row = wx.iloc[min(hour, len(wx) - 1)]
        return {
            "temp": float(row.get("temp", np.nan)),
            "wspd": float(row.get("wspd", np.nan)),
            "prcp": float(row.get("prcp", np.nan)),
            "snow": float(row.get("snow", np.nan)),
            "coco": float(row.get("coco", np.nan)),
        }
    except Exception:
        return {}


@st.cache_data
def load_airport_coords():
    df = pd.read_csv("Data/airports.csv")
    df.columns = [c.lower() for c in df.columns]
    df["iata"] = df["iata"].astype(str).str.strip().str.upper()
    df = df.drop_duplicates(subset="iata", keep="first")
    return df.set_index("iata")[["lat", "lon"]].to_dict("index")

airport_coords = load_airport_coords()


def haversine_miles(iata1: str, iata2: str) -> float:
    if iata1 not in airport_coords or iata2 not in airport_coords:
        return 800.0
    import math
    lat1, lon1 = math.radians(airport_coords[iata1]["lat"]), math.radians(airport_coords[iata1]["lon"])
    lat2, lon2 = math.radians(airport_coords[iata2]["lat"]), math.radians(airport_coords[iata2]["lon"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return round(3958.8 * 2 * math.asin(math.sqrt(a)), 1)


def is_us_airport(iata: str) -> bool:
    if iata not in airport_coords:
        return True  # unknown, let it through
    lat = airport_coords[iata]["lat"]
    lon = airport_coords[iata]["lon"]
    continental = (24 <= lat <= 50) and (-125 <= lon <= -66)
    alaska      = (54 <= lat <= 72) and (-180 <= lon <= -130)
    hawaii      = (18 <= lat <= 23) and (-162 <= lon <= -154)
    return continental or alaska or hawaii


def predict(origin, dest, carrier, dep_hour, arr_hour,
            month, dayofweek, distance, flight_date) -> dict:

    o = origin if origin in top_orig else "OTHER"
    d = dest   if dest   in top_dest  else "OTHER"

    match           = ROUTE_AVG[(ROUTE_AVG.origin_topK == o) & (ROUTE_AVG.dest_topK == d)]
    route_avg_delay = float(match["route_avg_delay"].iloc[0]) if len(match) else GLOBAL_AVG

    wx = {}
    if origin in airport_coords:
        coords = airport_coords[origin]
        wx = get_weather(origin, coords["lat"], coords["lon"], flight_date, dep_hour)

    row = {
        "dep_hour":        dep_hour,
        "arr_hour":        arr_hour,
        "month":           month,
        "dayofweek":       dayofweek,
        "distance":        distance,
        "temp":            wx.get("temp", NUM_MEDIANS.get("temp", 15.0)),
        "wspd":            wx.get("wspd", NUM_MEDIANS.get("wspd", 10.0)),
        "prcp":            wx.get("prcp", NUM_MEDIANS.get("prcp", 0.0)),
        "snow":            wx.get("snow", NUM_MEDIANS.get("snow", 0.0)),
        "coco":            wx.get("coco", NUM_MEDIANS.get("coco", 1.0)),
        "origin_topK":     o,
        "dest_topK":       d,
        "carrier":         carrier,
        "route_avg_delay": route_avg_delay,
    }

    df_row = pd.DataFrame([row])
    for col in NUM_COLS:
        df_row[col] = pd.to_numeric(df_row[col], errors="coerce").fillna(NUM_MEDIANS.get(col, 0))
    df_row[CAT_COLS] = enc.transform(df_row[CAT_COLS]).astype(int)
    for col in CAT_COLS:
        df_row[col] = pd.Categorical(df_row[col])
    df_row = df_row[FEATURE_COLS]

    cls     = int(clf.predict(df_row)[0])
    minutes = float(reg.predict(df_row)[0])
    probs   = clf.predict_proba(df_row)[0]

    return {
        "class":     cls,
        "label":     CLASS_LABELS[cls],
        "color":     CLASS_COLORS[cls],
        "bg":        CLASS_BG[cls],
        "border":    CLASS_BORDER[cls],
        "minutes":   minutes,
        "probs":     probs.tolist(),
        "weather":   wx,
        "route_avg": route_avg_delay,
    }


@st.cache_data(ttl=300, show_spinner=False)
def lookup_flight(flight_iata: str) -> dict:
    if not AVIATION_KEY:
        return {}
    try:
        resp = requests.get(
            "http://api.aviationstack.com/v1/flights",
            params={"access_key": AVIATION_KEY, "flight_iata": flight_iata.upper()},
            timeout=8,
        )
        data = resp.json().get("data", [])
        if not data:
            return {}
        flight = data[0]
        dep    = flight.get("departure", {})
        arr    = flight.get("arrival",   {})
        al     = flight.get("airline",   {})

        scheduled = dep.get("scheduled") or dep.get("estimated") or ""
        dep_hour  = None
        fl_date   = None
        if scheduled:
            try:
                dt       = datetime.fromisoformat(scheduled.replace("Z", "+00:00"))
                dep_hour = dt.hour
                fl_date  = dt.strftime("%Y-%m-%d")
            except Exception:
                pass

        return {
            "origin":      dep.get("iata") or None,
            "dest":        arr.get("iata") or None,
            "carrier":     al.get("iata")  or None,
            "dep_hour":    dep_hour,
            "flight_date": fl_date,
        }
    except Exception:
        return {}


def parse_with_claude(user_message: str) -> dict:
    client = anthropic.Anthropic(api_key=API_KEY)
    today_str = date.today().strftime("%Y-%m-%d")
    system = f"""Today's date is {today_str}. You are a flight information parser. Extract flight details from the user's message.
Return ONLY a valid JSON object with these exact keys (use null if not mentioned):
{{
  "origin": "3-letter IATA airport code or null",
  "dest": "3-letter IATA airport code or null",
  "carrier": "2-letter airline IATA code or null",
  "dep_hour": integer 0-23 or null,
  "flight_date": "YYYY-MM-DD or null",
  "distance": number in miles or null,
  "flight_number": "full IATA flight code e.g. WN3739 or null"
}}
Date rules (use today {today_str} as reference):
- "tomorrow" = next day from today
- "today" = today's date
- "next Friday" = the coming Friday
- "in January" or "this January" = the 15th of the next upcoming January (if January has already passed this year, use next year)
- If no date is mentioned, use null — do NOT default to today.
Airline codes: AA=American, DL=Delta, UA=United, WN=Southwest, B6=JetBlue, AS=Alaska, NK=Spirit, F9=Frontier, HA=Hawaiian, G4=Allegiant.
City to airport (US): New York=JFK, Chicago=ORD, LA=LAX, Dallas=DFW, Atlanta=ATL, Denver=DEN, Seattle=SEA, Miami=MIA, Boston=BOS, San Francisco=SFO.
International cities (extract the code but these are NOT US airports): London=LHR, Paris=CDG, Toronto=YYZ, Vancouver=YVR, Mexico City=MEX, Cancun=CUN.
Return ONLY the JSON object."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    text  = response.content[0].text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group()) if match else {}


def generate_response_with_claude(user_message: str, prediction: dict, flight_info: dict) -> str:
    client = anthropic.Anthropic(api_key=API_KEY)
    wx     = prediction["weather"]
    wx_parts = []
    if wx:
        if not np.isnan(wx.get("temp", float("nan"))):
            wx_parts.append(f"temperature {wx['temp']:.0f}°C")
        if not np.isnan(wx.get("wspd", float("nan"))):
            wx_parts.append(f"wind {wx['wspd']:.0f} km/h")
        if wx.get("prcp", 0) > 0:
            wx_parts.append(f"precipitation {wx['prcp']:.1f}mm")
        if wx.get("snow", 0) > 0:
            wx_parts.append(f"snow {wx['snow']:.0f}mm")
    wx_desc = ("Weather at origin: " + ", ".join(wx_parts) + ".") if wx_parts else ""

    system = f"""You are a helpful flight delay assistant. Be conversational and concise.
Prediction summary:
- Route: {flight_info.get('origin','?')} to {flight_info.get('dest','?')} ({flight_info.get('carrier','?')})
- Result: {prediction['label']}
- Estimated delay: {prediction['minutes']:.0f} minutes
- On-time chance: {prediction['probs'][0]*100:.0f}%
- Route historical average: {prediction['route_avg']:.1f} min
{wx_desc}

Write 2-3 friendly sentences explaining the result. Mention weather or route history if relevant.
For on-time flights be reassuring. For delays suggest practical tips (check the airline app, arrive early).
Do not mention ML, models, or probabilities. Speak naturally.
Do NOT use any markdown formatting — no asterisks, no bold, no italic, no bullet points. Plain text only."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text.strip()


def render_result(result, response_text=None):
    delay_str = (f"+{result['minutes']:.0f} min late"
                 if result["minutes"] > 0
                 else f"{abs(result['minutes']):.0f} min early")

    st.markdown(f"""
    <div style="background:{result['bg']};border-left:5px solid {result['color']};
                border-radius:12px;padding:1.2rem 1.5rem;margin:1rem 0;">
        <div style="font-size:1.3rem;font-weight:700;color:{result['color']};">
            {result['label']}
        </div>
        <div style="color:#374151;margin-top:0.3rem;">Estimated: {delay_str}</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    for col, label, prob in zip(
        [col1, col2, col3],
        ["On-time", "Minor delay", "Major delay"],
        result["probs"]
    ):
        col.metric(label, f"{prob*100:.0f}%")

    if response_text:
        st.markdown(f"""
        <div style="border-radius:10px;padding:1rem 1.2rem;
                    margin-top:0.8rem;border:1px solid #e9d5ff;line-height:1.6;">
            {md_to_html(response_text)}
        </div>
        """, unsafe_allow_html=True)

    wx = result.get("weather", {})
    if any(not np.isnan(wx.get(k, float("nan"))) for k in ["temp","wspd"]):
        with st.expander("Weather at origin airport"):
            wc1, wc2, wc3 = st.columns(3)
            if not np.isnan(wx.get("temp", float("nan"))):
                wc1.metric("Temperature", f"{wx['temp']:.0f}°C")
            if not np.isnan(wx.get("wspd", float("nan"))):
                wc2.metric("Wind speed", f"{wx['wspd']:.0f} km/h")
            if wx.get("prcp", 0) > 0:
                wc3.metric("Precipitation", f"{wx['prcp']:.1f} mm")


st.markdown("<h1 style='margin-bottom:0'>Flight Delay Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='margin-top:0.2rem;'>Ask about any US domestic flight</p>",
            unsafe_allow_html=True)
st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.08);margin:0.8rem 0 1rem 0'>", unsafe_allow_html=True)

tab_chat, tab_manual = st.tabs(["Natural Language", "Manual Input"])

with tab_chat:
    st.markdown("Type your question in plain English and the system will extract the flight details automatically.")

    examples = [
        "Will my Delta flight from Atlanta to JFK tomorrow morning be delayed?",
        "Southwest flight from Denver to Las Vegas Friday afternoon, on time?",
        "AA flight from Chicago to Miami at 7pm tonight, any delays?",
    ]
    examples_html = "".join(
        f"<div style='font-size:0.85rem;color:#9ca3af;margin-bottom:0.2rem;'>• {ex}</div>"
        for ex in examples
    )
    st.markdown(
        f"<div style='margin-bottom:0.8rem;'>"
        f"<div style='font-size:0.85rem;margin-bottom:0.3rem;'>Try an example:</div>"
        f"{examples_html}</div>",
        unsafe_allow_html=True
    )

    user_input = st.text_input(
        "Your question",
        value=st.session_state.get("chat_input", ""),
        placeholder="e.g. Will my AA flight from JFK to LAX tomorrow morning be delayed?",
        key="chat_text",
        label_visibility="collapsed",
    )

    if st.button("Check flight", type="primary", key="chat_predict"):
        if not user_input.strip():
            st.warning("Please enter a question.")
        else:
            _warn = None
            _err  = None
            _res  = None
            _resp = None
            _info = None

            with st.spinner("Analyzing your flight..."):
                try:
                    parsed = parse_with_claude(user_input)
                    flight_num = parsed.get("flight_number")

                    if (not parsed.get("origin") and not parsed.get("dest")
                            and not parsed.get("carrier") and parsed.get("dep_hour") is None
                            and not flight_num):
                        _warn = ("This app only answers **flight delay questions**. "
                                 "Try: *\"Will my Delta flight from ATL to JFK tomorrow at 8am be delayed?\"*")

                    if not _warn and flight_num:
                        if AVIATION_KEY:
                            looked_up = lookup_flight(flight_num)
                            if looked_up:
                                for key in ("origin", "dest", "carrier", "dep_hour", "flight_date"):
                                    if not parsed.get(key) and looked_up.get(key) is not None:
                                        parsed[key] = looked_up[key]
                            else:
                                _warn = (f"Couldn't find flight **{flight_num}** in the live database. "
                                         "It may not be operating today. Please provide the route details: "
                                         "origin, destination, airline, and departure time.")
                        else:
                            _warn = "Flight number lookup is not configured. Please provide origin, destination, airline, and departure time."

                    if not _warn:
                        for code in filter(None, [parsed.get("origin"), parsed.get("dest")]):
                            if not is_us_airport(code.upper()):
                                _warn = (f"**{code.upper()}** is an international airport. "
                                         "This model is trained on **US domestic flights only** "
                                         "(continental US, Alaska, and Hawaii).")
                                break

                    if not _warn:
                        missing = []
                        if not parsed.get("origin"):
                            missing.append("**origin airport** (e.g. ORD, LAX, JFK)")
                        if not parsed.get("dest"):
                            missing.append("**destination airport** (e.g. ATL, DFW, MIA)")
                        if not parsed.get("carrier"):
                            missing.append("**airline** (e.g. Southwest, Delta, United)")
                        if parsed.get("dep_hour") is None:
                            missing.append("**departure time** (e.g. 8am, 3pm)")
                        if missing:
                            _warn = ("Missing info: " + ", ".join(missing) + ". "
                                     "Please include these details and try again. "
                                     "Example: *\"Will my Southwest flight from ORD to ATL at 3pm be delayed?\"*")

                    if not _warn:
                        origin   = parsed["origin"].upper()
                        dest     = parsed["dest"].upper()
                        carrier  = parsed["carrier"].upper()
                        dep_hour = int(parsed["dep_hour"])
                        arr_hour = int((dep_hour + max(1, round(distance / 500))) % 24)

                        raw_date = parsed.get("flight_date")
                        try:
                            flight_date = str(datetime.strptime(raw_date, "%Y-%m-%d").date()) if raw_date else str(date.today())
                        except Exception:
                            flight_date = str(date.today())

                        dt        = datetime.strptime(flight_date, "%Y-%m-%d")
                        month     = dt.month
                        dayofweek = dt.isoweekday()
                        distance  = float(parsed.get("distance") or haversine_miles(origin, dest))

                        _info = {"origin": origin, "dest": dest, "carrier": carrier,
                                 "dep_hour": dep_hour, "flight_date": flight_date}
                        _res  = predict(origin, dest, carrier, dep_hour, arr_hour,
                                        month, dayofweek, distance, flight_date)
                        _resp = generate_response_with_claude(user_input, _res, _info)

                except anthropic.AuthenticationError:
                    _err = "Invalid API key. Please check your Anthropic API key in the .env file."
                except anthropic.APIConnectionError:
                    _err = "Could not reach the AI service. Please check your internet connection and try again."
                except anthropic.RateLimitError:
                    _err = "Too many requests. Please wait a moment and try again."
                except Exception as e:
                    _err = "Something went wrong. Please try again."

            if _warn:
                st.error(_warn)
            elif _err:
                st.error(_err)
            elif _res:
                st.divider()
                render_result(_res, _resp)
                with st.expander("Parsed flight details"):
                    st.json({"origin": _info["origin"], "dest": _info["dest"],
                             "carrier": _info["carrier"], "departure_hour": _info["dep_hour"],
                             "flight_date": _info["flight_date"]})

with tab_manual:
    st.markdown("Enter flight details directly, or provide a flight number to auto-fill.")

    fn_col, btn_col, _ = st.columns([2, 1, 3])
    with fn_col:
        flight_num_m = st.text_input(
            "Flight number (optional)",
            placeholder="e.g. AA100, DL400",
            key="manual_flight_num"
        ).strip().upper().replace(" ", "")
    with btn_col:
        st.markdown("<div style='margin-top:1.75rem'></div>", unsafe_allow_html=True)
        autofill_btn = st.button("Auto-fill", key="manual_autofill")

    if autofill_btn:
        if not flight_num_m:
            st.warning("Enter a flight number first.")
        elif not AVIATION_KEY:
            st.warning("Flight lookup not configured. Fill in the fields manually.")
        else:
            with st.spinner(f"Looking up {flight_num_m}..."):
                lk = lookup_flight(flight_num_m)
            if lk:
                if lk.get("origin"):               st.session_state["mi_origin"]  = lk["origin"]
                if lk.get("dest"):                 st.session_state["mi_dest"]    = lk["dest"]
                if lk.get("carrier") and lk["carrier"] in AIRLINE_CODES:
                                                   st.session_state["mi_carrier"] = lk["carrier"]
                if lk.get("dep_hour") is not None: st.session_state["mi_hour"]   = lk["dep_hour"]
                if lk.get("flight_date"):
                    st.session_state["mi_date"] = datetime.strptime(lk["flight_date"], "%Y-%m-%d").date()
                st.session_state["_m_lookup_ok"]   = flight_num_m
                st.session_state["_m_lookup_fail"] = False
                st.rerun()
            else:
                st.session_state["_m_lookup_ok"]   = False
                st.session_state["_m_lookup_fail"] = flight_num_m
                st.rerun()

    if st.session_state.get("_m_lookup_ok"):
        fn_ = st.session_state["_m_lookup_ok"]
        st.success(
            f"Flight **{fn_}** found: "
            f"{st.session_state.get('mi_origin','?')} → {st.session_state.get('mi_dest','?')}, "
            f"{airline_label(st.session_state.get('mi_carrier','?'))}, "
            f"{st.session_state.get('mi_hour', 0):02d}:00. "
            "Fields updated — adjust if needed, then click Check flight."
        )
    elif st.session_state.get("_m_lookup_fail"):
        fn_ = st.session_state["_m_lookup_fail"]
        st.error(f"Couldn't find flight **{fn_}** in the live database. Fill in the fields manually.")

    for _k, _v in [("mi_origin","ORD"),("mi_dest","JFK"),
                    ("mi_carrier","AA"),("mi_hour",8),("mi_date",date.today())]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    col1, col2, col3 = st.columns(3)
    with col1:
        origin_m = st.text_input("Origin",      max_chars=3, key="mi_origin").upper()
        dest_m   = st.text_input("Destination", max_chars=3, key="mi_dest").upper()
    with col2:
        carrier_m     = st.selectbox("Airline", options=AIRLINE_CODES,
                                     format_func=airline_label, key="mi_carrier")
        flight_date_m = st.date_input("Date", key="mi_date")
    with col3:
        dep_hour_m = st.slider("Departure hour", 0, 23, key="mi_hour", format="%d:00")

    if st.button("Check flight", type="primary", key="manual_predict"):
        _merr  = None
        _mres  = None
        _mresp = None

        with st.spinner("Running prediction..."):
            try:
                origin_f   = origin_m.upper()
                dest_f     = dest_m.upper()
                carrier_f  = carrier_m.upper()
                dep_hour_f = dep_hour_m
                date_f     = str(flight_date_m)

                if not is_us_airport(origin_f) or not is_us_airport(dest_f):
                    intl = " and ".join([a for a in [origin_f, dest_f] if not is_us_airport(a)])
                    _merr = (f"{intl} is an international airport. "
                             "This model is trained on US domestic flights only.")
                else:
                    dt_f       = datetime.strptime(date_f, "%Y-%m-%d")
                    arr_hour_f = (dep_hour_f + 2) % 24
                    _mres = predict(origin_f, dest_f, carrier_f, dep_hour_f, arr_hour_f,
                                    dt_f.month, dt_f.isoweekday(),
                                    haversine_miles(origin_f, dest_f), date_f)
                    if API_KEY:
                        user_q_m = f"{airline_label(carrier_f)} flight from {origin_f} to {dest_f} at {dep_hour_f}:00 on {date_f}"
                        _mresp = generate_response_with_claude(
                            user_q_m, _mres,
                            {"origin": origin_f, "dest": dest_f, "carrier": carrier_f,
                             "dep_hour": dep_hour_f, "flight_date": date_f}
                        )

            except Exception as e:
                _merr = str(e)

        if _merr:
            st.error(_merr)
        elif _mres:
            st.divider()
            render_result(_mres, _mresp)
            st.metric("Route historical average", f"{_mres['route_avg']:.1f} min")
