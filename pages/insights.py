import warnings
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.nav import render_nav
render_nav("pages/insights.py")

AIRLINE_NAMES = {
    "AA": "American", "DL": "Delta", "WN": "Southwest", "UA": "United",
    "AS": "Alaska", "B6": "JetBlue", "NK": "Spirit", "F9": "Frontier",
    "HA": "Hawaiian", "G4": "Allegiant", "OO": "SkyWest", "9E": "Endeavor",
    "MQ": "Envoy", "YX": "Republic", "OH": "PSA", "QX": "Horizon", "YV": "Mesa",
}


@st.cache_data
def load_data():
    insights   = joblib.load("models/insights_stats.joblib")
    df         = pd.read_parquet("Data/processed_flights_sample.parquet")
    airports   = pd.read_csv("Data/airports.csv")
    airports.columns = [c.lower() for c in airports.columns]
    airports["iata"] = airports["iata"].str.strip().str.upper()
    airports = airports.drop_duplicates("iata").set_index("iata")[["lat","lon"]]
    # only continental US — Hawaii/Alaska routes draw diagonally across the albers usa projection
    airports = airports[(airports["lat"].between(24, 50)) & (airports["lon"].between(-125, -66))]
    return insights["airline_stats"], insights["route_stats"], airports, df

airline_stats, route_stats, airports, df = load_data()

st.markdown("<h1 style='margin-bottom:0'>Insights</h1>", unsafe_allow_html=True)
st.markdown("<p style='margin-top:0.2rem;'>Explore patterns in the flight delay dataset.</p>",
            unsafe_allow_html=True)
st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.08);margin:0.8rem 0 1rem 0'>", unsafe_allow_html=True)

tab_airline, tab_map = st.tabs(["Airline Reliability", "Route Map"])

with tab_airline:
    st.markdown("Ranked by on-time performance across **987,000 flights** (2022–2025).")

    grp = airline_stats.rename(columns={0: "on_time", 1: "minor", 2: "major", "avg_delay": "avg_delay_min"})
    grp["carrier_name"] = grp["carrier"].map(lambda x: f"{AIRLINE_NAMES.get(x, x)} ({x})")
    grp = grp.sort_values("on_time", ascending=True).reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="On-time", y=grp["carrier_name"], x=grp["on_time"] * 100,
        orientation="h", marker_color="#16a34a",
        text=[f"{v:.0f}%" for v in grp["on_time"] * 100], textposition="inside",
    ))
    fig.add_trace(go.Bar(
        name="Minor delay", y=grp["carrier_name"], x=grp["minor"] * 100,
        orientation="h", marker_color="#d97706",
        text=[f"{v:.0f}%" for v in grp["minor"] * 100], textposition="inside",
    ))
    fig.add_trace(go.Bar(
        name="Major delay", y=grp["carrier_name"], x=grp["major"] * 100,
        orientation="h", marker_color="#dc2626",
        text=[f"{v:.0f}%" for v in grp["major"] * 100], textposition="inside",
    ))
    fig.update_layout(
        barmode="stack",
        xaxis_title="Percentage of flights (%)",
        height=500,
        margin=dict(l=20, r=40, t=30, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e5e7eb",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    table = grp[["carrier_name","on_time","minor","major","avg_delay_min"]].copy()
    table.columns = ["Airline", "On-time %", "Minor Delay %", "Major Delay %", "Avg Delay (min)"]
    table["On-time %"]      = (table["On-time %"]      * 100).round(1).astype(str) + "%"
    table["Minor Delay %"]  = (table["Minor Delay %"]  * 100).round(1).astype(str) + "%"
    table["Major Delay %"]  = (table["Major Delay %"]  * 100).round(1).astype(str) + "%"
    table["Avg Delay (min)"] = table["Avg Delay (min)"].round(1)
    table = table.sort_values("On-time %", ascending=False).reset_index(drop=True)
    st.dataframe(table, use_container_width=True, hide_index=True)

with tab_map:
    st.markdown("Top routes colored by average delay class. Select a filter below.")

    col1, col2, col3 = st.columns(3)
    with col1:
        carrier_filter = st.selectbox(
            "Filter by airline", ["All"] + sorted(df["carrier"].unique().tolist()),
            format_func=lambda x: f"{AIRLINE_NAMES.get(x, x)} ({x})" if x != "All" else "All Airlines"
        )
    with col2:
        delay_filter = st.selectbox(
            "Show routes by delay type",
            ["All", "On-time", "Minor Delay", "Major Delay"]
        )
    with col3:
        DATASET_MONTHS = 47
        min_per_month  = st.slider("Minimum flights per month on route", 1, 100, 5, step=1)
        min_flights    = int(min_per_month * DATASET_MONTHS)

    if carrier_filter == "All":
        filtered_routes = route_stats.copy()
    else:
        carrier_route_pairs = (
            df[df["carrier"] == carrier_filter][["origin_topK", "dest_topK"]]
            .drop_duplicates()
        )
        filtered_routes = route_stats.merge(
            carrier_route_pairs, on=["origin_topK", "dest_topK"], how="inner"
        )

    filtered_routes = filtered_routes[
        (filtered_routes["flights"] >= min_flights) &
        (filtered_routes["origin_topK"] != "OTHER") &
        (filtered_routes["dest_topK"]   != "OTHER")
    ]

    if delay_filter == "On-time":
        filtered_routes = filtered_routes[filtered_routes["avg_class"] < 0.4]
    elif delay_filter == "Minor Delay":
        filtered_routes = filtered_routes[(filtered_routes["avg_class"] >= 0.4) & (filtered_routes["avg_class"] < 0.8)]
    elif delay_filter == "Major Delay":
        filtered_routes = filtered_routes[filtered_routes["avg_class"] >= 0.8]

    filtered_routes = filtered_routes.merge(
        airports.rename(columns={"lat":"orig_lat","lon":"orig_lon"}),
        left_on="origin_topK", right_index=True, how="inner"
    ).merge(
        airports.rename(columns={"lat":"dest_lat","lon":"dest_lon"}),
        left_on="dest_topK", right_index=True, how="inner"
    )

    if filtered_routes.empty:
        st.warning("No routes found with the current filters.")
    else:
        fig2 = go.Figure()

        for _, row in filtered_routes.iterrows():
            color = (
                "#16a34a" if row["avg_class"] < 0.4 else
                "#d97706" if row["avg_class"] < 0.8 else
                "#dc2626"
            )
            fig2.add_trace(go.Scattergeo(
                lon=[row["orig_lon"], row["dest_lon"]],
                lat=[row["orig_lat"], row["dest_lat"]],
                mode="lines",
                line=dict(width=1.2, color=color),
                opacity=0.6,
                showlegend=False,
                hoverinfo="skip",
            ))

        shown = pd.concat([
            filtered_routes[["origin_topK","orig_lat","orig_lon"]].rename(
                columns={"origin_topK":"iata","orig_lat":"lat","orig_lon":"lon"}),
            filtered_routes[["dest_topK","dest_lat","dest_lon"]].rename(
                columns={"dest_topK":"iata","dest_lat":"lat","dest_lon":"lon"}),
        ]).drop_duplicates("iata")

        fig2.add_trace(go.Scattergeo(
            lon=shown["lon"], lat=shown["lat"],
            text=shown["iata"],
            mode="markers+text",
            marker=dict(size=5, color="#6366f1"),
            textfont=dict(size=9, color="#e5e7eb"),
            textposition="top center",
            showlegend=False,
        ))

        for label, color in [("Mostly on-time","#16a34a"),("Minor delays","#d97706"),("Major delays","#dc2626")]:
            fig2.add_trace(go.Scattergeo(
                lon=[None], lat=[None], mode="lines",
                line=dict(color=color, width=3),
                name=label, showlegend=True,
            ))

        fig2.update_layout(
            geo=dict(
                scope="usa",
                projection_type="albers usa",
                showland=True, landcolor="#1e293b",
                showocean=True, oceancolor="#0f172a",
                showlakes=True, lakecolor="#0f172a",
                showcountries=False,
                bgcolor="rgba(0,0,0,0)",
            ),
            height=520,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            legend=dict(orientation="h", yanchor="bottom", y=0, xanchor="right", x=1),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"Showing {len(filtered_routes)} routes. 🟢 On-time &nbsp; 🟡 Minor delays &nbsp; 🔴 Major delays")
