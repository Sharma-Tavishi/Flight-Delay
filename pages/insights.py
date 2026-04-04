import warnings
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.nav import render_nav, get_theme
from utils.constants import AIRLINE_NAMES, MODEL_PATHS
render_nav("pages/insights.py")
t = get_theme()


@st.cache_data
def load_data():
    insights   = joblib.load(MODEL_PATHS["insights"])
    df         = pd.read_parquet("Data/processed_flights_sample.parquet")
    airports   = pd.read_csv("Data/airports.csv")
    airports.columns = [c.lower() for c in airports.columns]
    airports["iata"] = airports["iata"].str.strip().str.upper()
    airports = airports.drop_duplicates("iata").set_index("iata")[["lat","lon"]]
    # only continental US — Hawaii/Alaska routes draw diagonally across the albers usa projection
    airports = airports[(airports["lat"].between(24, 50)) & (airports["lon"].between(-125, -66))]
    return insights["airline_stats"], insights["route_stats"], airports, df

airline_stats, route_stats, airports, df = load_data()

st.markdown("<div style='font-size:2.2rem;font-weight:800;line-height:1.2;margin-bottom:0'>Insights</div>", unsafe_allow_html=True)
st.markdown("<p style='margin-top:0.2rem;'>Explore patterns in the flight delay dataset.</p>",
            unsafe_allow_html=True)
st.markdown(f"<hr style='border:none;border-top:1px solid {t['border']};margin:0.8rem 0 1rem 0'>", unsafe_allow_html=True)

tab_airline, tab_patterns, tab_map = st.tabs(["Airline Reliability", "Delay Patterns", "Route Map"])

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
        font=dict(color=t["font_color"]),
        xaxis=dict(
            title_font=dict(color=t["font_color"]),
            tickfont=dict(color=t["font_color"]),
        ),
        yaxis=dict(
            tickfont=dict(color=t["font_color"]),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color=t["font_color"]),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    table = grp[["carrier_name","on_time","minor","major","avg_delay_min"]].copy()
    table = table.sort_values("on_time", ascending=False).reset_index(drop=True)

    header_bg   = "rgba(99,102,241,0.15)"
    row_alt_bg  = "rgba(99,102,241,0.04)"
    border_col  = t["border"]
    font_col    = t["font_color"]

    rows_html = ""
    for i, r in table.iterrows():
        bg = row_alt_bg if i % 2 == 0 else "transparent"
        rows_html += f"""
        <tr style="background:{bg}">
          <td>{r['carrier_name']}</td>
          <td style="color:#16a34a;font-weight:600;">{r['on_time']*100:.1f}%</td>
          <td style="color:#d97706;font-weight:600;">{r['minor']*100:.1f}%</td>
          <td style="color:#dc2626;font-weight:600;">{r['major']*100:.1f}%</td>
          <td>{r['avg_delay_min']:.1f} min</td>
        </tr>"""

    st.markdown(f"""
    <style>
      .rel-table {{ width:100%;border-collapse:collapse;font-size:0.97rem;color:{font_col}; }}
      .rel-table th, .rel-table td {{
        text-align:center;padding:0.65rem 0.5rem;
        border-bottom:1px solid {border_col};width:20%;
      }}
      .rel-table th {{
        background:{header_bg};font-weight:700;letter-spacing:0.02em;
      }}
      .rel-table tr:last-child td {{ border-bottom:none; }}
      .rel-table tr:hover {{ background:rgba(99,102,241,0.08) !important; }}
    </style>
    <table class="rel-table">
      <thead><tr>
        <th>Airline</th>
        <th>On-time %</th>
        <th>Minor Delay %</th>
        <th>Major Delay %</th>
        <th>Avg Delay</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

with tab_patterns:
    st.markdown("On-time performance broken down by **departure hour**, **month**, and **day of week** across 987K flights.")

    MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    DOW_LABELS   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

    hour_ontime = df.groupby("dep_hour")["DELAY_CLASS"].apply(lambda x: (x == 0).mean() * 100).reset_index()
    hour_ontime.columns = ["hour", "ontime_pct"]

    month_ontime = df.groupby("month")["DELAY_CLASS"].apply(lambda x: (x == 0).mean() * 100).reset_index()
    month_ontime.columns = ["month", "ontime_pct"]
    month_ontime["month"] = month_ontime["month"].astype(int)
    month_ontime["label"] = month_ontime["month"].apply(lambda m: MONTH_LABELS[m - 1])

    dow_ontime = df.groupby("dayofweek")["DELAY_CLASS"].apply(lambda x: (x == 0).mean() * 100).reset_index()
    dow_ontime.columns = ["dow", "ontime_pct"]
    dow_ontime["dow"] = dow_ontime["dow"].astype(int)
    dow_ontime["label"] = dow_ontime["dow"].apply(lambda d: DOW_LABELS[d - 1])

    bar_color   = "#6366f1"
    hover_color = "#818cf8"

    fig_p = make_subplots(
        rows=1, cols=3,
        subplot_titles=["By Departure Hour", "By Month", "By Day of Week"],
        horizontal_spacing=0.10,
    )

    fig_p.add_trace(go.Scatter(
        x=hour_ontime["hour"],
        y=hour_ontime["ontime_pct"],
        mode="lines+markers",
        line=dict(color=bar_color, width=2.5),
        marker=dict(size=6, color=bar_color),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.12)",
        hovertemplate="%{x}:00 → %{y:.1f}% on-time<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    fig_p.add_trace(go.Bar(
        x=month_ontime["label"],
        y=month_ontime["ontime_pct"],
        marker_color=bar_color,
        hovertemplate="%{x}: %{y:.1f}% on-time<extra></extra>",
        showlegend=False,
    ), row=1, col=2)

    fig_p.add_trace(go.Bar(
        x=dow_ontime["label"],
        y=dow_ontime["ontime_pct"],
        marker_color=bar_color,
        hovertemplate="%{x}: %{y:.1f}% on-time<extra></extra>",
        showlegend=False,
    ), row=1, col=3)

    fig_p.update_yaxes(
        title_text="On-time %", title_font=dict(color=t["font_color"]),
        tickfont=dict(color=t["font_color"]), range=[70, 100], row=1, col=1,
    )
    fig_p.update_yaxes(
        tickfont=dict(color=t["font_color"]), range=[70, 100], row=1, col=2,
    )
    fig_p.update_yaxes(
        tickfont=dict(color=t["font_color"]), range=[70, 100], row=1, col=3,
    )
    fig_p.update_xaxes(tickfont=dict(color=t["font_color"]), title_font=dict(color=t["font_color"]))
    fig_p.update_annotations(font=dict(color=t["font_color"]))  # subplot titles

    fig_p.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["font_color"]),
    )

    st.plotly_chart(fig_p, use_container_width=True)
    st.caption("Y-axis starts at 70% to highlight relative differences between groups.")

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
            textfont=dict(size=9, color=t["font_color"]),
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
                showland=True, landcolor=t["map_land"],
                showocean=True, oceancolor=t["map_ocean"],
                showlakes=True, lakecolor=t["map_lake"],
                showcountries=False,
                bgcolor="rgba(0,0,0,0)",
            ),
            height=520,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=t["font_color"]),
            legend=dict(
                orientation="h", yanchor="bottom", y=0, xanchor="right", x=1,
                font=dict(color=t["font_color"]),
            ),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"Showing {len(filtered_routes)} routes. 🟢 On-time &nbsp; 🟡 Minor delays &nbsp; 🔴 Major delays")
