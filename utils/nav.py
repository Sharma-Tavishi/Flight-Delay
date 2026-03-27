import streamlit as st

PAGES = [
    ("pages/predict.py",  "Predict"),
    ("pages/explain.py",  "Explain"),
    ("pages/insights.py", "Insights"),
    ("pages/about.py",    "About"),
]

DARK = {
    "nav_bg":          "#0e1117",
    "text_primary":    "#e2e8f0",
    "text_muted":      "#94a3b8",
    "border":          "rgba(255,255,255,0.08)",
    "font_color":      "#e5e7eb",
    "map_land":        "#1e293b",
    "map_ocean":       "#0f172a",
    "map_lake":        "#0f172a",
    "link_btn_bg":     "rgba(255,255,255,0.06)",
    "link_btn_border": "rgba(255,255,255,0.12)",
}

LIGHT = {
    "nav_bg":          "#ffffff",
    "text_primary":    "#1e293b",
    "text_muted":      "#64748b",
    "border":          "rgba(0,0,0,0.08)",
    "font_color":      "#1e293b",
    "map_land":        "#dde8f0",
    "map_ocean":       "#bfdbfe",
    "map_lake":        "#bfdbfe",
    "link_btn_bg":     "rgba(0,0,0,0.04)",
    "link_btn_border": "rgba(0,0,0,0.12)",
}


def get_theme() -> dict:
    return DARK if st.session_state.get("dark_mode", True) else LIGHT


def render_nav(current: str):
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = True

    t = get_theme()
    nav_bg = t["nav_bg"]

    base_css = f"""
    <style>
      [data-testid="collapsedControl"]  {{ display: none !important; }}
      section[data-testid="stSidebar"]  {{ display: none !important; }}
      header[data-testid="stHeader"]    {{ display: none !important; }}
      [data-testid="stDecoration"]      {{ display: none !important; }}
      .block-container {{ padding-top: 0.5rem !important; }}

      section[data-testid="stMain"] div.block-container
        > div[data-testid="stVerticalBlock"]:first-child {{
        position: sticky;
        top: 0;
        z-index: 999;
        background: {nav_bg};
        padding-top: 0.6rem;
      }}

      [data-testid="stMarkdownContainer"]:has(.nav-brand) + [data-testid="stPageLink"] a,
      [data-testid="stMarkdownContainer"]:has(.nav-brand) + [data-testid="stPageLink"] a * {{
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        color: {t["text_primary"]} !important;
        background: transparent !important;
        border: none !important;
        letter-spacing: -0.01em !important;
        padding: 0.35rem 0 !important;
      }}

      div[data-testid="stPageLink"] a {{
        display: inline-block !important;
        border-radius: 8px !important;
        padding: 0.35rem 1.1rem !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        text-decoration: none !important;
        color: {t["text_muted"]} !important;
        background: transparent !important;
        border: 1px solid transparent !important;
        white-space: nowrap !important;
        letter-spacing: 0.01em !important;
        transition: background 0.15s, color 0.15s, border 0.15s !important;
      }}
      div[data-testid="stPageLink"] a:hover {{
        background: rgba(99,102,241,0.1) !important;
        color: #c7d2fe !important;
        border: 1px solid rgba(99,102,241,0.25) !important;
      }}
      [data-testid="stMarkdownContainer"]:has(.nav-active) + [data-testid="stPageLink"] a,
      [data-testid="stMarkdownContainer"]:has(.nav-active) + [data-testid="stPageLink"] a * {{
        background: rgba(99,102,241,0.18) !important;
        color: #6366f1 !important;
        border: 1px solid rgba(99,102,241,0.5) !important;
        font-weight: 700 !important;
      }}
      div[data-testid="stHorizontalBlock"] {{
        gap: 0 !important;
        align-items: center !important;
      }}
      div[data-testid="stColumn"] {{ padding: 0 2px !important; }}

      /* toggle button styling */
      div[data-testid="stColumn"]:last-child div[data-testid="stButton"] button {{
        background: transparent !important;
        border: 1px solid {t["border"]} !important;
        border-radius: 8px !important;
        color: {t["text_muted"]} !important;
        font-size: 1.1rem !important;
        padding: 0.2rem 0.5rem !important;
        line-height: 1 !important;
        min-height: unset !important;
        width: 100% !important;
      }}
      div[data-testid="stColumn"]:last-child div[data-testid="stButton"] button:hover {{
        border-color: rgba(99,102,241,0.4) !important;
        background: rgba(99,102,241,0.08) !important;
      }}
    </style>
    """
    st.markdown(base_css, unsafe_allow_html=True)

    # light mode overrides — injected only when dark_mode is False
    if not st.session_state["dark_mode"]:
        light_css = """
        <style>
          /* app background */
          .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
          div.block-container {
            background-color: #f8fafc !important;
          }

          /* all text */
          .stApp, .stMarkdown, p, span, label, div,
          [data-testid="stMarkdownContainer"] p,
          [data-testid="stMarkdownContainer"] li,
          [data-testid="stMarkdownContainer"] td,
          [data-testid="stMarkdownContainer"] th {
            color: #1e293b !important;
          }

          /* muted / caption / small */
          .stCaption, small, caption,
          [data-testid="stCaptionContainer"] {
            color: #64748b !important;
          }

          /* headings */
          h1, h2, h3, h4, h5, h6 {
            color: #1e293b !important;
          }

          /* tabs */
          [data-testid="stTabs"] button[role="tab"] {
            color: #64748b !important;
          }
          [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            color: #1e293b !important;
            border-bottom-color: #6366f1 !important;
          }

          /* inputs & selects */
          [data-testid="stTextInput"] input,
          [data-testid="stNumberInput"] input,
          [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
          [data-testid="stTextArea"] textarea {
            background-color: #ffffff !important;
            color: #1e293b !important;
            border-color: #cbd5e1 !important;
          }
          [data-testid="stSelectbox"] svg { color: #64748b !important; }

          /* slider */
          [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
            background-color: #6366f1 !important;
          }

          /* metrics */
          [data-testid="stMetricValue"] { color: #1e293b !important; }
          [data-testid="stMetricLabel"] { color: #64748b !important; }

          /* expander */
          [data-testid="stExpander"] {
            border-color: rgba(0,0,0,0.1) !important;
          }
          [data-testid="stExpander"] summary { color: #1e293b !important; }

          /* alerts */
          [data-testid="stAlert"] { color: #1e293b !important; }
          [data-testid="stInfo"] {
            background-color: #eff6ff !important;
            border-color: #93c5fd !important;
            color: #1e293b !important;
          }
          [data-testid="stWarning"] {
            background-color: #fffbeb !important;
            border-color: #fcd34d !important;
            color: #1e293b !important;
          }
          [data-testid="stSuccess"] {
            background-color: #f0fdf4 !important;
            border-color: #86efac !important;
            color: #1e293b !important;
          }
          [data-testid="stError"] {
            background-color: #fef2f2 !important;
            border-color: #fca5a5 !important;
            color: #1e293b !important;
          }

          /* divider */
          hr { border-color: rgba(0,0,0,0.1) !important; }

          /* buttons (non-primary) */
          [data-testid="stButton"] button[kind="secondary"],
          [data-testid="stButton"] button:not([kind="primary"]) {
            color: #1e293b !important;
            border-color: #e2e8f0 !important;
            background-color: #ffffff !important;
          }
          [data-testid="stButton"] button[kind="secondary"]:hover,
          [data-testid="stButton"] button:not([kind="primary"]):hover {
            background-color: #f1f5f9 !important;
          }

          /* dataframe / table */
          [data-testid="stDataFrame"] * { color: #1e293b !important; }
          .stDataFrame table { background-color: #ffffff !important; }
          .stDataFrame thead th { background-color: #f1f5f9 !important; }
          .stDataFrame tbody tr:nth-child(even) td { background-color: #f8fafc !important; }
        </style>
        """
        st.markdown(light_css, unsafe_allow_html=True)

    brand, gap, p1, p2, p3, p4, toggle = st.columns([3, 0.5, 1, 1, 1, 1, 0.7])

    with brand:
        st.markdown('<div class="nav-brand">', unsafe_allow_html=True)
        st.page_link("pages/predict.py", label="✈ Flight Delay Predictor")
        st.markdown('</div>', unsafe_allow_html=True)

    for col, (path, label) in zip([p1, p2, p3, p4], PAGES):
        with col:
            if current == path:
                st.markdown('<div class="nav-active">', unsafe_allow_html=True)
            st.page_link(path, label=label)
            if current == path:
                st.markdown('</div>', unsafe_allow_html=True)

    with toggle:
        icon = "☀️" if st.session_state["dark_mode"] else "🌙"
        if st.button(icon, key="theme_toggle"):
            st.session_state["dark_mode"] = not st.session_state["dark_mode"]
            st.rerun()

    st.markdown(
        '<hr style="border:none;height:2px;margin:0.4rem 0 1.2rem 0;'
        'background:linear-gradient(90deg,#6366f1 0%,#a855f7 50%,rgba(99,102,241,0) 100%);">',
        unsafe_allow_html=True
    )
