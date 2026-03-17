import streamlit as st

PAGES = [
    ("pages/predict.py",  "Predict"),
    ("pages/explain.py",  "Explain"),
    ("pages/insights.py", "Insights"),
    ("pages/about.py",    "About"),
]

def render_nav(current: str):
    st.markdown("""
    <style>
      [data-testid="collapsedControl"]  { display: none !important; }
      section[data-testid="stSidebar"] { display: none !important; }
      .block-container { padding-top: 1.2rem !important; }

      [data-testid="stMarkdownContainer"]:has(.nav-brand) + [data-testid="stPageLink"] a,
      [data-testid="stMarkdownContainer"]:has(.nav-brand) + [data-testid="stPageLink"] a * {
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        color: #e2e8f0 !important;
        background: transparent !important;
        border: none !important;
        letter-spacing: -0.01em !important;
        padding: 0.35rem 0 !important;
      }

      div[data-testid="stPageLink"] a {
        display: inline-block !important;
        border-radius: 8px !important;
        padding: 0.35rem 1.1rem !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        text-decoration: none !important;
        color: #94a3b8 !important;
        background: transparent !important;
        border: 1px solid transparent !important;
        white-space: nowrap !important;
        letter-spacing: 0.01em !important;
        transition: background 0.15s, color 0.15s, border 0.15s !important;
      }
      div[data-testid="stPageLink"] a:hover {
        background: rgba(99,102,241,0.1) !important;
        color: #c7d2fe !important;
        border: 1px solid rgba(99,102,241,0.25) !important;
      }
      [data-testid="stMarkdownContainer"]:has(.nav-active) + [data-testid="stPageLink"] a,
      [data-testid="stMarkdownContainer"]:has(.nav-active) + [data-testid="stPageLink"] a * {
        background: rgba(99,102,241,0.18) !important;
        color: #6366f1 !important;
        border: 1px solid rgba(99,102,241,0.5) !important;
        font-weight: 700 !important;
      }
      div[data-testid="stHorizontalBlock"] {
        gap: 0 !important;
        align-items: center !important;
      }
      div[data-testid="stColumn"] { padding: 0 2px !important; }
    </style>
    """, unsafe_allow_html=True)

    brand, gap, p1, p2, p3, p4 = st.columns([3, 1, 1, 1, 1, 1])

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

    st.markdown(
        '<hr style="border:none;height:2px;margin:0.4rem 0 1.2rem 0;'
        'background:linear-gradient(90deg,#6366f1 0%,#a855f7 50%,rgba(99,102,241,0) 100%);">',
        unsafe_allow_html=True
    )
