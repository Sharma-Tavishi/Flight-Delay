import streamlit as st

st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="centered",
)

pg = st.navigation([
    st.Page("pages/predict.py",  title="Predict",      icon="✈️", default=True),
    st.Page("pages/explain.py",  title="Explain",      icon="🔍"),
    st.Page("pages/insights.py", title="Insights",     icon="📊"),
    st.Page("pages/about.py",    title="About Project", icon="ℹ️"),
])
pg.run()
