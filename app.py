"""
Streamlit Community Cloud entry point for AI-Powered Behavioral Anomaly Detection.

This thin wrapper delegates to the main application in anomaly_detection/app.py.
"""

import sys
from pathlib import Path

# Ensure the project root is on the Python path
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Set page config BEFORE importing the main app module (Streamlit requirement)
import streamlit as st

st.set_page_config(
    page_title="AI Behavioral Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
)

# Now import and run the main application logic
from anomaly_detection.app import (
    render_overview,
    render_live_alerts,
    render_entity_explorer,
    render_threat_analytics,
    _initialise_pipeline,
    ATTACK_SEVERITY,
    ATTACK_TYPES,
    DASH_CFG,
)

st.title("AI-Powered Behavioral Anomaly Detection for Cybersecurity")
st.caption("Real-time enterprise log monitoring with Isolation Forest, Rule Engine, and Markov Sequence Model")

# Run the full pipeline (cached after first execution)
df = _initialise_pipeline()

# Sidebar stats
with st.sidebar:
    st.header("Dataset Info")
    st.write(f"Events: **{len(df):,}**")
    st.write(f"Users: **{df['user_id'].nunique()}**")
    attacks = df[df["predicted_attack_type"] != "normal"]
    st.write(f"Anomalies: **{len(attacks):,}**")
    st.write(f"Anomaly rate: **{len(attacks)/len(df)*100:.1f}%**")

    st.divider()
    st.header("Attack Types")
    for atype in ATTACK_TYPES:
        count = (df["predicted_attack_type"] == atype).sum()
        severity = ATTACK_SEVERITY[atype]
        st.write(f"{'🔴' if severity >= 4 else '🟡' if severity >= 3 else '🟢'} {atype.replace('_', ' ').title()}: {count}")

# --- Tabs ---
tab_overview, tab_alerts, tab_entity, tab_analytics = st.tabs(
    ["Overview", "Live Alerts", "Entity Explorer", "Threat Analytics"]
)

with tab_overview:
    render_overview(df)

with tab_alerts:
    render_live_alerts(df)

with tab_entity:
    render_entity_explorer(df)

with tab_analytics:
    render_threat_analytics(df)
