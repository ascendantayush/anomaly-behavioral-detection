"""
Streamlit dashboard for the AI-Powered Behavioral Anomaly Detection system.

Provides four tabs:
    1. **Overview**        – KPIs, event volume, attack distribution
    2. **Live Alerts**     – Alert feed with severity filtering and details
    3. **Entity Explorer** – Drill into individual user / device / IP behaviour
    4. **Threat Analytics**– Detector performance, risk heatmap, attack timeline
"""

import sys
from pathlib import Path

# Ensure the project root is on the Python path when run directly
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from anomaly_detection.config import (
    ATTACK_SEVERITY,
    ATTACK_TYPES,
    DASH_CFG,
    DATA_CFG,
)
from anomaly_detection.data.generator import generate_dataset
from anomaly_detection.data.injector import inject_all_attacks
from anomaly_detection.features.engineer import build_feature_matrix
from anomaly_detection.detection.isolation_forest import IsolationForestDetector
from anomaly_detection.detection.rule_engine import build_default_engine
from anomaly_detection.detection.markov_model import MarkovDetector
from anomaly_detection.classification.classifier import AttackClassifier
from anomaly_detection.utils.helpers import compute_basic_metrics

# ---------------------------------------------------------------------------
# Pipeline caching  – runs once per session, stored in st.session_state
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Running full pipeline (data → attacks → features → models → classification) …")
def _run_pipeline() -> pd.DataFrame:
    """Generate data, inject attacks, extract features, train models, classify (cached).

    Returns:
        Fully classified and scored DataFrame.
    """
    # 1. Generate synthetic data
    df, profiles = generate_dataset()

    # 2. Inject attacks
    df = inject_all_attacks(df, profiles)

    # 3. Feature engineering
    df, feature_cols = build_feature_matrix(df)

    # 4. Isolation Forest
    iso = IsolationForestDetector()
    iso.fit(df[feature_cols])
    df = iso.apply_to_dataframe(df, feature_cols)

    # 5. Rule Engine
    engine = build_default_engine()
    df = engine.evaluate(df)

    # 6. Markov Model
    markov = MarkovDetector()
    markov.fit(df)
    df = markov.apply_to_dataframe(df)

    # 7. Classifier + Risk Score
    classifier = AttackClassifier()
    df = classifier.classify(df)

    # Store metadata for other tabs
    st.session_state["profiles"] = profiles
    st.session_state["feature_cols"] = feature_cols

    return df


def _initialise_pipeline() -> pd.DataFrame:
    """Orchestrates the full pipeline and returns the final DataFrame.

    Returns:
        Classified and scored DataFrame.
    """
    return _run_pipeline()


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------

def render_overview(df: pd.DataFrame) -> None:
    """Render the Overview tab with KPIs and high-level charts.

    Args:
        df: Classified DataFrame.
    """
    st.header("Overview")

    # --- KPIs ---
    metrics = compute_basic_metrics(
        total_events=len(df),
        total_anomalies=int((df["predicted_attack_type"] != "normal").sum()),
        total_users=df["user_id"].nunique(),
        total_days=df["timestamp"].dt.date.nunique(),
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Events", f"{metrics['Total Events']:,}")
    c2.metric("Anomalies Detected", f"{metrics['Total Anomalies']:,}")
    c3.metric("Anomaly Rate", f"{metrics['Anomaly Rate (%)']}%")
    c4.metric("Unique Users", f"{metrics['Unique Users']}")

    c5, c6 = st.columns(2)
    c5.metric("Days Covered", metrics["Days Covered"])
    c6.metric("Avg Events / Day / User", metrics["Avg Events / Day / User"])

    st.divider()

    # --- Event volume over time ---
    st.subheader("Daily Event Volume")
    daily = df.set_index("timestamp").resample("D").size().reset_index(name="count")
    fig_vol = px.line(daily, x="timestamp", y="count", markers=True)
    fig_vol.update_layout(xaxis_title="Date", yaxis_title="Events")
    st.plotly_chart(fig_vol, use_container_width=True)

    # --- Attack type distribution ---
    st.subheader("Attack Type Distribution")
    attacks_only = df[df["predicted_attack_type"] != "normal"]
    if len(attacks_only) > 0:
        atype_counts = attacks_only["predicted_attack_type"].value_counts().reset_index()
        atype_counts.columns = ["Attack Type", "Count"]
        fig_pie = px.pie(
            atype_counts,
            names="Attack Type",
            values="Count",
            color="Attack Type",
            color_discrete_sequence=DASH_CFG.color_palette,
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No attacks detected.")

    # --- Risk score histogram ---
    st.subheader("Risk Score Distribution")
    fig_hist = px.histogram(
        df, x="risk_score", nbins=50, color="risk_level",
        color_discrete_map={
            "LOW": DASH_CFG.color_normal,
            "MEDIUM": DASH_CFG.color_low,
            "HIGH": DASH_CFG.color_medium,
            "CRITICAL": DASH_CFG.color_high,
        },
    )
    fig_hist.update_layout(xaxis_title="Risk Score", yaxis_title="Count")
    st.plotly_chart(fig_hist, use_container_width=True)


def render_live_alerts(df: pd.DataFrame) -> None:
    """Render the Live Alerts tab with filterable alert feed.

    Args:
        df: Classified DataFrame.
    """
    st.header("Live Alerts")

    attacks = df[df["predicted_attack_type"] != "normal"].copy()
    attacks = attacks.sort_values("risk_score", ascending=False)

    if len(attacks) == 0:
        st.info("No alerts to display.")
        return

    # --- Filters ---
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        selected_types = st.multiselect(
            "Attack Type",
            options=ATTACK_TYPES,
            default=ATTACK_TYPES,
        )
    with col_b:
        selected_level = st.multiselect(
            "Risk Level",
            options=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            default=["MEDIUM", "HIGH", "CRITICAL"],
        )
    with col_c:
        max_display = st.slider("Max alerts to show", 10, 500, 50)

    # Apply filters
    mask = (
        attacks["predicted_attack_type"].isin(selected_types)
        & attacks["risk_level"].isin(selected_level)
    )
    filtered = attacks.loc[mask].head(max_display)

    st.write(f"Showing **{len(filtered)}** of **{len(attacks)}** total alerts")

    # --- Alert table ---
    display_cols = [
        "event_id", "timestamp", "user_id", "username", "event_type",
        "source_ip", "device_id", "predicted_attack_type", "risk_score",
        "risk_level", "rule_names",
    ]
    existing_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[existing_cols].style.applymap(
            lambda _: "background-color: #2d2d2d",
            subset=["risk_score"],
        ),
        use_container_width=True,
        height=400,
    )

    # --- Alert detail expanders ---
    st.subheader("Alert Details")
    for idx, row in filtered.head(10).iterrows():
        with st.expander(f"{row['event_id']}  |  {row['predicted_attack_type']}  |  Risk {row['risk_score']}"):
            st.json({
                "event_id": row["event_id"],
                "timestamp": str(row["timestamp"]),
                "user": row["username"],
                "user_id": row["user_id"],
                "event_type": row["event_type"],
                "source_ip": row["source_ip"],
                "device_id": row["device_id"],
                "resource": row.get("resource", "N/A"),
                "bytes_transferred": row.get("bytes_transferred", 0),
                "predicted_attack_type": row["predicted_attack_type"],
                "risk_score": row["risk_score"],
                "risk_level": row["risk_level"],
                "rule_names": row.get("rule_names", ""),
                "iso_anomaly_score": round(row.get("iso_anomaly_score", 0), 4),
                "markov_score": round(row.get("markov_score", 0), 4),
                "is_attack_ground_truth": bool(row.get("is_attack", False)),
            })


def render_entity_explorer(df: pd.DataFrame) -> None:
    """Render the Entity Explorer tab for drilling into specific entities.

    Args:
        df: Classified DataFrame.
    """
    st.header("Entity Explorer")

    entity_type = st.radio(
        "Explore by",
        options=["User", "Device", "Source IP"],
        horizontal=True,
    )

    if entity_type == "User":
        col_values = df["username"].unique()
        key_col = "username"
    elif entity_type == "Device":
        col_values = df["device_id"].unique()
        key_col = "device_id"
    else:
        col_values = df["source_ip"].unique()
        key_col = "source_ip"

    selected = st.selectbox(f"Select {entity_type}", options=sorted(col_values))
    entity_df = df[df[key_col] == selected].copy()

    # --- Entity summary ---
    st.subheader(f"Summary for {selected}")
    ec1, ec2, ec3, ec4 = st.columns(4)
    ec1.metric("Total Events", len(entity_df))
    ec2.metric(
        "Anomalies",
        int((entity_df["predicted_attack_type"] != "normal").sum()),
    )
    ec3.metric("Avg Risk Score", f"{entity_df['risk_score'].mean():.1f}")
    ec4.metric(
        "Attack Types",
        entity_df[entity_df["predicted_attack_type"] != "normal"]["predicted_attack_type"].nunique(),
    )

    # --- Timeline ---
    st.subheader("Event Timeline")
    fig_tl = px.scatter(
        entity_df,
        x="timestamp",
        y="event_type",
        color="predicted_attack_type",
        size="risk_score",
        color_discrete_map={"normal": DASH_CFG.color_normal, **{t: DASH_CFG.color_palette[i % len(DASH_CFG.color_palette)] for i, t in enumerate(ATTACK_TYPES)}},
        hover_data=["event_id", "source_ip", "device_id", "risk_score"],
    )
    fig_tl.update_layout(height=350)
    st.plotly_chart(fig_tl, use_container_width=True)

    # --- Risk over time ---
    st.subheader("Risk Score Over Time")
    fig_risk = px.line(
        entity_df, x="timestamp", y="risk_score",
        color_discrete_sequence=[DASH_CFG.color_high],
    )
    fig_risk.add_hline(y=50, line_dash="dash", line_color="yellow", annotation_text="Medium threshold")
    fig_risk.add_hline(y=75, line_dash="dash", line_color="red", annotation_text="High threshold")
    st.plotly_chart(fig_risk, use_container_width=True)

    # --- Event type breakdown ---
    st.subheader("Event Type Breakdown")
    etype_counts = entity_df["event_type"].value_counts().reset_index()
    etype_counts.columns = ["Event Type", "Count"]
    fig_etype = px.bar(etype_counts, x="Event Type", y="Count", color="Count", color_continuous_scale="Blues")
    st.plotly_chart(fig_etype, use_container_width=True)


def render_threat_analytics(df: pd.DataFrame) -> None:
    """Render the Threat Analytics tab with detector performance and heatmaps.

    Args:
        df: Classified DataFrame.
    """
    st.header("Threat Analytics")

    # --- Detector agreement matrix ---
    st.subheader("Detector Agreement")
    detector_cols = ["iso_is_anomaly", "rule_anomaly", "markov_is_anomaly"]
    existing = [c for c in detector_cols if c in df.columns]
    if len(existing) == 3:
        agree_all = ((df["iso_is_anomaly"] == 1) & (df["rule_anomaly"] == 1) & (df["markov_is_anomaly"] == 1)).sum()
        agree_if_rule = ((df["iso_is_anomaly"] == 1) & (df["rule_anomaly"] == 1)).sum()
        agree_if_markov = ((df["iso_is_anomaly"] == 1) & (df["markov_is_anomaly"] == 1)).sum()
        agree_rule_markov = ((df["rule_anomaly"] == 1) & (df["markov_is_anomaly"] == 1)).sum()
        only_if = ((df["iso_is_anomaly"] == 1) & (df["rule_anomaly"] == 0) & (df["markov_is_anomaly"] == 0)).sum()
        only_rule = ((df["iso_is_anomaly"] == 0) & (df["rule_anomaly"] == 1) & (df["markov_is_anomaly"] == 0)).sum()
        only_markov = ((df["iso_is_anomaly"] == 0) & (df["rule_anomaly"] == 0) & (df["markov_is_anomaly"] == 1)).sum()

        agree_data = pd.DataFrame({
            "Agreement": [
                "All three", "IF + Rule", "IF + Markov", "Rule + Markov",
                "IF only", "Rule only", "Markov only",
            ],
            "Count": [agree_all, agree_if_rule, agree_if_markov, agree_rule_markov, only_if, only_rule, only_markov],
        })
        fig_agree = px.bar(agree_data, x="Agreement", y="Count", color="Count", color_continuous_scale="Viridis")
        st.plotly_chart(fig_agree, use_container_width=True)

    # --- Attack type vs detector heatmap ---
    st.subheader("Attack Type Detection by Source")
    attacks = df[df["predicted_attack_type"] != "normal"]
    if len(attacks) > 0:
        # Assign each row a single detector source based on its own values
        def _detector_source(row):
            if row.get("rule_anomaly", 0) == 1:
                return "Rule Engine"
            if row.get("iso_is_anomaly", 0) == 1:
                return "Isolation Forest"
            return "Markov"

        attacks = attacks.copy()
        attacks["detector_source"] = attacks.apply(_detector_source, axis=1)

        cross = pd.crosstab(
            attacks["predicted_attack_type"],
            attacks["detector_source"],
        )
        fig_heat = px.imshow(
            cross,
            labels=dict(x="Detector", y="Attack Type", color="Events Detected"),
            color_continuous_scale="YlOrRd",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # --- Hourly attack heatmap ---
    st.subheader("Hourly Attack Distribution")
    if len(attacks) > 0:
        attacks_copy = attacks.copy()
        attacks_copy["hour"] = pd.to_datetime(attacks_copy["timestamp"]).dt.hour
        hour_dow = attacks_copy.groupby(["hour", "predicted_attack_type"]).size().reset_index(name="count")
        fig_hourly = px.bar(
            hour_dow, x="hour", y="count", color="predicted_attack_type",
            barmode="stack",
            color_discrete_sequence=DASH_CFG.color_palette,
        )
        fig_hourly.update_layout(xaxis_title="Hour of Day", yaxis_title="Attack Count")
        st.plotly_chart(fig_hourly, use_container_width=True)

    # --- Risk score by attack type ---
    st.subheader("Risk Score by Attack Type")
    if len(attacks) > 0:
        fig_box = px.box(
            attacks, x="predicted_attack_type", y="risk_score",
            color="predicted_attack_type",
            color_discrete_sequence=DASH_CFG.color_palette,
        )
        fig_box.update_layout(xaxis_title="Attack Type", yaxis_title="Risk Score", showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    # --- Ground truth accuracy ---
    st.subheader("Ground Truth Accuracy")
    if "is_attack" in df.columns:
        tp = ((df["predicted_attack_type"] != "normal") & (df["is_attack"] == True)).sum()  # noqa: E712
        fp = ((df["predicted_attack_type"] != "normal") & (df["is_attack"] == False)).sum()  # noqa: E712
        fn = ((df["predicted_attack_type"] == "normal") & (df["is_attack"] == True)).sum()  # noqa: E712
        tn = ((df["predicted_attack_type"] == "normal") & (df["is_attack"] == False)).sum()  # noqa: E712
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(df) if len(df) > 0 else 0

        ac1, ac2, ac3, ac4 = st.columns(4)
        ac1.metric("Precision", f"{precision:.1%}")
        ac2.metric("Recall", f"{recall:.1%}")
        ac3.metric("F1 Score", f"{f1:.1%}")
        ac4.metric("Accuracy", f"{accuracy:.1%}")

        cm_data = pd.DataFrame(
            {"Predicted Positive": [tp, fp], "Predicted Negative": [fn, tn]},
            index=["Actual Positive", "Actual Negative"],
        )
        fig_cm = px.imshow(cm_data, text_auto=True, color_continuous_scale="Blues",
                           labels=dict(color="Count"))
        fig_cm.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the Streamlit application."""
    st.set_page_config(
        page_title=DASH_CFG.page_title,
        page_icon=DASH_CFG.page_icon,
        layout=DASH_CFG.layout,
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


if __name__ == "__main__":
    main()
