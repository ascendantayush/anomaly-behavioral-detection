"""
Shared utility functions for the anomaly detection pipeline.

Provides formatting, metric computation, and colour helpers used across
data generation, detection, classification, and the Streamlit dashboard.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def format_timestamp(ts: pd.Timestamp, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Return a human-readable string for a pandas Timestamp.

    Args:
        ts: The timestamp to format.
        fmt: strftime format string.

    Returns:
        Formatted date-time string.
    """
    return ts.strftime(fmt)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide two numbers, returning *default* when the denominator is zero.

    Args:
        numerator: Top value.
        denominator: Bottom value.
        default: Value to return on zero-division.

    Returns:
        Result of the division or *default*.
    """
    if denominator == 0:
        return default
    return numerator / denominator


def compute_basic_metrics(
    total_events: int,
    total_anomalies: int,
    total_users: int,
    total_days: int,
) -> Dict[str, Any]:
    """Derive high-level summary statistics for the dashboard.

    Args:
        total_events: Count of all events in the dataset.
        total_anomalies: Count of events flagged as anomalous.
        total_users: Unique user count.
        total_days: Number of days covered by the dataset.

    Returns:
        Dictionary with human-readable metric names and values.
    """
    return {
        "Total Events": total_events,
        "Total Anomalies": total_anomalies,
        "Anomaly Rate (%)": round(safe_divide(total_anomalies, total_events) * 100, 2),
        "Unique Users": total_users,
        "Days Covered": total_days,
        "Avg Events / Day / User": round(
            safe_divide(total_events, total_users * total_days), 1
        ),
    }


def color_risk_score(score: float) -> str:
    """Map a 0-100 risk score to a hex colour string.

    Args:
        score: Risk score between 0 and 100.

    Returns:
        Hex colour code.
    """
    if score < 25:
        return "#2ecc71"  # green  – low
    if score < 50:
        return "#f1c40f"  # yellow – medium-low
    if score < 75:
        return "#e67e22"  # orange – medium-high
    return "#e74c3c"  # red    – high / critical


def risk_label(score: float) -> str:
    """Return a human-readable risk label for a 0-100 score.

    Args:
        score: Risk score between 0 and 100.

    Returns:
        One of 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'.
    """
    if score < 25:
        return "LOW"
    if score < 50:
        return "MEDIUM"
    if score < 75:
        return "HIGH"
    return "CRITICAL"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two geo-coordinates in km.

    Args:
        lat1: Latitude of point 1.
        lon1: Longitude of point 1.
        lat2: Latitude of point 2.
        lon2: Longitude of point 2.

    Returns:
        Distance in kilometres.
    """
    R = 6371.0  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1))
        * np.cos(np.radians(lat2))
        * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def time_delta_hours(t1: pd.Timestamp, t2: pd.Timestamp) -> float:
    """Absolute difference between two timestamps in fractional hours.

    Args:
        t1: First timestamp.
        t2: Second timestamp.

    Returns:
        Absolute difference in hours.
    """
    delta = t2 - t1
    return abs(delta.total_seconds()) / 3600.0


def rolling_window_mask(
    timestamps: pd.Series,
    current_idx: int,
    window_hours: int,
) -> pd.Series:
    """Return a boolean mask selecting rows within *window_hours* of the current row.

    Args:
        timestamps: Series of datetime values (same order as the dataframe).
        current_idx: Integer index of the reference row.
        window_hours: Look-back window in hours.

    Returns:
        Boolean Series aligned with *timestamps*.
    """
    ref = timestamps.iloc[current_idx]
    start = ref - timedelta(hours=window_hours)
    return (timestamps >= start) & (timestamps <= ref)


def compute_entropy(values: Sequence[Any]) -> float:
    """Shannon entropy of a discrete sequence.

    Args:
        values: Iterable of categorical values.

    Returns:
        Entropy in bits.
    """
    if len(values) == 0:
        return 0.0
    counts = pd.Series(values).value_counts(normalize=True)
    return float(-np.sum(counts * np.log2(counts + 1e-12)))


def build_transition_counts(
    sequences: List[List[str]],
    order: int = 1,
) -> Dict[Tuple[str, ...], Dict[str, int]]:
    """Count n-gram transitions across a list of sequences.

    Args:
        sequences: List of event-type sequences (one per user/session).
        order: Markov chain order (number of preceding states).

    Returns:
        Nested dict: {prefix_tuple: {next_event: count}}.
    """
    transitions: Dict[Tuple[str, ...], Dict[str, int]] = {}
    for seq in sequences:
        for i in range(order, len(seq)):
            prefix = tuple(seq[i - order : i])
            nxt = seq[i]
            transitions.setdefault(prefix, {})
            transitions[prefix][nxt] = transitions[prefix].get(nxt, 0) + 1
    return transitions


def transition_matrix_to_probabilities(
    counts: Dict[Tuple[str, ...], Dict[str, int]],
    smoothing: float = 1e-6,
) -> Dict[Tuple[str, ...], Dict[str, float]]:
    """Convert raw transition counts to normalised probabilities.

    Args:
        counts: Output of ``build_transition_counts``.
        smoothing: Additive Laplace smoothing floor.

    Returns:
        Nested dict: {prefix_tuple: {next_event: probability}}.
    """
    probs: Dict[Tuple[str, ...], Dict[str, float]] = {}
    for prefix, next_counts in counts.items():
        total = sum(next_counts.values()) + smoothing * len(next_counts)
        probs[prefix] = {
            evt: (cnt + smoothing) / total
            for evt, cnt in next_counts.items()
        }
    return probs
