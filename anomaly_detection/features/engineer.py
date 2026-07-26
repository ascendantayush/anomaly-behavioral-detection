"""
Feature engineering for the anomaly detection pipeline.

Transforms raw event logs into a numeric feature matrix suitable for the
Isolation Forest, rule engine, and Markov model.  Features capture temporal
patterns, frequency distributions, resource diversity, and sequential
behaviour per user.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from anomaly_detection.config import EVENT_TYPES, FEATURE_CFG
from anomaly_detection.utils.helpers import compute_entropy, safe_divide


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar and time-of-day features from each event.

    Args:
        df: Event DataFrame with a 'timestamp' column.

    Returns:
        DataFrame with added temporal columns.
    """
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["day_of_month"] = df["timestamp"].dt.day
    df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)
    return df


def compute_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-user frequency statistics over rolling windows.

    Features include event counts, unique resource/IP counts, failure rates,
    and byte-transfer aggregates within a configurable time window.

    Args:
        df: Event DataFrame (must already contain temporal features).

    Returns:
        DataFrame with added frequency columns.
    """
    df = df.copy()
    window = FEATURE_CFG.time_window_hours

    # Sort for rolling computations
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # --- Per-user rolling aggregations ---
    grouped = df.groupby("user_id")

    # Event count in rolling window (approximate via groupby size)
    df["events_in_window"] = grouped["event_id"].transform("count")

    # Unique IPs per user
    df["unique_ips"] = grouped["source_ip"].transform("nunique")

    # Unique devices per user
    df["unique_devices"] = grouped["device_id"].transform("nunique")

    # Unique resources per user
    df["unique_resources"] = grouped["resource"].transform("nunique")

    # Login failure rate per user
    login_mask = df["event_type"].isin(["login", "vpn_login"])
    total_logins = df.loc[login_mask].groupby("user_id")["success"].transform("count")
    failed_logins = df.loc[login_mask].groupby("user_id")["success"].transform(
        lambda s: (~s).sum()
    )
    df["login_failure_rate"] = 0.0
    df.loc[login_mask, "login_failure_rate"] = safe_divide_series(failed_logins, total_logins)

    # Bytes transferred aggregates
    df["total_bytes_user"] = grouped["bytes_transferred"].transform("sum")
    df["mean_bytes_user"] = grouped["bytes_transferred"].transform("mean")
    df["max_bytes_user"] = grouped["bytes_transferred"].transform("max")

    return df


def safe_divide_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise safe division avoiding zero-division errors.

    Args:
        numerator: Pandas Series.
        denominator: Pandas Series.

    Returns:
        Result Series with 0 where denominator is 0.
    """
    result = pd.Series(0.0, index=numerator.index)
    mask = denominator > 0
    result[mask] = numerator[mask] / denominator[mask]
    return result


def compute_event_type_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode event types as binary indicator columns and compute entropy.

    Args:
        df: Event DataFrame.

    Returns:
        DataFrame with one-hot event type columns and event_type_entropy.
    """
    df = df.copy()

    # One-hot encode event types
    for etype in EVENT_TYPES:
        col = f"etype_{etype}"
        df[col] = (df["event_type"] == etype).astype(int)

    # Per-user event-type entropy (measures behavioural diversity)
    # Incremental approach: maintain a running counter, not a growing list
    entropy_values = []
    for _, group in df.groupby("user_id"):
        from collections import Counter
        counter: Counter = Counter()
        total = 0
        group_entropy = []
        for etype in group["event_type"]:
            counter[etype] += 1
            total += 1
            # Shannon entropy from the counter
            probs = [c / total for c in counter.values()]
            ent = -sum(p * np.log2(p + 1e-12) for p in probs)
            group_entropy.append(ent)
        entropy_values.extend(group_entropy)
    df["event_type_entropy"] = entropy_values

    return df


def compute_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features related to event sequencing and transition patterns.

    Args:
        df: Event DataFrame sorted by user_id and timestamp.

    Returns:
        DataFrame with added sequence-related columns.
    """
    df = df.copy()
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Time since previous event for same user
    df["prev_timestamp"] = df.groupby("user_id")["timestamp"].shift(1)
    df["seconds_since_prev"] = (
        df["timestamp"] - df["prev_timestamp"]
    ).dt.total_seconds().fillna(-1)
    df["minutes_since_prev"] = df["seconds_since_prev"] / 60.0

    # Time since previous event of the same type
    df["prev_same_type_ts"] = df.groupby(["user_id", "event_type"])["timestamp"].shift(1)
    df["seconds_since_prev_same_type"] = (
        df["timestamp"] - df["prev_same_type_ts"]
    ).dt.total_seconds().fillna(-1)

    # Cumulative event count per user (position in session)
    df["cumulative_event_count"] = df.groupby("user_id").cumcount() + 1

    # Is first event of the day for this user?
    df["date"] = df["timestamp"].dt.date
    df["is_first_event_of_day"] = df.groupby(["user_id", "date"]).cumcount() == 0
    df["is_first_event_of_day"] = df["is_first_event_of_day"].astype(int)

    # Is last event of the day?
    df["is_last_event_of_day"] = df.groupby(["user_id", "date"]).cumcount(ascending=False) == 0
    df["is_last_event_of_day"] = df["is_last_event_of_day"].astype(int)

    # Drop helper columns
    df.drop(columns=["prev_timestamp", "prev_same_type_ts", "date"], inplace=True, errors="ignore")

    return df


def compute_sequence_transition_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise event-transition indicators.

    Creates columns like ``trans_login_to_file_access`` indicating when the
    current event follows a specific prior event for the same user.

    Args:
        df: Event DataFrame sorted by user_id and timestamp.

    Returns:
        DataFrame with transition indicator columns.
    """
    df = df.copy()
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    df["prev_event_type"] = df.groupby("user_id")["event_type"].shift(0)
    df["prev_event_type"] = df.groupby("user_id")["event_type"].shift(1)

    # Build transition columns for common transitions
    high_value_transitions = [
        ("login", "database_query"),
        ("login", "admin_command"),
        ("vpn_login", "file_access"),
        ("file_access", "email_access"),
        ("database_query", "shared_folder_access"),
        ("admin_command", "database_query"),
    ]

    for src, dst in high_value_transitions:
        col = f"trans_{src}_to_{dst}"
        df[col] = (
            (df["prev_event_type"] == src) & (df["event_type"] == dst)
        ).astype(int)

    df.drop(columns=["prev_event_type"], inplace=True, errors="ignore")
    return df


def compute_device_ip_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute device and IP novelty features.

    Tracks whether a given device_id or source_ip has been seen before for
    each user, which helps detect device spoofing.

    Args:
        df: Event DataFrame sorted by user_id and timestamp.

    Returns:
        DataFrame with device/ip novelty columns.
    """
    df = df.copy()
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Cumulative unique device count per user (proxy for new device detection)
    cum_devices = []
    for _, group in df.groupby("user_id"):
        seen: set = set()
        group_counts = []
        for dev in group["device_id"]:
            seen.add(dev)
            group_counts.append(len(seen))
        cum_devices.extend(group_counts)
    df["cumulative_devices_seen"] = cum_devices

    # Is this device new for this user? (first occurrence)
    df["is_new_device"] = (
        df.groupby(["user_id", "device_id"]).cumcount() == 0
    ).astype(int)

    # Is this IP new for this user?
    df["is_new_ip"] = (
        df.groupby(["user_id", "source_ip"]).cumcount() == 0
    ).astype(int)

    return df


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Build the complete numeric feature matrix for model consumption.

    Runs all feature engineering stages in sequence and returns only the
    numeric columns needed by downstream detectors.

    Args:
        df: Raw event DataFrame.

    Returns:
        Tuple of (feature_df, feature_names) where feature_df contains only
        numeric columns suitable for the Isolation Forest.
    """
    df = compute_temporal_features(df)
    df = compute_frequency_features(df)
    df = compute_event_type_features(df)
    df = compute_sequence_features(df)
    df = compute_sequence_transition_features(df)
    df = compute_device_ip_features(df)

    # Select only numeric columns for the model
    exclude_cols = {
        "event_id", "timestamp", "user_id", "username", "event_type",
        "source_ip", "device_id", "department", "resource", "is_attack",
        "attack_type", "latitude", "longitude", "city",
    }
    numeric_cols = [
        c for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    return df, numeric_cols
