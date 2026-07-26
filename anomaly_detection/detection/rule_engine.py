"""
Rule-based anomaly detection engine.

A plain-Python rule engine that evaluates boolean predicates against event
data.  Each rule carries a name, severity, and attack-type mapping.  The
engine evaluates all rules and produces a combined alert DataFrame.

All rules use vectorised pandas operations for performance — no nested
Python loops over individual events.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from anomaly_detection.config import RULE_CFG


@dataclass
class Rule:
    """A single detection rule.

    Attributes:
        name: Human-readable rule identifier.
        description: What this rule detects.
        attack_type: Canonical attack type this rule maps to.
        severity: 1 (low) to 5 (critical).
        evaluate_fn: Callable that takes a DataFrame and returns a boolean mask.
    """

    name: str
    description: str
    attack_type: str
    severity: int
    evaluate_fn: Callable[[pd.DataFrame], pd.Series]


class RuleEngine:
    """Container and evaluator for detection rules.

    Attributes:
        rules: List of registered Rule objects.
    """

    def __init__(self) -> None:
        """Initialise an empty rule engine."""
        self.rules: List[Rule] = []

    def add_rule(self, rule: Rule) -> None:
        """Register a new rule.

        Args:
            rule: A Rule instance to add to the engine.
        """
        self.rules.append(rule)

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate all registered rules against the DataFrame.

        For each rule, marks rows where the predicate is True and records the
        rule name, severity, and attack type.

        Args:
            df: Event DataFrame to evaluate.

        Returns:
            DataFrame with added columns: ``rule_anomaly``, ``rule_names``,
            ``rule_severity``, ``rule_attack_type``.
        """
        df = df.copy()
        df["rule_anomaly"] = 0
        df["rule_names"] = ""
        df["rule_severity"] = 0
        df["rule_attack_type"] = "none"

        for rule in self.rules:
            try:
                mask = rule.evaluate_fn(df)
                mask = mask.astype(bool)
            except Exception:
                continue

            # Update rows matched by this rule
            df.loc[mask, "rule_anomaly"] = 1

            # Accumulate rule names (comma-separated)
            new_names = df.loc[mask, "rule_names"].apply(
                lambda existing: f"{existing},{rule.name}" if existing else rule.name
            )
            df.loc[mask, "rule_names"] = new_names

            # Take the highest severity
            df["rule_severity"] = df["rule_severity"].where(
                df["rule_severity"] >= rule.severity,
                other=rule.severity,
            )

            # Set attack type (first match wins)
            still_none = df["rule_attack_type"] == "none"
            df.loc[still_none & mask, "rule_attack_type"] = rule.attack_type

        return df

    def get_triggered_rules(self, df: pd.DataFrame) -> Dict[str, int]:
        """Count how many events each rule triggered on.

        Args:
            df: The evaluated DataFrame (output of ``evaluate``).

        Returns:
            Dict mapping rule name to its trigger count.
        """
        counts: Dict[str, int] = {}
        all_names = df["rule_names"].str.split(",").explode()
        all_names = all_names[all_names != ""]
        return all_names.value_counts().to_dict()


# ---------------------------------------------------------------------------
# Vectorised rule definitions — no per-row Python loops
# ---------------------------------------------------------------------------

def _rule_brute_force_login(df: pd.DataFrame) -> pd.Series:
    """Detect brute-force login attempts using a rolling window count.

    Flags login/vpn_login events where the same user has >= N failed
    logins within a rolling time window.

    Args:
        df: Event DataFrame.

    Returns:
        Boolean Series marking brute-force events.
    """
    mask = pd.Series(False, index=df.index)
    login_mask = df["event_type"].isin(["login", "vpn_login"])
    logins = df.loc[login_mask].copy()

    if logins.empty:
        return mask

    logins = logins.sort_values(["user_id", "timestamp"])
    window = pd.Timedelta(minutes=RULE_CFG.brute_force_window_minutes)

    # For each user, count failed logins in a rolling window
    for user_id, group in logins.groupby("user_id"):
        timestamps = group["timestamp"]
        successes = group["success"]

        for i in range(len(group)):
            window_start = timestamps.iloc[i] - window
            in_window = (timestamps >= window_start) & (timestamps <= timestamps.iloc[i])
            failures_in_window = (~successes.iloc[in_window.values]).sum()
            if failures_in_window >= RULE_CFG.brute_force_failed_threshold:
                mask.iloc[group.index[i]] = True

    return mask


def _rule_impossible_travel(df: pd.DataFrame) -> pd.Series:
    """Detect impossible-travel logins from different IPs in short time.

    Uses shift-based comparison: for consecutive logins by the same user,
    if the IP changed and the time gap is < threshold, flag both events.

    Args:
        df: Event DataFrame.

    Returns:
        Boolean Series marking impossible-travel events.
    """
    mask = pd.Series(False, index=df.index)
    login_mask = df["event_type"].isin(["login", "vpn_login"])
    logins = df.loc[login_mask].copy()

    if logins.empty:
        return mask

    logins = logins.sort_values(["user_id", "timestamp"])

    # Shift within each user group to compare consecutive logins
    logins["prev_ip"] = logins.groupby("user_id")["source_ip"].shift(1)
    logins["prev_ts"] = logins.groupby("user_id")["timestamp"].shift(1)
    logins["ip_changed"] = logins["source_ip"] != logins["prev_ip"]
    logins["time_gap_hours"] = (
        (logins["timestamp"] - logins["prev_ts"]).dt.total_seconds() / 3600.0
    )

    impossible = (
        logins["ip_changed"]
        & (logins["time_gap_hours"] < RULE_CFG.impossible_travel_max_hours)
        & (logins["time_gap_hours"] > 0)
    )

    mask.loc[logins.index[impossible]] = True

    return mask


def _rule_credential_stuffing(df: pd.DataFrame) -> pd.Series:
    """Detect credential-stuffing (many users targeted from one IP).

    Flags all login events from IPs that target >= N unique users.

    Args:
        df: Event DataFrame.

    Returns:
        Boolean Series marking credential-stuffing events.
    """
    mask = pd.Series(False, index=df.index)
    login_mask = df["event_type"].isin(["login", "vpn_login"])
    logins = df.loc[login_mask]

    if logins.empty:
        return mask

    # Count unique users per source IP
    users_per_ip = logins.groupby("source_ip")["user_id"].nunique()
    flagged_ips = users_per_ip[users_per_ip >= RULE_CFG.credential_stuffing_unique_users].index
    mask.loc[logins[logins["source_ip"].isin(flagged_ips)].index] = True

    return mask


def _rule_lateral_movement(df: pd.DataFrame) -> pd.Series:
    """Detect lateral movement (many unique resources in short time).

    Uses a 2-hour rolling window per user to count unique resources.

    Args:
        df: Event DataFrame.

    Returns:
        Boolean Series marking lateral movement events.
    """
    mask = pd.Series(False, index=df.index)
    window = pd.Timedelta(hours=RULE_CFG.lateral_movement_window_hours)

    for user_id, group in df.groupby("user_id"):
        if len(group) < RULE_CFG.lateral_movement_unique_resources:
            continue
        group = group.sort_values("timestamp")
        timestamps = group["timestamp"]
        resources = group["resource"]

        for i in range(len(group)):
            window_start = timestamps.iloc[i] - window
            in_window = (timestamps >= window_start) & (timestamps <= timestamps.iloc[i])
            unique_res = resources.iloc[in_window.values].nunique()
            if unique_res >= RULE_CFG.lateral_movement_unique_resources:
                mask.iloc[group.index[i]] = True

    return mask


def _rule_device_spoofing(df: pd.DataFrame) -> pd.Series:
    """Detect device spoofing (login from a never-before-seen device).

    Uses cumulative device tracking per user.

    Args:
        df: Event DataFrame.

    Returns:
        Boolean Series marking device-spoofing events.
    """
    mask = pd.Series(False, index=df.index)
    login_mask = df["event_type"].isin(["login", "vpn_login"])
    logins = df.loc[login_mask].copy()

    if logins.empty:
        return mask

    logins = logins.sort_values(["user_id", "timestamp"])

    # For each user, check if this is a new device (and user has history)
    seen_devices: Dict[str, set] = {}
    for idx, row in logins.iterrows():
        uid = row["user_id"]
        dev = row["device_id"]
        if uid not in seen_devices:
            seen_devices[uid] = set()
        if dev not in seen_devices[uid] and len(seen_devices[uid]) > 0:
            mask.iloc[df.index.get_loc(idx)] = True
        seen_devices[uid].add(dev)

    return mask


def _rule_low_and_slow_exfiltration(df: pd.DataFrame) -> pd.Series:
    """Detect low-and-slow exfiltration via daily cumulative byte volume.

    Flags transfer events where the user's daily total exceeds threshold.

    Args:
        df: Event DataFrame.

    Returns:
        Boolean Series marking exfiltration events.
    """
    mask = pd.Series(False, index=df.index)
    transfer_types = ["file_access", "email_access", "shared_folder_access", "usb_usage"]
    transfer_mask = df["event_type"].isin(transfer_types)

    if not transfer_mask.any():
        return mask

    # Calculate daily totals per user
    transfers = df.loc[transfer_mask].copy()
    transfers["date"] = transfers["timestamp"].dt.date
    daily_totals = transfers.groupby(["user_id", "date"])["bytes_transferred"].transform("sum")

    # Flag events where daily total exceeds threshold
    threshold = RULE_CFG.exfiltration_daily_volume_kb * 1024
    mask.loc[transfers.index[daily_totals > threshold]] = True

    return mask


def _rule_insider_drift(df: pd.DataFrame) -> pd.Series:
    """Detect insider drift via increasing night activity over time.

    Compares first-half vs second-half night activity rate per user.

    Args:
        df: Event DataFrame.

    Returns:
        Boolean Series marking insider-drift events.
    """
    mask = pd.Series(False, index=df.index)

    for user_id, group in df.groupby("user_id"):
        if len(group) < 20:
            continue
        group = group.sort_values("timestamp")
        mid = len(group) // 2
        first_half = group.iloc[:mid]
        second_half = group.iloc[mid:]

        first_rate = first_half["is_night"].mean()
        second_rate = second_half["is_night"].mean()

        if second_rate > first_rate + RULE_CFG.insider_drift_deviation_threshold * 0.1:
            night_in_second = second_half[second_half["is_night"] == 1]
            mask.loc[night_in_second.index] = True

    return mask


def build_default_engine() -> RuleEngine:
    """Create a RuleEngine pre-loaded with all seven detection rules.

    Returns:
        A fully configured RuleEngine instance.
    """
    engine = RuleEngine()

    engine.add_rule(Rule(
        name="brute_force_login",
        description="Rapid failed login attempts against a single account",
        attack_type="brute_force",
        severity=4,
        evaluate_fn=_rule_brute_force_login,
    ))
    engine.add_rule(Rule(
        name="impossible_travel",
        description="Logins from geographically distant locations in impossible time",
        attack_type="impossible_travel",
        severity=5,
        evaluate_fn=_rule_impossible_travel,
    ))
    engine.add_rule(Rule(
        name="credential_stuffing",
        description="Many different user accounts targeted from a single IP",
        attack_type="credential_stuffing",
        severity=4,
        evaluate_fn=_rule_credential_stuffing,
    ))
    engine.add_rule(Rule(
        name="lateral_movement",
        description="Unusually many unique internal resources accessed in short time",
        attack_type="lateral_movement",
        severity=5,
        evaluate_fn=_rule_lateral_movement,
    ))
    engine.add_rule(Rule(
        name="device_spoofing",
        description="Login from a previously unseen device for this user",
        attack_type="device_spoofing",
        severity=4,
        evaluate_fn=_rule_device_spoofing,
    ))
    engine.add_rule(Rule(
        name="low_and_slow_exfiltration",
        description="Cumulative data transfer volume exceeds daily threshold",
        attack_type="low_and_slow_exfiltration",
        severity=3,
        evaluate_fn=_rule_low_and_slow_exfiltration,
    ))
    engine.add_rule(Rule(
        name="insider_drift",
        description="Increasing off-hours activity compared to historical baseline",
        attack_type="insider_drift",
        severity=2,
        evaluate_fn=_rule_insider_drift,
    ))

    return engine
