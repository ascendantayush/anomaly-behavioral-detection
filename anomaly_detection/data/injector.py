"""
Attack pattern injector for the anomaly detection pipeline.

Replaces a fraction of normal events with realistic attack patterns.
Each injector produces events that mimic a specific threat category while
preserving the surrounding data distribution so that detectors must actually
learn meaningful signals rather than trivial artefacts.
"""

import random
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from faker import Faker

from anomaly_detection.config import INJECTION_CFG

fake = Faker()
np.random.seed(42)
random.seed(42)

# Approximate geo-coordinates for impossible-travel simulation
CITY_GEO: Dict[str, Tuple[float, float]] = {
    "new_york": (40.7128, -74.0060),
    "london": (51.5074, -0.1278),
    "tokyo": (35.6762, 139.6503),
    "sydney": (-33.8688, 151.2093),
    "dubai": (25.2048, 55.2708),
    "san_francisco": (37.7749, -122.4194),
    "berlin": (52.5200, 13.4050),
    "mumbai": (19.0760, 72.8777),
}


def _select_normal_events(df: pd.DataFrame, ratio: float) -> pd.Index:
    """Select random normal event indices to be replaced by attacks.

    Args:
        df: Full event DataFrame.
        ratio: Fraction of total events to select.

    Returns:
        Index of selected rows.
    """
    n = max(1, int(len(df) * ratio))
    normal_mask = df["is_attack"] == False  # noqa: E712
    normal_indices = df.index[normal_mask]
    if len(normal_indices) == 0:
        return pd.Index([])
    return pd.Index(random.sample(list(normal_indices), min(n, len(normal_indices))))


def inject_brute_force(df: pd.DataFrame, profiles: List[Dict[str, Any]]) -> pd.DataFrame:
    """Inject brute-force login attacks.

    Simulates rapid repeated failed login attempts against a single account
    from one or more source IPs within a short time window.

    Args:
        df: Event DataFrame (will be copied).
        profiles: User profile list.

    Returns:
        Modified DataFrame with brute-force events injected.
    """
    df = df.copy()
    target_idx = _select_normal_events(df, INJECTION_CFG.brute_force_ratio)
    if len(target_idx) == 0:
        return df

    victim = random.choice(profiles)
    attacker_ip = fake.ipv4_public()
    window_start = df.loc[target_idx, "timestamp"].min()

    new_events: List[Dict[str, Any]] = []
    for i, idx in enumerate(target_idx):
        ts = window_start + timedelta(seconds=random.randint(0, 900))
        event = {
            "event_id": f"ATT-BF-{i:05d}",
            "timestamp": ts,
            "user_id": victim["user_id"],
            "username": victim["username"],
            "event_type": "login",
            "source_ip": attacker_ip,
            "device_id": f"DEV-{random.randint(10000, 99999)}",
            "department": victim["department"],
            "resource": "auth_service",
            "success": random.random() < 0.08,
            "bytes_transferred": 0,
            "is_attack": True,
            "attack_type": "brute_force",
        }
        new_events.append(event)

    attack_df = pd.DataFrame(new_events)
    df = pd.concat([df, attack_df], ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def inject_impossible_travel(df: pd.DataFrame, profiles: List[Dict[str, Any]]) -> pd.DataFrame:
    """Inject impossible-travel login events.

    Places two login events for the same user in distant cities with an
    impossibly short time gap between them.

    Args:
        df: Event DataFrame.
        profiles: User profile list.

    Returns:
        Modified DataFrame.
    """
    df = df.copy()
    target_idx = _select_normal_events(df, INJECTION_CFG.impossible_travel_ratio)
    if len(target_idx) == 0:
        return df

    cities = list(CITY_GEO.keys())
    victim = random.choice(profiles)

    new_events: List[Dict[str, Any]] = []
    for i, idx in enumerate(target_idx):
        city_a, city_b = random.sample(cities, 2)
        lat_a, lon_a = CITY_GEO[city_a]
        lat_b, lon_b = CITY_GEO[city_b]
        base_ts = df.loc[idx, "timestamp"]

        # First login in city A
        event_a = {
            "event_id": f"ATT-IT-{i:04d}-A",
            "timestamp": base_ts,
            "user_id": victim["user_id"],
            "username": victim["username"],
            "event_type": "login",
            "source_ip": fake.ipv4_public(),
            "device_id": random.choice(victim["device_ids"]),
            "department": victim["department"],
            "resource": "auth_service",
            "success": True,
            "bytes_transferred": 0,
            "is_attack": True,
            "attack_type": "impossible_travel",
            "latitude": lat_a,
            "longitude": lon_a,
            "city": city_a,
        }
        # Second login in city B within 30-90 minutes
        event_b = {
            "event_id": f"ATT-IT-{i:04d}-B",
            "timestamp": base_ts + timedelta(minutes=random.randint(30, 90)),
            "user_id": victim["user_id"],
            "username": victim["username"],
            "event_type": "login",
            "source_ip": fake.ipv4_public(),
            "device_id": random.choice(victim["device_ids"]),
            "department": victim["department"],
            "resource": "auth_service",
            "success": True,
            "bytes_transferred": 0,
            "is_attack": True,
            "attack_type": "impossible_travel",
            "latitude": lat_b,
            "longitude": lon_b,
            "city": city_b,
        }
        new_events.extend([event_a, event_b])

    attack_df = pd.DataFrame(new_events)
    df = pd.concat([df, attack_df], ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def inject_credential_stuffing(df: pd.DataFrame, profiles: List[Dict[str, Any]]) -> pd.DataFrame:
    """Inject credential-stuffing attacks.

    Simulates a single attacker IP attempting logins against many different
    user accounts with a mix of failures and unexpected successes.

    Args:
        df: Event DataFrame.
        profiles: User profile list.

    Returns:
        Modified DataFrame.
    """
    df = df.copy()
    target_idx = _select_normal_events(df, INJECTION_CFG.credential_stuffing_ratio)
    if len(target_idx) == 0:
        return df

    attacker_ip = fake.ipv4_public()
    num_targets = min(len(profiles), random.randint(6, 15))
    victims = random.sample(profiles, num_targets)

    new_events: List[Dict[str, Any]] = []
    base_ts = df.loc[target_idx, "timestamp"].min()

    for i, victim in enumerate(victims):
        n_attempts = random.randint(1, 3)
        for j in range(n_attempts):
            ts = base_ts + timedelta(seconds=random.randint(0, 1800))
            event = {
                "event_id": f"ATT-CS-{i:04d}-{j:02d}",
                "timestamp": ts,
                "user_id": victim["user_id"],
                "username": victim["username"],
                "event_type": "login",
                "source_ip": attacker_ip,
                "device_id": f"DEV-{random.randint(10000, 99999)}",
                "department": victim["department"],
                "resource": "auth_service",
                "success": random.random() < 0.12,
                "bytes_transferred": 0,
                "is_attack": True,
                "attack_type": "credential_stuffing",
            }
            new_events.append(event)

    attack_df = pd.DataFrame(new_events)
    df = pd.concat([df, attack_df], ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def inject_lateral_movement(df: pd.DataFrame, profiles: List[Dict[str, Any]]) -> pd.DataFrame:
    """Inject lateral movement sequences.

    Models an attacker who has gained a foothold and is pivoting through
    internal resources (file server -> database -> admin command) unusually
    quickly.

    Args:
        df: Event DataFrame.
        profiles: User profile list.

    Returns:
        Modified DataFrame.
    """
    df = df.copy()
    target_idx = _select_normal_events(df, INJECTION_CFG.lateral_movement_ratio)
    if len(target_idx) == 0:
        return df

    victim = random.choice(profiles)
    attacker_ip = fake.ipv4_public()

    lateral_sequence = [
        ("file_access", f"/data/sensitive/{fake.file_name()}"),
        ("shared_folder_access", f"\\\\fileserver\\confidential\\{fake.file_name()}"),
        ("database_query", f"db://prod/users SELECT * FROM credentials"),
        ("admin_command", "/usr/bin/sudo su - root"),
        ("email_access", "exchange_server"),
    ]

    new_events: List[Dict[str, Any]] = []
    base_ts = df.loc[target_idx, "timestamp"].min()

    for i, idx in enumerate(target_idx):
        for j, (etype, resource) in enumerate(lateral_sequence):
            ts = base_ts + timedelta(minutes=random.randint(0, 120), seconds=random.randint(0, 59))
            event = {
                "event_id": f"ATT-LM-{i:04d}-{j:02d}",
                "timestamp": ts,
                "user_id": victim["user_id"],
                "username": victim["username"],
                "event_type": etype,
                "source_ip": attacker_ip,
                "device_id": random.choice(victim["device_ids"]),
                "department": victim["department"],
                "resource": resource,
                "success": True,
                "bytes_transferred": int(np.random.lognormal(10, 1)),
                "is_attack": True,
                "attack_type": "lateral_movement",
            }
            new_events.append(event)

    attack_df = pd.DataFrame(new_events)
    df = pd.concat([df, attack_df], ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def inject_device_spoofing(df: pd.DataFrame, profiles: List[Dict[str, Any]]) -> pd.DataFrame:
    """Inject device-spoofing events.

    A known user authenticates from an unrecognised device while maintaining
    normal behavioural timing, making it harder to detect.

    Args:
        df: Event DataFrame.
        profiles: User profile list.

    Returns:
        Modified DataFrame.
    """
    df = df.copy()
    target_idx = _select_normal_events(df, INJECTION_CFG.device_spoofing_ratio)
    if len(target_idx) == 0:
        return df

    victim = random.choice(profiles)
    spoofed_device = f"DEV-{random.randint(10000, 99999)}"

    new_events: List[Dict[str, Any]] = []
    for i, idx in enumerate(target_idx):
        orig = df.loc[idx]
        event = {
            "event_id": f"ATT-DS-{i:05d}",
            "timestamp": orig["timestamp"],
            "user_id": victim["user_id"],
            "username": victim["username"],
            "event_type": "login",
            "source_ip": fake.ipv4_public(),
            "device_id": spoofed_device,
            "department": victim["department"],
            "resource": "auth_service",
            "success": True,
            "bytes_transferred": 0,
            "is_attack": True,
            "attack_type": "device_spoofing",
        }
        new_events.append(event)

    attack_df = pd.DataFrame(new_events)
    df = pd.concat([df, attack_df], ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def inject_low_and_slow_exfiltration(df: pd.DataFrame, profiles: List[Dict[str, Any]]) -> pd.DataFrame:
    """Inject low-and-slow exfiltration events.

    Small amounts of data are transferred repeatedly over an extended period,
    each transfer staying below typical alert thresholds.

    Args:
        df: Event DataFrame.
        profiles: User profile list.

    Returns:
        Modified DataFrame.
    """
    df = df.copy()
    target_idx = _select_normal_events(df, INJECTION_CFG.low_and_slow_exfiltration_ratio)
    if len(target_idx) == 0:
        return df

    victim = random.choice(profiles)
    new_events: List[Dict[str, Any]] = []

    for i, idx in enumerate(target_idx):
        base_ts = df.loc[idx, "timestamp"]
        # Spread small transfers across 2-6 hours
        n_transfers = random.randint(3, 8)
        for j in range(n_transfers):
            ts = base_ts + timedelta(hours=random.uniform(0, 6))
            small_bytes = int(np.random.uniform(200, 1500))
            event = {
                "event_id": f"ATT-LS-{i:04d}-{j:02d}",
                "timestamp": ts,
                "user_id": victim["user_id"],
                "username": victim["username"],
                "event_type": random.choice(["file_access", "email_access", "shared_folder_access"]),
                "source_ip": victim["primary_ip"],
                "device_id": random.choice(victim["device_ids"]),
                "department": victim["department"],
                "resource": f"/exfil/{fake.file_name()}",
                "success": True,
                "bytes_transferred": small_bytes,
                "is_attack": True,
                "attack_type": "low_and_slow_exfiltration",
            }
            new_events.append(event)

    attack_df = pd.DataFrame(new_events)
    df = pd.concat([df, attack_df], ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def inject_insider_drift(df: pd.DataFrame, profiles: List[Dict[str, Any]]) -> pd.DataFrame:
    """Inject insider-drift events.

    Models a gradually shifting behavioural baseline: a user slowly starts
    accessing new resource types, working unusual hours, and generating
    elevated data transfers over the course of days.

    Args:
        df: Event DataFrame.
        profiles: User profile list.

    Returns:
        Modified DataFrame.
    """
    df = df.copy()
    target_idx = _select_normal_events(df, INJECTION_CFG.insider_drift_ratio)
    if len(target_idx) == 0:
        return df

    victim = random.choice(profiles)
    drift_resources = [
        "db://prod/secrets", "db://admin/users",
        "\\\\fileserver\\executive\\salaries.xlsx",
        "/data/exports/customer_database.csv",
    ]

    new_events: List[Dict[str, Any]] = []
    for i, idx in enumerate(target_idx):
        base_ts = df.loc[idx, "timestamp"]
        # Drift intensifies over a 10-day window
        day_offset = random.randint(0, 10)
        hour_offset = 20 + day_offset * 2  # progressively later hours
        ts = base_ts + timedelta(hours=hour_offset)
        event = {
            "event_id": f"ATT-ID-{i:05d}",
            "timestamp": ts,
            "user_id": victim["user_id"],
            "username": victim["username"],
            "event_type": random.choice(["database_query", "file_access", "shared_folder_access"]),
            "source_ip": victim["primary_ip"],
            "device_id": random.choice(victim["device_ids"]),
            "department": victim["department"],
            "resource": random.choice(drift_resources),
            "success": True,
            "bytes_transferred": int(np.random.lognormal(11 + day_offset * 0.3, 1)),
            "is_attack": True,
            "attack_type": "insider_drift",
        }
        new_events.append(event)

    attack_df = pd.DataFrame(new_events)
    df = pd.concat([df, attack_df], ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def inject_all_attacks(
    df: pd.DataFrame,
    profiles: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Run all seven attack injectors sequentially on the dataset.

    Args:
        df: Normal event DataFrame from the generator.
        profiles: User profile list.

    Returns:
        DataFrame with all attack events injected.
    """
    df = inject_brute_force(df, profiles)
    df = inject_impossible_travel(df, profiles)
    df = inject_credential_stuffing(df, profiles)
    df = inject_lateral_movement(df, profiles)
    df = inject_device_spoofing(df, profiles)
    df = inject_low_and_slow_exfiltration(df, profiles)
    df = inject_insider_drift(df, profiles)
    return df
