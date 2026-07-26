"""
Synthetic enterprise authentication and access log generator.

Creates realistic user behaviour profiles and generates event logs that mimic
a corporate IT environment.  Each user has a behavioural archetype that
controls their working hours, resource access patterns, and device preferences.
"""

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from faker import Faker

from anomaly_detection.config import DATA_CFG, EVENT_TYPES
from anomaly_detection.utils.helpers import compute_entropy

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# ---------------------------------------------------------------------------
# User archetypes define realistic behavioural diversity
# ---------------------------------------------------------------------------
USER_ARCHETYPES: List[Dict[str, Any]] = [
    {
        "name": "early_bird",
        "label": "Early Bird Office Worker",
        "login_hour_mean": 7.5,
        "login_hour_std": 0.8,
        "logout_hour_mean": 17.0,
        "logout_hour_std": 1.0,
        "workday_only": True,
        "event_weights": {
            "login": 0.12, "logout": 0.10, "vpn_login": 0.05,
            "file_access": 0.20, "email_access": 0.18,
            "admin_command": 0.02, "database_query": 0.08,
            "shared_folder_access": 0.10, "usb_usage": 0.03,
            "application_launch": 0.12,
        },
    },
    {
        "name": "night_owl",
        "label": "Night-Shift Operator",
        "login_hour_mean": 21.0,
        "login_hour_std": 1.5,
        "logout_hour_mean": 5.5,
        "logout_hour_std": 1.0,
        "workday_only": False,
        "event_weights": {
            "login": 0.10, "logout": 0.08, "vpn_login": 0.12,
            "file_access": 0.12, "email_access": 0.08,
            "admin_command": 0.10, "database_query": 0.15,
            "shared_folder_access": 0.08, "usb_usage": 0.05,
            "application_launch": 0.12,
        },
    },
    {
        "name": "remote_worker",
        "label": "Remote / VPN Worker",
        "login_hour_mean": 9.0,
        "login_hour_std": 1.5,
        "logout_hour_mean": 18.0,
        "logout_hour_std": 2.0,
        "workday_only": True,
        "event_weights": {
            "login": 0.08, "logout": 0.06, "vpn_login": 0.18,
            "file_access": 0.15, "email_access": 0.20,
            "admin_command": 0.01, "database_query": 0.05,
            "shared_folder_access": 0.12, "usb_usage": 0.01,
            "application_launch": 0.14,
        },
    },
    {
        "name": "sysadmin",
        "label": "System Administrator",
        "login_hour_mean": 8.0,
        "login_hour_std": 2.0,
        "logout_hour_mean": 19.0,
        "logout_hour_std": 2.5,
        "workday_only": False,
        "event_weights": {
            "login": 0.08, "logout": 0.06, "vpn_login": 0.06,
            "file_access": 0.10, "email_access": 0.10,
            "admin_command": 0.22, "database_query": 0.10,
            "shared_folder_access": 0.12, "usb_usage": 0.06,
            "application_launch": 0.10,
        },
    },
    {
        "name": "executive",
        "label": "Executive / Manager",
        "login_hour_mean": 8.5,
        "login_hour_std": 1.0,
        "logout_hour_mean": 18.5,
        "logout_hour_std": 1.5,
        "workday_only": True,
        "event_weights": {
            "login": 0.10, "logout": 0.08, "vpn_login": 0.04,
            "file_access": 0.22, "email_access": 0.25,
            "admin_command": 0.01, "database_query": 0.03,
            "shared_folder_access": 0.12, "usb_usage": 0.02,
            "application_launch": 0.13,
        },
    },
]


def _generate_user_profiles(num_users: int) -> List[Dict[str, Any]]:
    """Create user profile dicts with archetype and persona attributes.

    Args:
        num_users: How many synthetic users to create.

    Returns:
        List of dicts, each representing one user profile.
    """
    profiles: List[Dict[str, Any]] = []
    for uid in range(num_users):
        archetype = random.choice(USER_ARCHETYPES)
        ip_pool_size = random.randint(2, 4)
        profile = {
            "user_id": f"user_{uid:04d}",
            "username": fake.user_name(),
            "department": random.choice([
                "Engineering", "Finance", "HR", "Marketing",
                "Operations", "Legal", "Sales", "IT",
            ]),
            "role": archetype["label"],
            "archetype": archetype["name"],
            "primary_ip": fake.ipv4_public(),
            "ip_pool": [fake.ipv4_public() for _ in range(ip_pool_size)],
            "device_ids": [f"DEV-{random.randint(1000, 9999)}" for _ in range(random.randint(1, 4))],
            "email": fake.email(),
            "archetype_config": archetype,
        }
        profiles.append(profile)
    return profiles


def _pick_event_time(
    archetype_cfg: Dict[str, Any],
    day: datetime,
) -> datetime:
    """Generate a single event timestamp based on the user archetype.

    Args:
        archetype_cfg: The archetype configuration dict.
        day: The calendar date to place the event on.

    Returns:
        A datetime within the allowed working window.
    """
    hour_mean = archetype_cfg["login_hour_mean"]
    hour_std = archetype_cfg["login_hour_std"]
    hour = np.clip(np.random.normal(hour_mean, hour_std), 0, 23)
    minute = np.random.randint(0, 60)
    second = np.random.randint(0, 60)
    return day.replace(hour=int(hour), minute=minute, second=second)


def _is_workday(dt: datetime) -> bool:
    """Return True if *dt* falls on a weekday (Mon-Fri).

    Args:
        dt: A datetime value.

    Returns:
        Boolean indicating weekday.
    """
    return dt.weekday() < 5


def _generate_events_for_user(
    profile: Dict[str, Any],
    start_date: datetime,
    days: int,
    events_per_day: int,
) -> List[Dict[str, Any]]:
    """Generate all events for a single user across the date range.

    Args:
        profile: User profile dictionary.
        start_date: First day of the simulation.
        days: Number of days to simulate.
        events_per_day: Target events per active day.

    Returns:
        List of event dictionaries.
    """
    archetype_cfg = profile["archetype_config"]
    event_weights = archetype_cfg["event_weights"]
    events: List[Dict[str, Any]] = []

    for day_offset in range(days):
        current_day = start_date + timedelta(days=day_offset)

        # Skip non-workdays for workday-only archetypes
        if archetype_cfg["workday_only"] and not _is_workday(current_day):
            continue

        # Decide how many events this user generates today
        daily_count = max(1, int(np.random.poisson(events_per_day)))

        # Determine working window for this day
        base_time = _pick_event_time(archetype_cfg, current_day)
        logout_hour = np.clip(
            np.random.normal(
                archetype_cfg["logout_hour_mean"],
                archetype_cfg["logout_hour_std"],
            ),
            0,
            23,
        )

        for _ in range(daily_count):
            # Select event type based on archetype weights
            event_type = random.choices(
                list(event_weights.keys()),
                weights=list(event_weights.values()),
                k=1,
            )[0]

            # Generate timestamp within the working window
            spread = abs(logout_hour - base_time.hour) + 1
            offset_minutes = random.randint(0, int(spread * 60))
            ts = base_time + timedelta(minutes=offset_minutes)

            # Pick the IP and device for this event
            uses_vpn = event_type == "vpn_login" or random.random() < 0.3
            src_ip = random.choice(profile["ip_pool"]) if uses_vpn else profile["primary_ip"]
            device_id = random.choice(profile["device_ids"])

            event = {
                "timestamp": ts,
                "user_id": profile["user_id"],
                "username": profile["username"],
                "event_type": event_type,
                "source_ip": src_ip,
                "device_id": device_id,
                "department": profile["department"],
                "resource": _random_resource(event_type),
                "success": random.random() > DATA_CFG.normal_login_failure_rate,
                "bytes_transferred": _random_bytes(event_type),
                "is_attack": False,
                "attack_type": "none",
            }
            events.append(event)

    return events


def _random_resource(event_type: str) -> str:
    """Pick a plausible resource string for the given event type.

    Args:
        event_type: One of the EVENT_TYPES constants.

    Returns:
        Resource path or identifier.
    """
    if event_type in ("login", "logout", "vpn_login"):
        return "auth_service"
    if event_type == "file_access":
        return f"/data/files/{fake.file_path(extension='xlsx')}"
    if event_type == "email_access":
        return "exchange_server"
    if event_type == "admin_command":
        return f"/usr/bin/{random.choice(['systemctl', 'useradd', 'chmod', 'visudo', 'docker'])}"
    if event_type == "database_query":
        return f"db://{random.choice(['prod', 'staging', 'analytics'])}/{fake.word()}"
    if event_type == "shared_folder_access":
        return f"\\\\fileserver\\{random.choice(['finance', 'engineering', 'hr', 'shared'])}\\{fake.file_name()}"
    if event_type == "usb_usage":
        return f"/media/usb/{fake.file_name()}"
    if event_type == "application_launch":
        return random.choice([
            "outlook", "teams", "chrome", "vscode", "excel",
            "sap", "jira", "slack", "zoom", "powershell",
        ])
    return "unknown"


def _random_bytes(event_type: str) -> int:
    """Generate a realistic byte-transfer size for the event.

    Args:
        event_type: One of the EVENT_TYPES constants.

    Returns:
        Byte count (0 for non-transfer events).
    """
    if event_type in ("file_access", "usb_usage"):
        return int(np.random.lognormal(mean=12, sigma=2))
    if event_type == "email_access":
        return int(np.random.lognormal(mean=8, sigma=1.5))
    if event_type == "database_query":
        return int(np.random.lognormal(mean=10, sigma=1.8))
    if event_type == "shared_folder_access":
        return int(np.random.lognormal(mean=11, sigma=2))
    return 0


def generate_dataset(
    num_users: Optional[int] = None,
    days: Optional[int] = None,
    events_per_day: Optional[int] = None,
    start_date_str: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Generate the full synthetic enterprise log dataset.

    Args:
        num_users: Override for DATA_CFG.num_users.
        days: Override for DATA_CFG.days.
        events_per_day: Override for DATA_CFG.events_per_user_per_day.
        start_date_str: Override for DATA_CFG.start_date (YYYY-MM-DD).

    Returns:
        Tuple of (events DataFrame, list of user profile dicts).
    """
    n_users = num_users or DATA_CFG.num_users
    n_days = days or DATA_CFG.days
    epd = events_per_day or DATA_CFG.events_per_user_per_day
    start_str = start_date_str or DATA_CFG.start_date
    start_date = datetime.strptime(start_str, "%Y-%m-%d")

    profiles = _generate_user_profiles(n_users)
    all_events: List[Dict[str, Any]] = []

    for profile in profiles:
        user_events = _generate_events_for_user(profile, start_date, n_days, epd)
        all_events.extend(user_events)

    df = pd.DataFrame(all_events)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["event_id"] = [f"EVT-{i:06d}" for i in range(len(df))]

    # Ensure correct column ordering
    columns = [
        "event_id", "timestamp", "user_id", "username", "event_type",
        "source_ip", "device_id", "department", "resource", "success",
        "bytes_transferred", "is_attack", "attack_type",
    ]
    df = df[columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df, profiles
