"""
Central configuration for the AI-Powered Behavioral Anomaly Detection system.

Contains all constants, thresholds, model hyperparameters, risk score weights,
and dashboard settings used across the project.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Attack type identifiers
# ---------------------------------------------------------------------------
ATTACK_TYPES: List[str] = [
    "brute_force",
    "impossible_travel",
    "credential_stuffing",
    "lateral_movement",
    "device_spoofing",
    "low_and_slow_exfiltration",
    "insider_drift",
]

# ---------------------------------------------------------------------------
# Event type identifiers
# ---------------------------------------------------------------------------
EVENT_TYPES: List[str] = [
    "login",
    "logout",
    "vpn_login",
    "file_access",
    "email_access",
    "admin_command",
    "database_query",
    "shared_folder_access",
    "usb_usage",
    "application_launch",
]

# Severity mapping for each attack type  (1 = low, 5 = critical)
ATTACK_SEVERITY: Dict[str, int] = {
    "brute_force": 4,
    "impossible_travel": 5,
    "credential_stuffing": 4,
    "lateral_movement": 5,
    "device_spoofing": 4,
    "low_and_slow_exfiltration": 3,
    "insider_drift": 2,
}


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Parameters controlling synthetic dataset generation."""

    num_users: int = 50
    days: int = 30
    events_per_user_per_day: int = 40
    start_date: str = "2026-06-01"
    normal_login_failure_rate: float = 0.03
    num_source_ips: int = 200
    num_devices: int = 80
    num_file_paths: int = 150
    num_app_names: int = 60
    num_db_names: int = 30
    num_shared_folders: int = 40
    num_email_recipients: int = 200


# ---------------------------------------------------------------------------
# Attack injection ratios  (fraction of total events to inject)
# ---------------------------------------------------------------------------
@dataclass
class InjectionConfig:
    """Fractions of the dataset to replace with attack events."""

    brute_force_ratio: float = 0.03
    impossible_travel_ratio: float = 0.015
    credential_stuffing_ratio: float = 0.025
    lateral_movement_ratio: float = 0.02
    device_spoofing_ratio: float = 0.015
    low_and_slow_exfiltration_ratio: float = 0.03
    insider_drift_ratio: float = 0.02


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
@dataclass
class FeatureConfig:
    """Parameters for feature extraction."""

    time_window_hours: int = 1
    rolling_window_days: int = 7
    sequence_order: int = 1  # Markov chain order


# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------
@dataclass
class IsolationForestConfig:
    """Hyper-parameters for the Isolation Forest model."""

    contamination: float = 0.08
    n_estimators: int = 200
    max_samples: str = "auto"
    random_state: int = 42


# ---------------------------------------------------------------------------
# Markov model
# ---------------------------------------------------------------------------
@dataclass
class MarkovConfig:
    """Parameters for the Markov sequence anomaly detector."""

    order: int = 1
    smoothing: float = 1e-6  # Laplace smoothing floor
    threshold_percentile: float = 5.0


# ---------------------------------------------------------------------------
# Rule engine thresholds
# ---------------------------------------------------------------------------
@dataclass
class RuleConfig:
    """Thresholds used by the rule-engine detectors."""

    brute_force_window_minutes: int = 15
    brute_force_failed_threshold: int = 5
    impossible_travel_min_distance_km: float = 500.0
    impossible_travel_max_hours: float = 0.25
    credential_stuffing_unique_users: int = 5
    credential_stuffing_window_minutes: int = 30
    lateral_movement_unique_resources: int = 18
    lateral_movement_window_hours: int = 2
    device_spoof_new_device_tolerance: float = 0.3
    exfiltration_daily_volume_kb: float = 40000.0
    insider_drift_baseline_days: int = 14
    insider_drift_deviation_threshold: float = 3.5


# ---------------------------------------------------------------------------
# Risk scoring
# ---------------------------------------------------------------------------
@dataclass
class RiskConfig:
    """Weights and thresholds for the composite risk score (0-100)."""

    isolation_forest_weight: float = 0.35
    rule_engine_weight: float = 0.40
    markov_weight: float = 0.25
    risk_threshold_low: int = 25
    risk_threshold_medium: int = 50
    risk_threshold_high: int = 75


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
@dataclass
class DashboardConfig:
    """Streamlit dashboard display settings."""

    page_title: str = "AI Behavioral Anomaly Detection"
    page_icon: str = "🛡️"
    layout: str = "wide"
    max_alerts_display: int = 500
    default_lookback_days: int = 7
    color_normal: str = "#2ecc71"
    color_low: str = "#f1c40f"
    color_medium: str = "#e67e22"
    color_high: str = "#e74c3c"
    color_critical: str = "#8e44ad"
    color_palette: List[str] = field(default_factory=lambda: [
        "#3498db", "#e74c3c", "#2ecc71", "#f39c12",
        "#9b59b6", "#1abc9c", "#e67e22",
    ])


# ---------------------------------------------------------------------------
# Singleton-style instances for easy import
# ---------------------------------------------------------------------------
DATA_CFG = DataConfig()
INJECTION_CFG = InjectionConfig()
FEATURE_CFG = FeatureConfig()
ISO_CFG = IsolationForestConfig()
MARKOV_CFG = MarkovConfig()
RULE_CFG = RuleConfig()
RISK_CFG = RiskConfig()
DASH_CFG = DashboardConfig()
