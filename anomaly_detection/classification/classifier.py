"""
Multi-signal attack classifier and risk scorer.

Combines outputs from the Isolation Forest, rule engine, and Markov model
to produce a final attack classification and a composite risk score (0-100)
for each event.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from anomaly_detection.config import (
    ATTACK_SEVERITY,
    ATTACK_TYPES,
    RISK_CFG,
)
from anomaly_detection.utils.helpers import safe_divide


class AttackClassifier:
    """Combine detector signals and produce final classification + risk score.

    Attributes:
        if_weight: Weight for Isolation Forest signal.
        rule_weight: Weight for rule engine signal.
        markov_weight: Weight for Markov model signal.
    """

    def __init__(
        self,
        if_weight: Optional[float] = None,
        rule_weight: Optional[float] = None,
        markov_weight: Optional[float] = None,
    ) -> None:
        """Initialise the classifier with configurable weights.

        Args:
            if_weight: Isolation Forest contribution weight.
            rule_weight: Rule engine contribution weight.
            markov_weight: Markov model contribution weight.
        """
        self.if_weight = if_weight if if_weight is not None else RISK_CFG.isolation_forest_weight
        self.rule_weight = rule_weight if rule_weight is not None else RISK_CFG.rule_engine_weight
        self.markov_weight = markov_weight if markov_weight is not None else RISK_CFG.markov_weight

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify each event and compute a composite risk score.

        Expects the DataFrame to already contain detector output columns:
        - ``iso_is_anomaly``, ``iso_anomaly_score``
        - ``rule_anomaly``, ``rule_attack_type``, ``rule_severity``
        - ``markov_is_anomaly``, ``markov_score``

        Adds:
        - ``predicted_attack_type``: final classification label
        - ``risk_score``: 0-100 composite score
        - ``risk_level``: LOW / MEDIUM / HIGH / CRITICAL

        Args:
            df: DataFrame with detector outputs.

        Returns:
            Augmented DataFrame.
        """
        df = df.copy()

        # --- Compute the composite anomaly score (0-1) ---
        df["composite_score"] = (
            self.if_weight * df["iso_anomaly_score"].fillna(0.0)
            + self.rule_weight * df["rule_anomaly"].fillna(0.0).astype(float)
            + self.markov_weight * df["markov_score"].fillna(0.0)
        )

        # Scale composite to 0-100
        cmin = df["composite_score"].min()
        cmax = df["composite_score"].max()
        if cmax - cmin > 0:
            df["risk_score"] = ((df["composite_score"] - cmin) / (cmax - cmin) * 100).round(1)
        else:
            df["risk_score"] = 0.0

        # --- Determine predicted attack type ---
        df["predicted_attack_type"] = "normal"

        # Priority 1: Rule engine provides the most specific signal
        rule_classified = df["rule_attack_type"] != "none"
        df.loc[rule_classified, "predicted_attack_type"] = df.loc[rule_classified, "rule_attack_type"]

        # Priority 2: For events not classified by rules, use anomaly scores
        unclassified = df["predicted_attack_type"] == "normal"
        anomaly_mask = unclassified & (
            (df["iso_is_anomaly"] == 1) | (df["markov_is_anomaly"] == 1)
        )

        # Heuristic classification for unclassified anomalies
        df.loc[anomaly_mask, "predicted_attack_type"] = df.loc[anomaly_mask].apply(
            self._heuristic_classify, axis=1
        )

        # --- Risk level ---
        df["risk_level"] = df["risk_score"].apply(self._risk_level_label)

        # --- Human-readable risk colour ---
        df["risk_color"] = df["risk_score"].apply(self._risk_color)

        return df

    def _heuristic_classify(self, row: pd.Series) -> str:
        """Apply heuristic rules to classify an unclassified anomaly.

        Uses available feature signals to guess the most likely attack type.

        Args:
            row: A single DataFrame row.

        Returns:
            Predicted attack type string.
        """
        # Check for brute-force indicators
        login_fail_rate = row.get("login_failure_rate", 0)
        if login_fail_rate > 0.4:
            return "brute_force"

        # Check for impossible travel (geo columns may not exist)
        if row.get("is_new_device", 0) == 1 and row.get("iso_anomaly_score", 0) > 0.7:
            return "device_spoofing"

        # Check for lateral movement (high resource diversity)
        unique_resources = row.get("unique_resources", 0)
        if unique_resources > 5:
            return "lateral_movement"

        # High byte transfer with low score = possible exfiltration
        if row.get("bytes_transferred", 0) > 10000:
            return "low_and_slow_exfiltration"

        # Night activity
        if row.get("is_night", 0) == 1:
            return "insider_drift"

        # Default: mark as generic anomaly with the highest-scoring detector
        return "brute_force"

    def _risk_level_label(self, score: float) -> str:
        """Map risk score to a level label.

        Args:
            score: Risk score 0-100.

        Returns:
            Level label string.
        """
        if score < RISK_CFG.risk_threshold_low:
            return "LOW"
        if score < RISK_CFG.risk_threshold_medium:
            return "MEDIUM"
        if score < RISK_CFG.risk_threshold_high:
            return "HIGH"
        return "CRITICAL"

    def _risk_color(self, score: float) -> str:
        """Map risk score to a display colour.

        Args:
            score: Risk score 0-100.

        Returns:
            Hex colour code.
        """
        if score < RISK_CFG.risk_threshold_low:
            return "#2ecc71"
        if score < RISK_CFG.risk_threshold_medium:
            return "#f1c40f"
        if score < RISK_CFG.risk_threshold_high:
            return "#e67e22"
        return "#e74c3c"

    def get_attack_summary(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Summarise detected attacks by type.

        Args:
            df: Classified DataFrame.

        Returns:
            Dict mapping attack type to summary statistics.
        """
        attacks = df[df["predicted_attack_type"] != "normal"]
        summary: Dict[str, Dict] = {}

        for atype in ATTACK_TYPES:
            subset = attacks[attacks["predicted_attack_type"] == atype]
            if len(subset) == 0:
                continue
            summary[atype] = {
                "count": len(subset),
                "avg_risk_score": round(subset["risk_score"].mean(), 1),
                "max_risk_score": round(subset["risk_score"].max(), 1),
                "unique_users": subset["user_id"].nunique(),
                "unique_ips": subset["source_ip"].nunique(),
                "severity": ATTACK_SEVERITY.get(atype, 0),
            }

        summary["total"] = {
            "count": len(attacks),
            "avg_risk_score": round(attacks["risk_score"].mean(), 1) if len(attacks) > 0 else 0,
            "unique_users": attacks["user_id"].nunique() if len(attacks) > 0 else 0,
        }

        return summary
