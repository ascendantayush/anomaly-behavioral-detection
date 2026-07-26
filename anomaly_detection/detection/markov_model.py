"""
Markov sequence anomaly detector.

Learns the typical event-type transition probabilities from normal user
behaviour and flags events whose preceding context has unusually low
likelihood under the learned model.
"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from anomaly_detection.config import MARKOV_CFG, EVENT_TYPES
from anomaly_detection.utils.helpers import (
    build_transition_counts,
    transition_matrix_to_probabilities,
)


class MarkovDetector:
    """Sequence-based anomaly detector using n-gram transition probabilities.

    Attributes:
        order: Markov chain order (number of preceding states).
        smoothing: Laplace smoothing floor for unseen transitions.
        threshold_percentile: Percentile of normal log-probs used as threshold.
        transition_probs: Learned transition probability table.
        threshold: Log-probability cutoff; below this = anomaly.
        is_fitted: Whether the model has been trained.
    """

    def __init__(
        self,
        order: Optional[int] = None,
        smoothing: Optional[float] = None,
        threshold_percentile: Optional[float] = None,
    ) -> None:
        """Initialise the Markov detector.

        Args:
            order: Markov chain order.
            smoothing: Laplace smoothing constant.
            threshold_percentile: Percentile for anomaly threshold.
        """
        self.order = order if order is not None else MARKOV_CFG.order
        self.smoothing = smoothing if smoothing is not None else MARKOV_CFG.smoothing
        self.threshold_percentile = (
            threshold_percentile if threshold_percentile is not None
            else MARKOV_CFG.threshold_percentile
        )

        self.transition_probs: Dict[Tuple[str, ...], Dict[str, float]] = {}
        self.threshold: float = -math.inf
        self.is_fitted: bool = False

    def _build_sequences(self, df: pd.DataFrame) -> List[List[str]]:
        """Extract per-user event-type sequences from the DataFrame.

        Args:
            df: Event DataFrame with 'user_id' and 'event_type' columns.

        Returns:
            List of event-type sequences (one per user).
        """
        sequences: List[List[str]] = []
        for _, group in df.sort_values(["user_id", "timestamp"]).groupby("user_id"):
            seq = group["event_type"].tolist()
            if len(seq) > self.order:
                sequences.append(seq)
        return sequences

    def fit(self, df: pd.DataFrame) -> "MarkovDetector":
        """Learn transition probabilities from normal event sequences.

        The threshold is calibrated from per-event log-probabilities so it
        matches the scoring scale used by :meth:`predict`.

        Args:
            df: Event DataFrame with 'user_id', 'event_type', 'timestamp'.

        Returns:
            self (for method chaining).
        """
        if "is_attack" in df.columns:
            normal_df = df[df["is_attack"] == False].copy()
        else:
            normal_df = df.copy()
        sequences = self._build_sequences(normal_df)
        counts = build_transition_counts(sequences, order=self.order)
        self.transition_probs = transition_matrix_to_probabilities(
            counts, smoothing=self.smoothing
        )

        all_event_log_probs = self._score_all_events(sequences)
        if len(all_event_log_probs) > 0:
            self.threshold = float(np.percentile(all_event_log_probs, self.threshold_percentile))
        else:
            self.threshold = -10.0

        self.is_fitted = True
        return self

    def _score_all_events(self, sequences: List[List[str]]) -> List[float]:
        """Collect per-event log-probabilities from every training sequence.

        This matches the scoring scale used by :meth:`predict` so the
        threshold is on the same axis as the prediction scores.

        Args:
            sequences: List of event-type sequences.

        Returns:
            Flat list of per-event log-probability scores.
        """
        all_log_probs: List[float] = []
        for seq in sequences:
            if len(seq) <= self.order:
                continue
            for i in range(self.order, len(seq)):
                prefix = tuple(seq[i - self.order : i])
                next_event = seq[i]
                if prefix in self.transition_probs:
                    prob = self.transition_probs[prefix].get(next_event, self.smoothing)
                else:
                    prob = self.smoothing
                all_log_probs.append(math.log(prob + 1e-12))
        return all_log_probs

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict anomaly labels for each event.

        Uses per-user sliding windows: for each event, compute the log-prob
        of the preceding ``order`` events and flag if below threshold.

        Args:
            df: Event DataFrame with 'user_id', 'event_type', 'timestamp'.

        Returns:
            Array of labels: +1 for normal, -1 for anomalous.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        labels = np.ones(len(df), dtype=int)
        scores = self.get_sequence_anomaly_scores(df)

        for i in range(len(df)):
            if scores[i] < self.threshold:
                labels[i] = -1

        return labels

    def get_sequence_anomaly_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Compute per-event anomaly scores based on transition likelihood.

        For each event, the score is the log-probability of observing that
        event given the preceding ``order`` events for the same user.

        Args:
            df: Event DataFrame.

        Returns:
            Array of log-probability scores (higher = more normal).
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        df_sorted = df.sort_values(["user_id", "timestamp"]).reset_index()
        scores = np.zeros(len(df_sorted))

        # Group by user and compute sliding-window log-probs
        for _, group in df_sorted.groupby("user_id"):
            events = group["event_type"].tolist()
            indices = group["index"].tolist()

            for j in range(len(events)):
                if j < self.order:
                    scores[indices[j]] = 0.0
                    continue

                prefix = tuple(events[j - self.order : j])
                next_event = events[j]

                if prefix in self.transition_probs:
                    prob = self.transition_probs[prefix].get(next_event, self.smoothing)
                else:
                    prob = self.smoothing

                scores[indices[j]] = math.log(prob + 1e-12)

        return scores

    def get_normalized_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores normalised to 0-1 (1 = most anomalous).

        Args:
            df: Event DataFrame.

        Returns:
            Array of normalised scores.
        """
        raw = self.get_sequence_anomaly_scores(df)
        # Invert: lower log-prob = more anomalous
        inverted = -raw
        min_val = inverted.min()
        max_val = inverted.max()
        if max_val - min_val == 0:
            return np.zeros_like(inverted)
        return (inverted - min_val) / (max_val - min_val)

    def apply_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Markov predictions and scores to the DataFrame.

        Args:
            df: Event DataFrame.

        Returns:
            DataFrame with added columns: ``markov_label``, ``markov_score``.
        """
        df = df.copy()
        df["markov_label"] = self.predict(df)
        df["markov_score"] = self.get_normalized_scores(df)
        df["markov_is_anomaly"] = (df["markov_label"] == -1).astype(int)
        return df
