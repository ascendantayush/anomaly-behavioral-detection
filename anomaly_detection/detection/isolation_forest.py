"""
Isolation Forest anomaly detector.

Wraps scikit-learn's ``IsolationForest`` with a clean interface for fitting,
predicting, and retrieving continuous anomaly scores.  The model is trained
on the full feature matrix and labels each event as normal (-1) or anomalous (+1).
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from anomaly_detection.config import ISO_CFG


class IsolationForestDetector:
    """Unsupervised anomaly detector based on Isolation Forest.

    Attributes:
        model: The underlying scikit-learn IsolationForest instance.
        scaler: StandardScaler used to normalise features before fitting.
        feature_names: List of feature column names used during training.
        is_fitted: Whether the model has been trained.
    """

    def __init__(
        self,
        contamination: Optional[float] = None,
        n_estimators: Optional[int] = None,
        max_samples: Optional[str] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialise the detector with configurable hyper-parameters.

        Args:
            contamination: Expected fraction of anomalies in the training data.
            n_estimators: Number of isolation trees in the ensemble.
            max_samples: Number of samples drawn to build each tree.
            random_state: Seed for reproducibility.
        """
        self.contamination = contamination if contamination is not None else ISO_CFG.contamination
        self.n_estimators = n_estimators if n_estimators is not None else ISO_CFG.n_estimators
        self.max_samples = max_samples if max_samples is not None else ISO_CFG.max_samples
        self.random_state = random_state if random_state is not None else ISO_CFG.random_state

        self.model: Optional[IsolationForest] = None
        self.scaler: StandardScaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
    ) -> "IsolationForestDetector":
        """Fit the Isolation Forest on the provided feature matrix.

        Args:
            X: DataFrame of numeric features (rows = events, cols = features).
            feature_names: Optional list of feature names; inferred from *X* if None.

        Returns:
            self (for method chaining).
        """
        self.feature_names = feature_names if feature_names is not None else list(X.columns)

        # Handle NaN / inf by filling with 0
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        X_scaled = self.scaler.fit_transform(X_clean)

        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
        )
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict anomaly labels for the given feature matrix.

        Args:
            X: DataFrame of numeric features (same columns as training data).

        Returns:
            Array of labels: +1 for normal, -1 for anomalous.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_scaled = self.scaler.transform(X_clean)
        return self.model.predict(X_scaled)

    def get_anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Return continuous anomaly scores (lower = more anomalous).

        Args:
            X: DataFrame of numeric features.

        Returns:
            Array of anomaly scores (negative values are more anomalous).
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_scaled = self.scaler.transform(X_clean)
        return self.model.score_samples(X_scaled)

    def get_normalized_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores normalised to 0-1 range (1 = most anomalous).

        Args:
            X: DataFrame of numeric features.

        Returns:
            Array of normalised scores in [0, 1].
        """
        raw = self.get_anomaly_scores(X)
        # Invert so higher = more anomalous, then min-max scale
        inverted = -raw
        min_val = inverted.min()
        max_val = inverted.max()
        if max_val - min_val == 0:
            return np.zeros_like(inverted)
        return (inverted - min_val) / (max_val - min_val)

    def apply_to_dataframe(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
    ) -> pd.DataFrame:
        """Add Isolation Forest predictions and scores to the DataFrame.

        Args:
            df: DataFrame to augment (will be copied).
            feature_columns: Column names to use as model input.

        Returns:
            DataFrame with added columns: ``iso_anomaly_label`` and ``iso_anomaly_score``.
        """
        df = df.copy()
        X = df[feature_columns]

        df["iso_anomaly_label"] = self.predict(X)
        df["iso_anomaly_score"] = self.get_normalized_scores(X)
        df["iso_is_anomaly"] = (df["iso_anomaly_label"] == -1).astype(int)

        return df
