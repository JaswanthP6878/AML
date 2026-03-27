"""
models.py
---------
Model training wrappers for AML detection.

Supervised models return a fitted model object.
Unsupervised models return (fitted_model, normalized_scores) tuples so callers
don't need to repeat the score-extraction pattern inline.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None  # type: ignore
    _XGBOOST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize(s: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1]."""
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo + 1e-9)


# ---------------------------------------------------------------------------
# Supervised models
# ---------------------------------------------------------------------------

def train_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> LogisticRegression:
    """Logistic Regression with balanced class weights."""
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Random Forest with balanced class weights."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
):
    """XGBoost with scale_pos_weight to handle class imbalance."""
    if not _XGBOOST_AVAILABLE:
        raise RuntimeError(
            "XGBoost is not available. Fix: brew install libomp, then restart the kernel."
        )
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Unsupervised anomaly detection
# ---------------------------------------------------------------------------

def train_isolation_forest(
    X_train: np.ndarray,
    contamination: float | str = "auto",
    random_state: int = 42,
) -> Tuple[IsolationForest, np.ndarray]:
    """Isolation Forest anomaly detector.

    Returns
    -------
    (fitted_model, scores) where scores are normalized so higher = more anomalous.
    """
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    scores = _normalize(-model.decision_function(X_train))
    return model, scores


def train_lof(
    X_train: np.ndarray,
    n_neighbors: int = 20,
    contamination: float = 0.01,
) -> LocalOutlierFactor:
    """Local Outlier Factor (novelty=True so it supports predict on new data).

    Returns the fitted model only (LOF scores are not stable on training data).
    Use decision_function() on held-out data for evaluation.
    """
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def train_kmeans(
    X_train: np.ndarray,
    n_clusters: int = 10,
    random_state: int = 42,
) -> Tuple[KMeans, np.ndarray]:
    """K-Means anomaly detector via distance to nearest cluster centre.

    Returns
    -------
    (fitted_model, scores) where scores are normalized min-distance (higher = more anomalous).
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    model.fit(X_train)
    dists = np.min(model.transform(X_train), axis=1)
    scores = _normalize(dists)
    return model, scores


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

class Autoencoder(nn.Module):
    """Shallow autoencoder for reconstruction-error anomaly detection.

    Parameters
    ----------
    in_dim     : number of input features
    hidden     : first hidden layer width
    bottleneck : bottleneck (latent) dimension
    """

    def __init__(self, in_dim: int, hidden: int = 32, bottleneck: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, bottleneck), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train_autoencoder(
    X_train: np.ndarray,
    device: str = "cpu",
    epochs: int = 30,
    lr: float = 1e-3,
    hidden: int = 32,
    bottleneck: int = 8,
) -> Tuple[Autoencoder, np.ndarray]:
    """Train Autoencoder and return reconstruction-error anomaly scores.

    Returns
    -------
    (fitted_model, scores) where scores are normalized MSE (higher = more anomalous).
    """
    model = Autoencoder(X_train.shape[1], hidden=hidden, bottleneck=bottleneck).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = ((model(X_t) - X_t) ** 2).mean()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        recon = model(X_t).cpu().numpy()

    mse = np.mean((recon - X_train) ** 2, axis=1)
    scores = _normalize(mse)
    return model, scores


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def compute_ensemble_scores(*score_arrays: np.ndarray) -> np.ndarray:
    """Average multiple normalized anomaly score arrays into an ensemble score.

    Each input array is re-normalized before averaging so different scales
    don't bias the result.

    Parameters
    ----------
    *score_arrays : one or more 1-D score arrays of the same length

    Returns
    -------
    np.ndarray of shape (n_samples,) in [0, 1]
    """
    return np.mean([_normalize(s) for s in score_arrays], axis=0)


# ---------------------------------------------------------------------------
# GNN scaffold (PyTorch Geometric)
# ---------------------------------------------------------------------------

try:
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv

    class GNNScaffold(nn.Module):
        """Two-layer GraphSAGE for transaction graph node classification.

        Graph construction (TODO):
          - Nodes  : unique accounts
          - Edges  : transactions (Account → Account.1)
          - Node features : account-level aggregates from preprocessing

        Usage (skeleton):
          model = GNNScaffold(in_channels=..., hidden=64, out_channels=2)
          optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

          for epoch in range(epochs):
              model.train()
              optimizer.zero_grad()
              out = model(data.x, data.edge_index)
              loss = F.cross_entropy(out[train_mask], data.y[train_mask],
                                     weight=class_weights)
              loss.backward()
              optimizer.step()
        """

        def __init__(self, in_channels: int, hidden: int = 64, out_channels: int = 2):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden)
            self.conv2 = SAGEConv(hidden, out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)
            x = self.conv2(x, edge_index)
            return x

except ImportError:
    GNNScaffold = None  # type: ignore
