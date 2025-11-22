"""Quantum distance-based nearest centroid classifier.

This module implements a minimal QNearestCentroid model. It computes
class centroids in the original feature space, embeds them into quantum
states using a chosen feature map, and classifies new points by comparing
state fidelities. The core idea is that quantum embeddings can reshape
geometry so that overlapping clusters become more separable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .feature_map import embed_point


@dataclass
class QDistanceModel:
    """Stores the learned components of the quantum classifier."""

    feature_map: QuantumCircuit
    centroids: Dict[int, np.ndarray]
    centroid_states: Dict[int, Statevector]


def _compute_centroids(X: np.ndarray, y: np.ndarray) -> Dict[int, np.ndarray]:
    """Compute classical centroids for each class label."""

    centroids: Dict[int, np.ndarray] = {}
    for label in np.unique(y):
        # Mean across samples belonging to the current class.
        centroids[int(label)] = np.mean(X[y == label], axis=0)
    return centroids


def _fidelity(state_a: Statevector, state_b: Statevector) -> float:
    """Compute fidelity |⟨a|b⟩|^2 between two statevectors."""

    overlap = np.vdot(state_a.data, state_b.data)
    return float(np.abs(overlap) ** 2)


def train_qdistance_classifier(
    X: np.ndarray, y: np.ndarray, feature_map: QuantumCircuit
) -> QDistanceModel:
    """Train the quantum distance-based nearest centroid classifier.

    Parameters
    ----------
    X: np.ndarray
        Training features with shape (n_samples, 2).
    y: np.ndarray
        Integer class labels.
    feature_map: QuantumCircuit
        The feature map used to embed both samples and centroids.

    Returns
    -------
    QDistanceModel
        A lightweight structure containing the feature map, centroids, and their
        quantum embeddings.
    """

    centroids = _compute_centroids(X, y)

    centroid_states: Dict[int, Statevector] = {}
    for label, centroid in centroids.items():
        centroid_states[label] = embed_point(feature_map, centroid)

    return QDistanceModel(feature_map=feature_map, centroids=centroids, centroid_states=centroid_states)


def predict(X: np.ndarray, model: QDistanceModel) -> np.ndarray:
    """Predict class labels for new samples using quantum distances.

    The prediction rule minimises the quantum distance:
    distance(x, centroid) = 1 - fidelity(|phi(x)>, |phi(centroid)>).
    """

    predictions: List[int] = []
    for x in X:
        # Embed the sample into the quantum state space.
        x_state = embed_point(model.feature_map, x)

        # Compute distances to each centroid state and pick the smallest.
        distances = {}
        for label, c_state in model.centroid_states.items():
            fidelity = _fidelity(x_state, c_state)
            distances[label] = 1 - fidelity
        best_label = min(distances, key=distances.get)
        predictions.append(int(best_label))

    return np.array(predictions)


def compute_quantum_kernel(X: np.ndarray, model: QDistanceModel) -> np.ndarray:
    """Compute a simple kernel matrix using fidelity between embedded samples."""

    n = len(X)
    kernel = np.zeros((n, n))
    embedded_states = [embed_point(model.feature_map, x) for x in X]

    for i in range(n):
        for j in range(n):
            kernel[i, j] = _fidelity(embedded_states[i], embedded_states[j])
    return kernel


__all__ = [
    "QDistanceModel",
    "train_qdistance_classifier",
    "predict",
    "compute_quantum_kernel",
]
