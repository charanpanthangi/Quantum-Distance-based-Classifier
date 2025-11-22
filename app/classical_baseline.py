"""Classical nearest centroid baseline classifier.

This module mirrors the quantum classifier but works directly in the
original feature space using Euclidean distances. It provides a simple
reference to highlight how the quantum embedding can change the geometry
of the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


def _compute_centroids(X: np.ndarray, y: np.ndarray) -> Dict[int, np.ndarray]:
    """Compute centroids in the classical space for each label."""

    centroids: Dict[int, np.ndarray] = {}
    for label in np.unique(y):
        centroids[int(label)] = np.mean(X[y == label], axis=0)
    return centroids


@dataclass
class ClassicalCentroidModel:
    """Stores centroids for the classical baseline."""

    centroids: Dict[int, np.ndarray]


def train_classical_centroid(X: np.ndarray, y: np.ndarray) -> ClassicalCentroidModel:
    """Fit the classical nearest centroid model."""

    centroids = _compute_centroids(X, y)
    return ClassicalCentroidModel(centroids=centroids)


def predict(X: np.ndarray, model: ClassicalCentroidModel) -> np.ndarray:
    """Predict labels by choosing the nearest centroid using Euclidean distance."""

    predictions: List[int] = []
    for x in X:
        distances = {}
        for label, centroid in model.centroids.items():
            # Euclidean distance in the original feature space.
            distances[label] = np.linalg.norm(x - centroid)
        best_label = min(distances, key=distances.get)
        predictions.append(int(best_label))
    return np.array(predictions)


__all__ = ["ClassicalCentroidModel", "train_classical_centroid", "predict"]
