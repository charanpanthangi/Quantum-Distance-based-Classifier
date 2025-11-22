"""Tests for the classical centroid baseline."""

import numpy as np

from app.classical_baseline import predict, train_classical_centroid


def test_classical_centroid_predicts():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1])
    model = train_classical_centroid(X, y)
    preds = predict(X, model)
    assert list(preds) == [0, 1]


def test_classical_handles_nearby_points():
    X = np.array([[0.0, 0.0], [0.2, 0.1], [1.0, 1.0]])
    y = np.array([0, 0, 1])
    model = train_classical_centroid(X, y)
    pred = predict(np.array([[0.15, 0.1]]), model)[0]
    assert pred == 0
