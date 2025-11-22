"""Tests for the quantum distance-based classifier."""

import numpy as np

from app.dataset import make_dataset
from app.distance_classifier import predict, train_qdistance_classifier
from app.feature_map import build_angle_feature_map


def test_training_and_prediction():
    splits = make_dataset(n_samples=30, noise=0.05, type="moons", test_size=0.3, random_state=1)
    feature_map = build_angle_feature_map(entangle=False)
    model = train_qdistance_classifier(splits.X_train, splits.y_train, feature_map)
    preds = predict(splits.X_test, model)
    # Predictions should be 0/1 and length of test set
    assert len(preds) == len(splits.X_test)
    assert set(np.unique(preds)) <= {0, 1}


def test_distance_prefers_exact_match():
    feature_map = build_angle_feature_map(entangle=False)
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1])
    model = train_qdistance_classifier(X, y, feature_map)
    # Point identical to class 0 centroid should predict 0
    pred = predict(np.array([[0.0, 0.0]]), model)[0]
    assert pred == 0
