"""Tests for dataset generation utilities."""

import numpy as np

from app.dataset import make_dataset


def test_make_dataset_shapes():
    splits = make_dataset(n_samples=20, noise=0.0, type="moons", test_size=0.2, random_state=0)
    assert splits.X_train.shape[1] == 2
    assert len(splits.X_train) + len(splits.X_test) == 20
    assert set(np.unique(splits.y_train)) <= {0, 1}


def test_invalid_type_raises():
    try:
        make_dataset(type="unknown")
    except ValueError:
        assert True
    else:
        assert False
