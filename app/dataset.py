"""Dataset utilities for the quantum distance-based classifier.

This module creates simple two-class toy datasets that are commonly
used to illustrate non-linear decision boundaries. The functions here
wrap scikit-learn's generators and explain why such data challenges
classical distance metrics and motivates quantum embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split


@dataclass
class DatasetSplits:
    """Container for train and test splits.

    Attributes
    ----------
    X_train, X_test: np.ndarray
        Feature matrices for training and testing.
    y_train, y_test: np.ndarray
        Corresponding label vectors.
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def make_dataset(
    n_samples: int = 150,
    noise: float = 0.1,
    type: str = "moons",
    test_size: float = 0.25,
    random_state: int = 42,
) -> DatasetSplits:
    """Create a simple two-class dataset and split it into train/test sets.

    Parameters
    ----------
    n_samples: int
        Total number of samples to generate.
    noise: float
        Standard deviation of the noise added to the data. Noise makes the
        moons/circles overlap slightly, which highlights the benefit of more
        expressive distance measures.
    type: str
        Which dataset generator to use: "moons" (default) or "circles".
    test_size: float
        Fraction of samples reserved for testing.
    random_state: int
        Seed for reproducibility.

    Returns
    -------
    DatasetSplits
        A dataclass holding NumPy arrays for train and test features/labels.

    Notes
    -----
    Non-linear datasets like moons or circles are perfect for distance-based
    demonstrations. Classical Euclidean distance struggles when classes wrap
    around each other. By computing centroids in the original space and then
    embedding both samples and centroids into a quantum Hilbert space, the
    geometry can change in a way that makes the classes easier to separate.
    """

    # Select the appropriate generator based on the requested type.
    if type == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif type == "circles":
        X, y = make_circles(
            n_samples=n_samples,
            noise=noise,
            factor=0.5,
            random_state=random_state,
        )
    else:
        raise ValueError("type must be either 'moons' or 'circles'")

    # Split the dataset into train and test subsets so we can evaluate generalisation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return DatasetSplits(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


__all__ = ["DatasetSplits", "make_dataset"]
