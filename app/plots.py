"""Visualization helpers that only produce SVG outputs.

All plotting functions save vector graphics (.svg) so the repository
remains text-friendly and easy to review. The visuals illustrate the
original data layout, how the quantum kernel measures similarity, and how
the decision boundary shifts.
"""

from __future__ import annotations

import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Ensure default style is consistent and the backend writes SVG files.
sns.set_theme(context="notebook", style="whitegrid")


def _ensure_dir(path: str) -> None:
    """Create the parent directory for an output file if needed."""

    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_original_data(X: np.ndarray, y: np.ndarray, output_path: str) -> None:
    """Plot the raw dataset and save as SVG."""

    _ensure_dir(output_path)
    plt.figure(figsize=(5, 4))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="black")
    plt.title("Original data (non-linear layout)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_quantum_distance_heatmap(kernel: np.ndarray, output_path: str) -> None:
    """Visualize fidelity-based similarities as a heatmap."""

    _ensure_dir(output_path)
    plt.figure(figsize=(5, 4))
    sns.heatmap(kernel, cmap="mako", square=True, cbar_kws={"label": "Fidelity"})
    plt.title("Quantum kernel (fidelity) between samples")
    plt.xlabel("Sample index")
    plt.ylabel("Sample index")
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_decision_regions(
    X: np.ndarray,
    y: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    output_path: str,
    grid_step: float = 0.05,
) -> None:
    """Plot decision regions for a classifier using a dense mesh grid."""

    _ensure_dir(output_path)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step),
        np.arange(y_min, y_max, grid_step),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_pred = predict_fn(grid_points).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, grid_pred, cmap="coolwarm", alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="black")
    plt.title("Decision regions (quantum distance)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


__all__ = [
    "plot_original_data",
    "plot_quantum_distance_heatmap",
    "plot_decision_regions",
]
