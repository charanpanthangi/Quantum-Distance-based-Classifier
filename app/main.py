"""Command-line entrypoint for the quantum distance-based classifier.

Running this script will train both the quantum and classical centroid
models, evaluate them on a held-out test set, and generate SVG plots that
illustrate the data, kernel heatmap, and decision regions.
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np

from .dataset import make_dataset
from .distance_classifier import compute_quantum_kernel, predict as q_predict, train_qdistance_classifier
from .feature_map import build_angle_feature_map
from .classical_baseline import predict as c_predict, train_classical_centroid
from .plots import plot_original_data, plot_quantum_distance_heatmap, plot_decision_regions



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for configuring the run."""

    parser = argparse.ArgumentParser(description="Quantum distance-based nearest centroid classifier")
    parser.add_argument("--feature-map", choices=["angle"], default="angle", help="Feature map to use (angle encoding)")
    parser.add_argument("--n-samples", type=int, default=200, help="Number of samples to generate")
    parser.add_argument("--noise", type=float, default=0.1, help="Noise level for the dataset")
    parser.add_argument("--dataset", choices=["moons", "circles"], default="moons", help="Dataset type")
    parser.add_argument("--entangle", action="store_true", help="Add entanglement to the feature map")
    return parser.parse_args()



def main() -> None:
    """Execute the full workflow: data, training, evaluation, and plots."""

    args = parse_args()

    # 1. Load dataset.
    splits = make_dataset(n_samples=args.n_samples, noise=args.noise, type=args.dataset)
    X_train, X_test, y_train, y_test = splits.X_train, splits.X_test, splits.y_train, splits.y_test

    # 2. Build feature map.
    feature_map = build_angle_feature_map(entangle=args.entangle)

    # 3. Train quantum classifier.
    q_model = train_qdistance_classifier(X_train, y_train, feature_map)

    # 4. Train classical baseline.
    c_model = train_classical_centroid(X_train, y_train)

    # 5. Evaluate.
    q_train_pred = q_predict(X_train, q_model)
    q_test_pred = q_predict(X_test, q_model)
    c_train_pred = c_predict(X_train, c_model)
    c_test_pred = c_predict(X_test, c_model)

    q_train_acc = np.mean(q_train_pred == y_train)
    q_test_acc = np.mean(q_test_pred == y_test)
    c_train_acc = np.mean(c_train_pred == y_train)
    c_test_acc = np.mean(c_test_pred == y_test)

    # 6. Generate visualisations (SVG only).
    os.makedirs("examples", exist_ok=True)
    plot_original_data(np.vstack([X_train, X_test]), np.hstack([y_train, y_test]), "examples/original_data.svg")

    kernel = compute_quantum_kernel(X_train[:50], q_model)  # subset for speed
    plot_quantum_distance_heatmap(kernel, "examples/quantum_distance_heatmap.svg")

    plot_decision_regions(X_train, y_train, lambda pts: q_predict(pts, q_model), "examples/decision_regions.svg")

    # 7. Print summary.
    print("Quantum classifier accuracy: train={:.3f}, test={:.3f}".format(q_train_acc, q_test_acc))
    print("Classical centroid accuracy: train={:.3f}, test={:.3f}".format(c_train_acc, c_test_acc))
    print("SVG plots saved to examples/")


if __name__ == "__main__":
    main()
