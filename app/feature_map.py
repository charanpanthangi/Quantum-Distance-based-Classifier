"""Quantum feature maps used for embedding classical data.

The feature map converts 2D classical inputs into quantum states using
simple rotation gates. Once in the Hilbert space, distances are computed
as state overlaps (fidelity), which can capture relationships that are
non-linear in the original space.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def build_angle_feature_map(entangle: bool = True) -> QuantumCircuit:
    """Create a basic angle-encoding feature map.

    Parameters
    ----------
    entangle: bool
        If ``True``, add a controlled-Z gate to correlate the two qubits. This
        light entanglement helps the embedding capture feature interactions.

    Returns
    -------
    QuantumCircuit
        A two-qubit circuit that expects two angles (x1, x2) as inputs.

    Notes
    -----
    The circuit applies RY(x1) and RZ(x2) rotations to encode the classical
    features. Optional entanglement makes the resulting state depend on both
    features jointly, which reshapes distances between points once they are in
    the quantum state space.
    """

    circuit = QuantumCircuit(2)

    # Placeholders for parameters are simply the input values; we apply the
    # rotations directly when embedding a specific point.
    circuit.ry(0, 0)
    circuit.rz(0, 1)

    if entangle:
        # A controlled-Z gate introduces a simple form of entanglement that can
        # increase the expressiveness of the embedding.
        circuit.cz(0, 1)

    return circuit


def _apply_angles(circuit: QuantumCircuit, angles: Iterable[float]) -> QuantumCircuit:
    """Attach concrete angles to a copy of the feature map.

    This helper avoids mutating the base template circuit and keeps the input
    handling in a single place.
    """

    if len(angles) != 2:
        raise ValueError("Feature map expects exactly two features (x1, x2)")

    # Create a shallow copy to avoid side effects when embedding multiple points.
    populated = circuit.copy()
    x1, x2 = angles
    populated.data = []
    populated.ry(float(x1), 0)
    populated.rz(float(x2), 1)
    if any(instr.operation.name == 'cz' for instr in circuit.data):
        populated.cz(0, 1)
    return populated


def embed_point(feature_map: QuantumCircuit, x: Iterable[float]) -> Statevector:
    """Encode a single 2D point into a quantum statevector.

    Parameters
    ----------
    feature_map: QuantumCircuit
        The base feature map template.
    x: Iterable[float]
        The classical coordinates to encode.

    Returns
    -------
    Statevector
        The quantum state |phi(x)\rangle representing the embedded point.

    Notes
    -----
    Distances in the Hilbert space are computed via fidelity, which measures
    how similar two quantum states are. By embedding both data points and
    class centroids, we can compare them in a space where overlapping clusters
    may become more separable.
    """

    circuit = _apply_angles(feature_map, x)
    # Statevector.from_instruction executes the circuit and returns the final state.
    return Statevector.from_instruction(circuit)


__all__ = ["build_angle_feature_map", "embed_point"]
