"""Tests for quantum feature maps."""

from qiskit import QuantumCircuit

from app.feature_map import build_angle_feature_map, embed_point


def test_build_angle_feature_map():
    circuit = build_angle_feature_map()
    assert isinstance(circuit, QuantumCircuit)
    # Expect two qubits and at least two operations (RY and RZ)
    assert circuit.num_qubits == 2
    assert len(circuit.data) >= 2


def test_embed_point_runs():
    circuit = build_angle_feature_map()
    state = embed_point(circuit, [0.1, -0.2])
    # Statevector has dimension 4 for two qubits
    assert len(state.data) == 4
