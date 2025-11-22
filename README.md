# Quantum Distance-based Classifier (QNearestCentroid)

## What This Project Does
This repository shows a tiny example of a quantum nearest-centroid classifier. It:
- makes a simple two-class dataset (moons or circles)
- computes the classical centroids for each class
- embeds points and centroids into quantum states with rotation gates
- measures distance with **1 – |⟨φ(x)|φ(centroid)⟩|²** (fidelity-based distance)
- compares the quantum classifier to a classical centroid baseline

## Why Quantum Helps
- Quantum feature maps reshape the geometry of the data.
- Entanglement lets the embedding capture interactions between features.
- Distances in Hilbert space can separate overlapping regions that are tangled in the original space.

## Why SVG Instead of PNG
GitHub’s CODEX interface cannot preview PNG/JPG and shows
“Binary files are not supported.”
To avoid this, all plots in this repository are stored as lightweight SVG
vector images. SVGs render cleanly and are text-based, which avoids diff issues.

## How It Works (Plain English)
1. Compute the centroid of each class in the classical feature space.
2. Build a quantum circuit that turns two real numbers into a quantum state.
3. Embed both samples and centroids with that circuit.
4. Measure similarity with fidelity (state overlap) and turn it into a distance.
5. Pick the class whose centroid is closest; compare with the classical centroid model.

## How to Run
```bash
pip install -r requirements.txt
python app/main.py --n-samples 200 --dataset moons --noise 0.1 --entangle
```

## What You Should See
- Printed quantum vs classical accuracy on the test split.
- SVG plots saved into `examples/`:
  - `original_data.svg`
  - `quantum_distance_heatmap.svg`
  - `decision_regions.svg`

## Future Extensions
- Multi-class QNearestCentroid
- Different feature maps (e.g., ZZFeatureMap)
- Running on real quantum hardware
- Kernel PCA or SVMs using the quantum kernel
