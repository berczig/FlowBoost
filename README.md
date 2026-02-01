# FlowBoost

FlowBoost is a codebase for **rare / extremal mathematical structure discovery** via **closed-loop Simulation-Based Optimization (SBO)**. It combines:

1. **Conditional Flow Matching (CFM)** in continuous configuration space to learn a generative policy over geometric objects,
2. **Geometry-Aware Sampling (GAS)** to enforce hard feasibility constraints *during* sampling (not only post-hoc), and
3. **Reward-Guided Fine-Tuning (RG-CFM)** with **action exploration** and a **teacher–student consistency / self-distillation** trust region to steer generation toward higher objective values while avoiding collapse.

The pipeline is paired with a strong local search/refinement backend (**SRP: Stochastic Relaxation with Perturbations**), used both to generate training data and to “push/polish” generated candidates.

---

## Pipeline

At a high level, FlowBoost casts geometric optimization as a **learned stochastic policy** that is updated online using objective feedback.

```text
      (training data)                          (samples)
+----------------------+                 +----------------------+
|  Local Search (SRP)  |  ---- D_k ----> |   CFM Training       |
+----------------------+                 |   (learn v_theta)    |
           ^                             +----------------------+
           |                                       |
           | (refine/push)                         | (GAS: feasible sampling)
           |                                       v
+----------------------+                 +----------------------+
|  Push / Polish (SRP) |                 |     GAS Sampler      |
+----------------------+                 +----------------------+
           ^                                       |
           | (select elites, compute J / R)        | (Action Exploration E(x'|x))
           |                                       v
+----------------------+                 +----------------------+
|  Selection & Eval    | ---- R(x) --->  |  RG-CFM Fine-Tuning   |
| (objective + checks) |                 | (reward-weighted FM + |
+----------------------+                 |  consistency to v_ref)|
           |                             +----------------------+
           +-----------------------------(update theta; iterate)
```

**Key differences vs open-loop boosters (e.g., “retrain on elites”):**

* rewards flow **directly into parameter updates** (reward-weighted flow matching),
* sampling is **constraint-aware** (projection/corrections during generation),
* exploration is **structured** (geometry-aware action exploration),
* a Self-Distillation (SD) velocity field provides a **trust region** or teacher (consistency regularization) to prevent generative collapse.

---

## Implemented problem modules

This repository contains pipelines for multiple continuous geometric optimization problems, including:

* **Sphere packing in a hypercube** (`flow_boost/spheres_in_hypercube/`)
* **Circle packing in the unit square** incl. **max-sum-of-radii** variants (`flow_boost/circles_in_square/`)
* **Heilbronn triangle problem** (`flow_boost/heilbronn_square/`)
* **Star discrepancy minimization** (`flow_boost/star_discrepancy/`)

Additional experimental modules may exist (e.g. Tammes) and are included as research code.

---

## Repository layout

* `flow_boost/spheres_in_hypercube/` — d-dimensional sphere packing experiments in a box
* `flow_boost/circles_in_square/` — circle packing + sum-of-radii variants
* `flow_boost/heilbronn_square/` — Heilbronn triangle problem in the unit square
* `flow_boost/tammes_problem/` — Tammes-style point configurations on the sphere (experimental)
* `flow_boost/star_discrepancy/` — star discrepancy optimization
* `flow_boost/result_Viewer/` — NiceGUI viewer for `.pt` datasets/models
* `flow_boost/output/` — example artifacts (datasets, plots, logs)
* `config.cfg` — shared configuration for all pipelines
* `minimal_requirements.txt` — minimal Python dependencies

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r minimal_requirements.txt
pip install -e .
```

> Note: install PyTorch separately if you need a specific CUDA build.

---

## Configuration

`flow_boost` loads `config.cfg` on import. Each module has dedicated sections
(e.g. `[circle_packing_SRP]`, `[heilbronn_flow]`, etc.).

Common knob families you will see across pipelines:

* **Local search (SRP / push):** step sizes, annealing schedules, number of iterations, tolerances
* **CFM training:** batch size, epochs, optimizer settings, time sampling strategy
* **GAS sampling:** `gas_steps`, projection/proximal parameters, repair tolerances
* **Reward guidance (RG-CFM):** reward temperature, clipping, number of fine-tuning steps/epochs
* **Action exploration:** magnitude, mixing between contact- and wall-driven directions, repair after move
* **Self-distillation regularization:** consistency coefficient controlling the trust region

---

## Running experiments

Most modules expose a pipeline script that reads `config.cfg` and writes outputs under
`flow_boost/output/` by default.

Examples:

```bash
python flow_boost/circles_in_square/pipeline_sumradii.py
python flow_boost/heilbronn_square/pipeline_heilbronn.py
python flow_boost/star_discrepancy/pipeline_star_discrepancy.py
```

To inspect datasets/models with the GUI:

```bash
python flow_boost/result_Viewer/result_viewer.py
```

---

## Reference

If you use our method, please cite the FlowBoost paper.

```bibtex
@article{berczi2026flow,
  title={Flow-based Extremal Mathematical Structure Discovery},
  author={B{\'e}rczi, Gergely and Hashemi, Baran and Kl{\"u}ver, Jonas},
  journal={arXiv preprint arXiv:2601.18005},
  year={2026}
}
```
