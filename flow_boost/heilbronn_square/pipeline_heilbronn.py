import os

# Allow running this file directly (e.g. `python path/to/pipeline_heilbronn.py`)
# without requiring an editable install. When executed as a script, Python only
# adds this file's directory to sys.path, so the repo root isn't visible.
if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from rich.console import Console

import flow_boost
from flow_boost import cfg

"""
Heilbronn pipeline: (TrainSetGen or Push) -> (Train+Sample) -> Push -> (Re-train+Sample) -> Push -> ...

Config sections used:

[heilbronn_SRP]
mode = training_set_gen | final_push
output_dir
final_push_output
final_push_input

[heilbronn_flow]
mode = training_and_sampling | retrain_and_sampling | sampling_only
dataset_path
resume_model_path

[heilbronn_square_pipeline]
iterations
start_at_step = start | push | train_and_sampling | retrain_and_sampling
"""

# --------------------------
# cfg helpers
# --------------------------
def _ensure_section(section: str):
    if not cfg.has_section(section):
        cfg.add_section(section)

def _set_cfg(section: str, key: str, value):
    _ensure_section(section)
    cfg.set(section, key, str(value))

def _get_cfg(section: str, key: str, fallback=None):
    try:
        if isinstance(fallback, bool):
            return cfg.getboolean(section, key, fallback=fallback)
        if isinstance(fallback, int):
            return cfg.getint(section, key, fallback=fallback)
        if isinstance(fallback, float):
            return cfg.getfloat(section, key, fallback=fallback)
        if fallback is None:
            return cfg.get(section, key)
        return cfg.get(section, key, fallback=fallback)
    except Exception:
        return fallback

# --------------------------
# pipeline state
# --------------------------
class PipelineState:
    model_path: str = ""
    samples_path: str = ""
    pushed_samples_path: str = ""

    def set_model_path(self, model_path: str): self.model_path = model_path
    def set_samples_path(self, samples_path: str): self.samples_path = samples_path
    def set_pushed_samples_path(self, pushed_samples_path: str): self.pushed_samples_path = pushed_samples_path

# --------------------------
# entry
# --------------------------
if __name__ == "__main__":
    # Absolute imports (your requested paths)
    from flow_boost.heilbronn_square import sample_generation as data_generation
    from flow_boost.heilbronn_square import flow_matching_heilbronn
    from flow_boost.heilbronn_square import plot_data_heilbronn  # optional

    console = Console()
    state = PipelineState()

    iterations = _get_cfg("heilbronn_square_pipeline", "iterations", 1)
    start_at_step = str(_get_cfg("heilbronn_square_pipeline", "start_at_step", "push")).strip().lower()
    console.print(f"[Heilbronn Pipeline] Start step: {start_at_step}", style="blue")

    # ----------------------------------------
    # Step 0: either generate training set or push from a given input
    # ----------------------------------------
    if start_at_step == "start":
        console.print("[Heilbronn Pipeline] [Start] training_set_gen", style="blue")
        _set_cfg("heilbronn_SRP", "mode", "training_set_gen")
        data_generation.main(state=state)  # wrapper needed in sample_generation.py (see below)
        if not state.pushed_samples_path:
            raise RuntimeError("[Heilbronn Pipeline] training_set_gen completed but no dataset path was recorded in state.pushed_samples_path")
        if not os.path.exists(state.pushed_samples_path):
            raise FileNotFoundError(f"[Heilbronn Pipeline] Generated training dataset not found: '{state.pushed_samples_path}'")
        _set_cfg("heilbronn_flow", "dataset_path", state.pushed_samples_path)

    elif start_at_step == "push":
        console.print("[Heilbronn Pipeline] [Start] final_push", style="blue")
        _set_cfg("heilbronn_SRP", "mode", "final_push")
        data_generation.main(state=state)
        if not state.pushed_samples_path:
            raise RuntimeError("[Heilbronn Pipeline] final_push completed but no pushed dataset path was recorded in state.pushed_samples_path")
        if not os.path.exists(state.pushed_samples_path):
            raise FileNotFoundError(f"[Heilbronn Pipeline] Pushed dataset not found: '{state.pushed_samples_path}'")
        _set_cfg("heilbronn_flow", "dataset_path", state.pushed_samples_path)

    elif start_at_step in {"train_and_sampling", "training_and_sampling"}:
        console.print(
            "[Heilbronn Pipeline] [Start] training_and_sampling (expects heilbronn_flow.dataset_path already set)",
            style="blue",
        )

        dataset_path = _get_cfg("heilbronn_flow", "dataset_path", "")
        if not dataset_path:
            raise ValueError("start_at_step=training_and_sampling requires heilbronn_flow.dataset_path")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"start_at_step=training_and_sampling but dataset_path does not exist: '{dataset_path}'"
            )

    elif start_at_step in {"retrain_and_sampling", "retrain"}:
        console.print(
            "[Heilbronn Pipeline] [Start] retrain_and_sampling (expects heilbronn_flow.dataset_path + heilbronn_flow.resume_model_path)",
            style="blue",
        )

        dataset_path = _get_cfg("heilbronn_flow", "dataset_path", "")
        if not dataset_path:
            raise ValueError("start_at_step=retrain_and_sampling requires heilbronn_flow.dataset_path")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"start_at_step=retrain_and_sampling but dataset_path does not exist: '{dataset_path}'"
            )

        resume_model_path = _get_cfg("heilbronn_flow", "resume_model_path", "")
        if not resume_model_path:
            raise ValueError("start_at_step=retrain_and_sampling requires heilbronn_flow.resume_model_path")
        if not os.path.exists(resume_model_path):
            raise FileNotFoundError(
                f"start_at_step=retrain_and_sampling but resume_model_path does not exist: '{resume_model_path}'"
            )
    else:
        raise ValueError(
            "start_at_step must be one of: start | push | train_and_sampling | retrain_and_sampling"
        )

    # ----------------------------------------
    # Main loop
    # ----------------------------------------
    for i in range(iterations):
        console.print(f"[Heilbronn Pipeline] Iteration ({i+1}/{iterations})", style="blue")

        # Train + sample
        if i == 0:
            # First iteration mode should match the configured entrypoint.
            if start_at_step in {"retrain_and_sampling", "retrain"}:
                _set_cfg("heilbronn_flow", "mode", "retrain_and_sampling")
            else:
                _set_cfg("heilbronn_flow", "mode", "training_and_sampling")
                # Avoid accidental resume from stale cfg when doing a fresh train.
                _set_cfg("heilbronn_flow", "resume_model_path", "")
        else:
            _set_cfg("heilbronn_flow", "mode", "retrain_and_sampling")

        console.print(f"[Heilbronn Pipeline] [Train+Sample] Iteration ({i+1}/{iterations})", style="blue")
        flow_matching_heilbronn.main(state=state)  # wrapper needed in flow_matching_heilbronn.py

        if not state.samples_path:
            raise RuntimeError("[Heilbronn Pipeline] Train+Sample completed but no generated samples path was recorded in state.samples_path")
        if not os.path.exists(state.samples_path):
            raise FileNotFoundError(f"[Heilbronn Pipeline] Generated samples file not found: '{state.samples_path}'")

        # Wire outputs -> push
        _set_cfg("heilbronn_SRP", "final_push_input", state.samples_path)
        _set_cfg("heilbronn_flow", "resume_model_path", state.model_path)

        # Push samples
        console.print(f"[Heilbronn Pipeline] [Push] Iteration ({i+1}/{iterations})", style="blue")
        _set_cfg("heilbronn_SRP", "mode", "final_push")
        data_generation.main(state=state)

        if not state.pushed_samples_path:
            raise RuntimeError("[Heilbronn Pipeline] Push completed but no pushed dataset path was recorded in state.pushed_samples_path")
        if not os.path.exists(state.pushed_samples_path):
            raise FileNotFoundError(f"[Heilbronn Pipeline] Pushed dataset not found: '{state.pushed_samples_path}'")

        # New pushed dataset becomes the next training dataset
        _set_cfg("heilbronn_flow", "dataset_path", state.pushed_samples_path)

        # Optional plotting step if you want it here
        # console.print(f"[Heilbronn Pipeline] [Plot] Iteration ({i+1}/{iterations})", style="blue")
        # plot_data_heilbronn.main()
