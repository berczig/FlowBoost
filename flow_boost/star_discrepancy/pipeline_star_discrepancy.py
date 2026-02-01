import flow_boost
import os
from flow_boost import cfg
from rich.console import Console

"""
Pipeline that loops Training --> Sampling --> Pushing --> Training --> ...

This mirrors the spheres pipeline structure, but uses:
- star_SRP.sample_generation (SRP + L-BFGS to generate/push pointsets)
- star_flow.flow_matching_star_discrepancy (FM training + sampling)

Key idea:
- The generator writes a dataset .pt and updates `state.samples_path` (training_set_gen)
  OR updates `state.pushed_samples_path` (push_only).
- The FM script writes a model checkpoint and generated samples, updating
  `state.model_path` and `state.samples_path`.

####  Important parameters to check before starting the pipeline  ####

[star_SRP]
num_points
num_samples             (if start_at_step == start)
mode                    (training_set_gen / push_only)
push_input              (if start_at_step == push; also set by pipeline before each push)
push_output_dir

[star_flow]
num_points
num_epochs
learning_rate
dataset_path            (if start_at_step == train_and_sampling or start_at_step == retrain_and_sampling)
resume_model_path       (if start_at_step == retrain_and_sampling)
num_generated_samples
train_top_fraction

[star_discrepancy_pipeline]
iterations
start_at_step           ("start", "train_and_sampling", "push", "retrain_and_sampling")
"""

class PipelineState:
    """
    Shared state object passed into both scripts.
    The scripts are responsible for writing these fields.
    """
    model_path: str = ""
    samples_path: str = ""
    pushed_samples_path: str = ""

    def set_model_path(self, model_path):
        self.model_path = model_path

    def set_samples_path(self, samples_path):
        self.samples_path = samples_path

    def set_pushed_samples_path(self, pushed_samples_path):
        self.pushed_samples_path = pushed_samples_path


if __name__ == "__main__":
    # Avoid circular imports; also matches the spheres pipeline style.
    from flow_boost.star_discrepancy.sample_generation import _get_cfg, _set_cfg
    from flow_boost.star_discrepancy import sample_generation
    from flow_boost.star_discrepancy import flow_matching_star_discrepancy

    console = Console()

    iterations = _get_cfg("star_discrepancy_pipeline", "iterations", 1)
    start_at_step = _get_cfg("star_discrepancy_pipeline", "start_at_step", "push")
    if isinstance(start_at_step, str):
        start_at_step = start_at_step.strip().lower()

    state = PipelineState()
    console.print(f"[Pipeline] Start step: {start_at_step}", style="blue")

    # ------------------------------------------------------------
    # Start outside the main loop
    #
    # We support two convenient entry points:
    # - "start": generate a fresh SRP training set, then train FM on it.
    # - "push":  push an existing dataset (configured via star_SRP.push_input),
    #            then train FM on the pushed results.
    #
    # If you start at "train_and_sampling" or "retrain_and_sampling",
    # you must set star_flow.dataset_path (and resume_model_path if retraining)
    # in the config yourself.
    # ------------------------------------------------------------
    if start_at_step == "start":
        console.print("[Pipeline] [Start training set generation]", style="blue")

        # Tell SRP script to generate a brand-new dataset
        _set_cfg("star_SRP", "mode", "training_set_gen")
        sample_generation.main(state=state)

        # IMPORTANT: training_set_gen writes to state.samples_path
        # (NOT pushed_samples_path)
        if getattr(state, "samples_path", ""):
            if not os.path.exists(state.samples_path):
                raise FileNotFoundError(f"SRP generation reported samples_path but file does not exist: '{state.samples_path}'")
            _set_cfg("star_flow", "dataset_path", state.samples_path)
        else:
            raise RuntimeError("SRP generation finished, but state.samples_path was not set.")

    elif start_at_step == "push":
        console.print("[Pipeline] [Start Push]", style="blue")

        # Tell SRP script to load `star_SRP.push_input` and push it
        _set_cfg("star_SRP", "mode", "push_only")
        sample_generation.main(state=state)

        # push_only writes to state.pushed_samples_path
        if getattr(state, "pushed_samples_path", ""):
            if not os.path.exists(state.pushed_samples_path):
                raise FileNotFoundError(f"SRP push reported pushed_samples_path but file does not exist: '{state.pushed_samples_path}'")
            _set_cfg("star_flow", "dataset_path", state.pushed_samples_path)
        else:
            raise RuntimeError("SRP push finished, but state.pushed_samples_path was not set.")

    elif start_at_step in ("train_and_sampling", "training_and_sampling"):
        console.print(
            "[Pipeline] [Start] train_and_sampling (expects star_flow.dataset_path already set)",
            style="blue",
        )
        # Nothing to generate/push here; flow_matching_star_discrepancy will read star_flow.dataset_path.

    elif start_at_step in ("retrain_and_sampling", "retrain", "retrain_and_sample"):
        console.print(
            "[Pipeline] [Start] retrain_and_sampling (expects star_flow.dataset_path and star_flow.resume_model_path already set)",
            style="blue",
        )
        # Nothing to generate/push here; flow_matching_star_discrepancy will read dataset_path + resume_model_path.

    else:
        raise ValueError(
            "star_discrepancy_pipeline.start_at_step must be one of: start | push | train_and_sampling | retrain_and_sampling"
        )

    # ------------------------------------------------------------
    # Main loop: (Re)train + sample  ->  push  ->  repeat
    #
    # After each FM run:
    # - flow_matching_star_discrepancy writes:
    #   state.model_path   (checkpoint)
    #   state.samples_path (generated pointsets)
    #
    # Then we push the generated samples using SRP:
    # - star_SRP.push_input = state.samples_path
    # - star_SRP.mode = push_only
    # - sample_generation writes state.pushed_samples_path
    #
    # And we train the next FM iteration on the pushed samples:
    # - star_flow.dataset_path = state.pushed_samples_path
    # ------------------------------------------------------------
    for i in range(iterations):
        console.print(f"[Pipeline] Iteration ({i+1}/{iterations})", style="blue")

        # Decide whether to do a fresh train or a retrain
        # (same pattern as spheres pipeline).
        # First iteration mode depends on how we started.
        if i == 0 and start_at_step in ("retrain_and_sampling", "retrain", "retrain_and_sample"):
            _set_cfg("star_flow", "mode", "retrain_and_sampling")
        else:
            # For start/push/train_and_sampling we want a fresh train on the first iteration.
            # For subsequent iterations we always retrain.
            _set_cfg("star_flow", "mode", "training_and_sampling" if i == 0 else "retrain_and_sampling")

        # Fail fast if required inputs are missing for the chosen mode.
        flow_mode = str(_get_cfg("star_flow", "mode", "training_and_sampling")).strip().lower()
        ds_path = str(_get_cfg("star_flow", "dataset_path", "") or "").strip()
        if flow_mode in ("training_and_sampling", "retrain_and_sampling"):
            if not ds_path:
                raise RuntimeError("star_flow.dataset_path must be set before running training/retraining.")
            if not os.path.exists(ds_path):
                raise FileNotFoundError(f"star_flow.dataset_path does not exist: '{ds_path}'")
        if flow_mode == "retrain_and_sampling":
            resume = str(_get_cfg("star_flow", "resume_model_path", "") or "").strip()
            if not resume:
                raise RuntimeError("star_flow.resume_model_path must be set for retrain_and_sampling.")
            if not os.path.exists(resume):
                raise FileNotFoundError(f"star_flow.resume_model_path does not exist: '{resume}'")
        console.print(
            f"[Pipeline] [Start {flow_mode}] - Iteration ({i+1}/{iterations})",
            style="blue",
        )
        flow_matching_star_discrepancy.main(state=state)

        # After FM run, we must have:
        # - state.samples_path: the generated samples .pt
        # - state.model_path:   the checkpoint .pth
        if not getattr(state, "samples_path", ""):
            raise RuntimeError("FM finished, but state.samples_path was not set.")
        if not os.path.exists(state.samples_path):
            raise FileNotFoundError(f"FM reported samples_path but file does not exist: '{state.samples_path}'")
        if not getattr(state, "model_path", ""):
            raise RuntimeError("FM finished, but state.model_path was not set.")
        if not os.path.exists(state.model_path):
            raise FileNotFoundError(f"FM reported model_path but file does not exist: '{state.model_path}'")

        # Configure next push + next retrain resume path
        _set_cfg("star_SRP", "push_input", state.samples_path)
        _set_cfg("star_flow", "resume_model_path", state.model_path)

        # Push samples (improve them via SRP/L-BFGS)
        console.print(f"[Pipeline] [Start Push] - Iteration ({i+1}/{iterations})", style="blue")
        _set_cfg("star_SRP", "mode", "push_only")
        sample_generation.main(state=state)

        # After push we must have pushed_samples_path
        if not getattr(state, "pushed_samples_path", ""):
            raise RuntimeError("Push finished, but state.pushed_samples_path was not set.")
        if not os.path.exists(state.pushed_samples_path):
            raise FileNotFoundError(f"Push reported pushed_samples_path but file does not exist: '{state.pushed_samples_path}'")

        # Train next iteration on pushed samples
        _set_cfg("star_flow", "dataset_path", state.pushed_samples_path)

    console.print("[Pipeline] Done.", style="green")
