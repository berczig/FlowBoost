import os
from rich.console import Console

"""
Pipeline that loops Training --> RG-fine tuning --> Sampling --> Pushing --> Training --> ...

RG-CFM (reward-guided CFM) usage:
- Set [flow_matching].mode = rg_cfm in config.cfg.
- Provide [flow_matching].dataset_path (for cond loader) and rg_ref_path or resume_model_path (reference checkpoint).
- RG-CFM runs only the online reward based sampling; sampling/pushing loop is unchanged unless you call rg_cfm_main directly.

What are different modes:

- training_and_sampling:
trains the FM model on the dataset
(supervised FM loss + penalty),
then samples new packings with GAS.
Objective: fit the dataset distribution, then generate.

- retrain_and_sampling:
loads a checkpoint, continues supervised FM training on the dataset,
then samples. Objective: fine-tune on data, then generate.

- train_only / retrain_only:
same training behavior as above, but skips sampling and push.

- sampling_only:
loads a checkpoint and only runs GAS sampling.
Objective: generate from an existing model, no training.

- rg_cfm:
skips the supervised FM loop; instead runs online reward-guided fine-tuning
using the current model to generate candidates and weight them by reward,
with self-distillation (SD) regularization to a reference. It trains the model toward higher-reward samples,
but does not, by itself, run the later sampling/push steps.
Use this when you want to fine-tune the model by reward rather than by the dataset.
"""


class PipelineState:
    model_path: str = ""
    samples_path: str = ""
    pushed_samples_path: str = ""

    def set_model_path(self, model_path):
        self.model_path = model_path

    def set_samples_path(self, samples_path):
        self.samples_path = samples_path

    def set_pushed_samples_path(self, pushed_samples_path):
        self.pushed_samples_path = pushed_samples_path


def _normalize_start_step(value: str) -> str:
    step = str(value).strip().lower()
    if step in ("start",):
        return "start"
    if step in ("push",):
        return "push"
    if step in ("train_and_sampling", "training_and_sampling", "train_and_sample"):
        return "training_and_sampling"
    if step in ("retrain_and_sampling", "retrain_and_sample", "retrain"):
        return "retrain_and_sampling"
    if step in ("train_only", "training_only"):
        return "train_only"
    if step in ("retrain_only", "retraining_only"):
        return "retrain_only"
    raise ValueError(
        "spheres_in_cube_new_pipeline.start_at_step must be one of: "
        "start | push | training_and_sampling | retrain_and_sampling | train_only | retrain_only"
    )


def _mode_has_sampling(mode: str) -> bool:
    return mode in ("training_and_sampling", "retrain_and_sampling", "sampling_only")


if __name__ == "__main__":
    # Avoid circular imports
    from flow_boost.spheres_in_hypercube.data_generation import _get_cfg, _set_cfg
    from flow_boost.spheres_in_hypercube import data_generation
    from flow_boost.spheres_in_hypercube import flow_matching_spheres

    console = Console()

    iterations = max(1, int(_get_cfg("spheres_in_cube_new_pipeline", "iterations", 1)))
    start_at_step = _normalize_start_step(_get_cfg("spheres_in_cube_new_pipeline", "start_at_step", "push"))
    use_rg_cfm = bool(_get_cfg("spheres_in_cube_new_pipeline", "use_rg_cfm", False))

    state = PipelineState()
    console.print(f"[Pipeline] Start step: {start_at_step} | use_rg_cfm={use_rg_cfm}", style="blue")

    # Start outside the main loop
    if start_at_step == "start":
        _set_cfg("sample_generation_PP+PBTS", "mode", "training_set_gen")
        data_generation.main(state=state)
        if not getattr(state, "samples_path", ""):
            raise RuntimeError("Training-set generation finished, but state.samples_path was not set.")
        if not os.path.exists(state.samples_path):
            raise FileNotFoundError(f"Generated dataset path does not exist: '{state.samples_path}'")
        _set_cfg("flow_matching", "dataset_path", state.samples_path)
    elif start_at_step == "push":
        console.print("[Pipeline] [Start Push]", style="blue")
        _set_cfg("sample_generation_PP+PBTS", "mode", "final_push")
        data_generation.main(state=state)
        if not getattr(state, "pushed_samples_path", ""):
            raise RuntimeError("Initial push finished, but state.pushed_samples_path was not set.")
        if not os.path.exists(state.pushed_samples_path):
            raise FileNotFoundError(f"Initial pushed dataset path does not exist: '{state.pushed_samples_path}'")
        _set_cfg("flow_matching", "dataset_path", state.pushed_samples_path)
    elif start_at_step in ("training_and_sampling", "train_only"):
        ds_path = str(_get_cfg("flow_matching", "dataset_path", "") or "").strip()
        if not ds_path:
            raise ValueError("start_at_step requires flow_matching.dataset_path to be set.")
        if not os.path.exists(ds_path):
            raise FileNotFoundError(f"flow_matching.dataset_path does not exist: '{ds_path}'")
    elif start_at_step in ("retrain_and_sampling", "retrain_only"):
        ds_path = str(_get_cfg("flow_matching", "dataset_path", "") or "").strip()
        if not ds_path:
            raise ValueError("start_at_step requires flow_matching.dataset_path to be set.")
        if not os.path.exists(ds_path):
            raise FileNotFoundError(f"flow_matching.dataset_path does not exist: '{ds_path}'")
        resume_path = str(_get_cfg("flow_matching", "resume_model_path", "") or "").strip()
        if not resume_path:
            raise ValueError(
                "start_at_step=retrain* requires flow_matching.resume_model_path "
                "(this should point to the previous RG-CFM fine-tuned checkpoint)."
            )
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"flow_matching.resume_model_path does not exist: '{resume_path}'")

    for i in range(iterations):
        console.print(f"[Pipeline] Iteration ({i + 1}/{iterations})", style="blue")

        if i == 0:
            if start_at_step == "retrain_and_sampling":
                flow_mode = "retrain_and_sampling"
            elif start_at_step == "retrain_only":
                flow_mode = "retrain_only"
            elif start_at_step == "train_only":
                flow_mode = "train_only"
            else:
                flow_mode = "training_and_sampling"
        else:
            flow_mode = "retrain_only" if start_at_step in ("train_only", "retrain_only") else "retrain_and_sampling"

        _set_cfg("flow_matching", "mode", flow_mode)
        console.print(
            f"[Pipeline] [Start {flow_mode}] - Iteration ({i + 1}/{iterations}); use_rg_cfm={use_rg_cfm}",
            style="blue",
        )
        flow_matching_spheres.main(state=state)

        if not getattr(state, "model_path", ""):
            raise RuntimeError("Flow matching finished, but state.model_path was not set.")
        if not os.path.exists(state.model_path):
            raise FileNotFoundError(f"Flow matching checkpoint does not exist: '{state.model_path}'")

        # Always propagate latest model; with RG enabled this is the RG fine-tuned checkpoint.
        _set_cfg("flow_matching", "resume_model_path", state.model_path)
        _set_cfg("flow_matching", "rg_ref_path", state.model_path)

        if not _mode_has_sampling(flow_mode):
            console.print(
                f"[Pipeline] Skipping push because mode={flow_mode} does not generate samples.",
                style="yellow",
            )
            continue

        if not getattr(state, "samples_path", ""):
            raise RuntimeError("Flow matching finished, but state.samples_path was not set.")
        if not os.path.exists(state.samples_path):
            raise FileNotFoundError(f"Generated samples path does not exist: '{state.samples_path}'")

        _set_cfg("sample_generation_PP+PBTS", "final_push_input", state.samples_path)

        # Final push Samples
        console.print(f"[Pipeline] [Start Push] - Iteration ({i + 1}/{iterations})", style="blue")
        _set_cfg("sample_generation_PP+PBTS", "mode", "final_push")
        data_generation.main(state=state)
        if not getattr(state, "pushed_samples_path", ""):
            raise RuntimeError("Push finished, but state.pushed_samples_path was not set.")
        if not os.path.exists(state.pushed_samples_path):
            raise FileNotFoundError(f"Pushed samples path does not exist: '{state.pushed_samples_path}'")
        _set_cfg("flow_matching", "dataset_path", state.pushed_samples_path)

    console.print("[Pipeline] Done.", style="green")
