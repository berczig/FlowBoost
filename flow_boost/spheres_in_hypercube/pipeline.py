import flow_boost
import numpy
from flow_boost import cfg
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

- sampling_only: 
loads a checkpoint and only runs GAS sampling. 
Objective: generate from an existing model, no training.

- rg_cfm: 
skips the supervised FM loop; instead runs online reward-guided fine-tuning 
using the current model to generate candidates and weight them by reward,
with self-distillation (SD) regularization to a reference. It trains the model toward higher-reward samples, 
but does not, by itself, run the later sampling/push steps. 
Use this when you want to fine-tune the model by reward rather than by the dataset.

####  Important parameters to check before starting the pipeline  ####

[sample_generation_PP+PBTS]
num_spheres
sphere_radius
num_samples             (if start_at_step == start)
num_srp_restarts
final_push_input        (if start_at_step == push)

[flow_matching]
num_spheres
sphere_radius
num_epochs
learning_rate
dataset_path            (if start_at_step == train_and_sampling or start_at_step == retrain_and_sampling)
resume_model_path       (if start_at_step == retrain_and_sampling)
num_generated_samples
train_top_fraction
wall_weight

########################################################################
"""

class PipelineState:
    model_path:str = ""
    samples_path:str = ""
    pushed_samples_path:str = ""

    def set_model_path(self, model_path):
        self.model_path = model_path

    def set_samples_path(self, samples_path):
        self.samples_path = samples_path

    def set_pushed_samples_path(self, pushed_samples_path):
        self.pushed_samples_path = pushed_samples_path

if __name__ == "__main__":
    # Avoid Circular imports
    from flow_boost.spheres_in_cube_12d.data_generation import _get_cfg, _set_cfg
    from flow_boost.spheres_in_cube_12d import data_generation
    from flow_boost.spheres_in_cube_12d import flow_matching_spheres
    from flow_boost.spheres_in_cube_12d import plot_data_spheres

    # Color 
    console = Console()

    iterations = _get_cfg("spheres_in_cube_new_pipeline", "iterations", 1)
    start_at_step = _get_cfg("spheres_in_cube_new_pipeline", "start_at_step", "push")
    use_rg_cfm = _get_cfg("spheres_in_cube_new_pipeline", "use_rg_cfm", False)

    #"start", "training_and_sampling", "push", "retrain_and_sampling"
    state = PipelineState()
    console.print(f"[Pipeline] Start step: {start_at_step}", style="blue")
    
    # Start outside the main loop
    if start_at_step == "start":
        _set_cfg("sample_generation_PP+PBTS", "mode", "training_set_gen")
        data_generation.main(state=state)
        if state.samples_path:
            _set_cfg("flow_matching", "dataset_path", state.samples_path)
    elif start_at_step == "push":
        console.print(f"[Pipeline] [Start Push]", style="blue")
        _set_cfg("sample_generation_PP+PBTS", "mode", "final_push")
        data_generation.main(state=state)
        _set_cfg("flow_matching", "dataset_path", state.pushed_samples_path)

    for i in range(iterations):
        console.print(f"[Pipeline] Iteration ({i+1}/{iterations})", style="blue")

        if use_rg_cfm:
            # Single call: training_and_sampling now handles RG-CFM internally (if enabled) then samples
            _set_cfg("flow_matching", "mode", "training_and_sampling")
            console.print(f"[Pipeline] [Start training_and_sampling (with RG-CFM enabled)] - Iteration ({i+1}/{iterations}); mode={_get_cfg('flow_matching','mode','')}", style="blue")
            flow_matching_spheres.main(state=state)
        else:
            # (Re)train Model
            if i == 0 and start_at_step in ["train_and_sampling", "training_and_sampling", "start"]:
                _set_cfg("flow_matching", "mode", "training_and_sampling")
            else:
                _set_cfg("flow_matching", "mode", "retrain_and_sampling")

            console.print(f"[Pipeline] [Start retraining and sampling] - Iteration ({i+1}/{iterations}); mode={_get_cfg('flow_matching','mode','')}", style="blue")
            flow_matching_spheres.main(state=state)

        _set_cfg("sample_generation_PP+PBTS", "final_push_input", state.samples_path)
        _set_cfg("flow_matching", "resume_model_path", state.model_path)

        # Final push Samples
        console.print(f"[Pipeline] [Start Push] - Iteration ({i+1}/{iterations})", style="blue")
        _set_cfg("sample_generation_PP+PBTS", "mode", "final_push")
        data_generation.main(state=state)
        _set_cfg("flow_matching", "dataset_path", state.pushed_samples_path)


            # plot_data_samples no input
            # plot_data_samples speed up

            # pipeline cases check
            # remove bloat .csv metrics ect
            # 
            # generate log file? (cant trust the console)
            # new data selection? right now 100% from new samples right?
