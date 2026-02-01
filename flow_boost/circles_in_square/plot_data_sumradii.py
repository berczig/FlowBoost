import os, platform

# Allow running this file directly (e.g. `python path/to/plot_data_sumradii.py`)
# without requiring an editable install. When executed as a script, Python only
# adds this file's directory to sys.path, so the repo root isn't visible.
if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import flow_boost
# Quick unblock for macOS OpenMP duplication. Disable via:
#   export SPHEREPACK_DISABLE_KMP_HACK=1
if platform.system() == "Darwin" and os.environ.get("SPHEREPACK_DISABLE_KMP_HACK") != "1":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime

# -----------------------------------------------------------------------------
# Edit these paths / datasets
# -----------------------------------------------------------------------------

# Preferred: define an arbitrary list of datasets to plot.
# Each entry is either:
#   - (path_or_tensor, "Title")
#   - (path_or_tensor,)  -> title auto-derived from file basename
#   - path_or_tensor     -> title auto-derived from file basename
PLOT_DATASETS = [
     (r"flow_boost/output/circles_in_square/training_sets/circle_srp_generated_3000x26_2025-12-15_161340.pt", "Training data"),
     (r"flow_boost/output/circles_in_square/generated_sets/flow_centers_only_gen_20251215_172211.pt", "Generated samples"),
     (r"flow_boost/output/circles_in_square/fixed_gen_sets/circle_final_push_mod_3000x26_2025-12-15_172216.pt", "Pushed samples"),
     #(r"flow_boost/output/circles_in_square/fixed_gen_sets/circle_final_push_mod_3000x26_2025-12-15_175708.pt", "3rd iteration"),
    ]

save_dir = "flow_boost/output/circles_in_square/fixed_gen_sets/distribution_plots"  # where to save the plot PNG

# Backward-compatible single variables (used only if PLOT_DATASETS is empty)
training_data = "flow_boost/output/circles_in_square/training_sets/circle_srp_generated_300x28_2025-11-04_150356.pt"  # e.g. "flow_boost/output/circles_in_square/training_sets/circle_srp_generated_2000x32_2025-10-22_120000.pt"
gen_samples = "flow_boost/output/circles_in_square/fixed_gen_sets/circle_final_push_mod_300x28_2025-11-04_152618.pt"
pushed_samples = "flow_boost/output/circles_in_square/fixed_gen_sets/circle_final_push_mod_1000x28_2025-11-04_165649.pt"

# -----------------------------------------------------------------------------
# Loaders / metrics
# -----------------------------------------------------------------------------
def load_dataset(file):
    if isinstance(file, torch.Tensor):
        return file
    obj = torch.load(file)
    if isinstance(obj, dict):
        # Try common keys; fallback to first tensor-like value
        for key in ("pushed", "samples", "data"):
            if key in obj and isinstance(obj[key], torch.Tensor):
                return obj[key]
        for v in obj.values():
            if isinstance(v, torch.Tensor):
                return v
        raise ValueError("Dict did not contain a tensor.")
    return obj

def _to_Mx3xN(arr):
    a = arr.detach().cpu().numpy() if isinstance(arr, torch.Tensor) else np.asarray(arr)
    if a.ndim == 3 and a.shape[1] == 3:
        return a
    if a.ndim == 3 and a.shape[2] == 3:
        return np.transpose(a, (0, 2, 1))
    if a.ndim == 2 and a.shape[0] == 3:
        return a[None, ...]
    if a.ndim == 2 and a.shape[1] == 3:
        return np.transpose(a, (1, 0))[None, ...]
    raise ValueError(f"Expected (M,3,N) or (M,N,3), got {a.shape}")

def compute_sum_radii(tensor_data):
    data = _to_Mx3xN(tensor_data)
    # sum over radii channel
    sum_r = data[:, 2, :].sum(axis=1)
    return {"sum_r": sum_r}

# -----------------------------------------------------------------------------
# Plot (overlay normalized histograms)
# -----------------------------------------------------------------------------
def plot(Arrays, labels, savepath, n_bins=100,
         # Add number of circles to the title automatically
         title="Sum of Radii Distribution", xlabel="Sum of Radii",
         ylabel="Frequency", plotmode="overlay", n_xticks=15,
         stats=None, outfile_name=None):
    max_val = -np.inf
    min_val = np.inf
    for values in Arrays:
        if len(values) == 0:
            continue
        max_val = max(max_val, np.max(values))
        min_val = min(min_val, np.min(values))

    if not np.isfinite(max_val) or not np.isfinite(min_val):
        raise ValueError("No finite values to plot.")

    # expand a bit for visual margins
    span = max_val - min_val
    if span <= 0:
        span = max(1.0, abs(max_val)) * 0.05
    max_val = max_val + 0.05 * span
    min_val = max(0.0, min_val - 0.05 * span)

    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    bin_widths = np.diff(bin_edges)
    bar_x_positions = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    if plotmode == "overlay":
        max_freq = 0.0
        for index, values in enumerate(Arrays):
            if len(values) == 0:
                continue
            hist_data = np.histogram(values, bins=bin_edges)[0] / max(1, len(values))
            color = mpl.colors.hsv_to_rgb((index / max(1, len(Arrays)), 0.85, 0.95))
            max_freq = max(max_freq, np.max(hist_data))
            label = f"{labels[index]}[{len(values)}]"
            plt.bar(bar_x_positions, hist_data, width=bin_widths,
                    edgecolor='black', alpha=0.45, label=label, color=color)
            plt.ylim(0, max_freq if max_freq > 0 else 1.0)
    else:
        datas = []
        weights = []
        labels_samples = []
        for index, values in enumerate(Arrays):
            datas.append(values)
            w = np.full(len(values), 1 / max(1, len(values))) if len(values) > 0 else np.array([1.0])
            weights.append(w)
            labels_samples.append(f"{labels[index]}[{len(values)}]")
        plt.hist(datas, bin_edges, weights=weights, label=labels_samples)

    # Ticks
    ticks = np.linspace(min_val, max_val, n_xticks)
    plt.xticks(ticks, rotation=45)

    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlim(min_val, max_val)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Add summary stats to the plot so the saved image is self-contained.
    if stats is not None:
        lines = []
        for i, st in enumerate(stats):
            if st is None:
                continue
            label = labels[i] if i < len(labels) else f"Set {i+1}"
            lines.append(
                f"{label}: mean={st['mean']:.6f}  std={st['std']:.6f}  min={st['min']:.6f}  max={st['max']:.6f}"
            )
        if lines:
            ax.text(
                0.01,
                0.99,
                "\n".join(lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="black"),
            )

    os.makedirs(savepath, exist_ok=True)
    if outfile_name is None:
        outfile_name = "sum_radii_hist.png"
    out_png = os.path.join(savepath, outfile_name)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved plot: {out_png}")

# -----------------------------------------------------------------------------
# Aggregate and plot the three sets
# -----------------------------------------------------------------------------
def plot_files_combined(files, labels, savepath, **kwargs):
    arrays_sum_r = []
    stats = []
    n_circles_list = []
    for file in files:
        dataset = load_dataset(file)
        arr = _to_Mx3xN(dataset)
        n_circles_list.append(int(arr.shape[2]))
        metr = compute_sum_radii(arr)
        values = metr["sum_r"]
        arrays_sum_r.append(values)
        # per-dataset stats for annotation
        if len(values) == 0:
            stats.append(None)
        else:
            stats.append(
                {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )

    # Save filename includes number of circles and date/time.
    # If datasets have different N, mark as mixed.
    n_unique = sorted(set(n_circles_list))
    n_tag = f"N{n_unique[0]}" if len(n_unique) == 1 else "Nmixed"
    date_tag = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    outfile_name = f"sum_radii_hist_{n_tag}_{date_tag}.png"

    if len(n_unique) == 1:
        n_title = f"N={n_unique[0]} circles"
    else:
        n_title = "N=mixed circles"
        if len(n_unique) <= 6:
            n_title = f"N=mixed ({', '.join(str(n) for n in n_unique)})"

    plot(Arrays=arrays_sum_r, labels=labels, savepath=os.path.join(savepath, "sum_radii"),
         title=f"Normalized Sum of Radii Distribution ({n_title})",
         xlabel="Sum of Radii", ylabel="Frequency",
         stats=stats, outfile_name=outfile_name, **kwargs)


def _normalize_plot_datasets(plot_datasets):
    files = []
    labels = []

    for item in plot_datasets:
        if isinstance(item, tuple) or isinstance(item, list):
            if len(item) == 0:
                continue
            path_or_tensor = item[0]
            title = item[1] if len(item) > 1 else None
        else:
            path_or_tensor = item
            title = None

        if title is None:
            if isinstance(path_or_tensor, str):
                title = os.path.basename(path_or_tensor) or path_or_tensor
            else:
                title = "Tensor"

        files.append(path_or_tensor)
        labels.append(title)

    return files, labels

if __name__ == "__main__":
    # Build list of datasets from in-script config.
    if PLOT_DATASETS:
        files, labels = _normalize_plot_datasets(PLOT_DATASETS)
    else:
        files = [training_data, gen_samples, pushed_samples]
        labels = ["Training data", "Samples (Flow matching)", "Final pushed samples"]

    # Remove empty entries
    files_labels = [(f, l) for f, l in zip(files, labels) if f]
    if not files_labels:
        print("Please set PLOT_DATASETS (preferred) or the legacy training_data/gen_samples/pushed_samples variables.")
        raise SystemExit(1)

    files, labels = zip(*files_labels)

    plot_files_combined(
        list(files),
        list(labels),
        save_dir,
        n_bins=120,
        plotmode="overlay",
        n_xticks=12,
    )

    # Optional: print summary stats
    for label, path_or_tensor in zip(labels, files):
        arr = _to_Mx3xN(load_dataset(path_or_tensor))
        sums = arr[:, 2, :].sum(axis=1)
        print(
            f"{label}: mean={sums.mean():.6f} std={sums.std():.6f} "
            f"min={sums.min():.6f} max={sums.max():.6f}"
        )

