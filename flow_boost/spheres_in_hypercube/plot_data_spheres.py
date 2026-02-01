import os, platform
import sys

# If you need the quick unblock, leave this enabled on macOS.
# Disable by exporting SPHEREPACK_DISABLE_KMP_HACK=1 in your shell.
if platform.system() == "Darwin" and os.environ.get("SPHEREPACK_DISABLE_KMP_HACK") != "1":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from pathlib import Path

from flow_boost.spheres_in_cube_new.pipeline import PipelineState
from flow_boost.spheres_in_cube_new.flow_matching_spheres import calculate_min_sep
from flow_boost.spheres_in_cube.best_results import load_best_results


# ============================================================
# Loading utilities
# ============================================================

def load_raw(file):
    """Load a PyTorch file (.pt) or return tensor directly."""
    if isinstance(file, torch.Tensor):
        return file
    return torch.load(file)


def extract_tensor(obj):
    """
    Try to extract a tensor of shape (M, d, N) or (M, N, d) from various formats:
    - torch.Tensor
    - dict with common keys like 'pushed', 'data', 'samples', etc.
    - numpy array
    """
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)

    if isinstance(obj, dict):
        # Try some common keys
        for key in ["pushed", "data", "samples", "points", "packings"]:
            if key in obj and isinstance(obj[key], torch.Tensor):
                return obj[key]
        # Fallback: if there's exactly one tensor value, use it
        tensor_vals = [v for v in obj.values() if isinstance(v, torch.Tensor)]
        if len(tensor_vals) == 1:
            return tensor_vals[0]
        raise ValueError("Could not infer which tensor to use from dict keys.")

    raise TypeError(f"Unsupported object type for extract_tensor: {type(obj)}")


def load_dataset(file):
    raw = load_raw(file)
    return extract_tensor(raw)

def get_all_file_names(folder:str, labeler:callable):
    """
    Returns a list of relative paths (folder + filename) 
    sorted alphabetically.
    """
    path = Path(folder)
    
    # f is the full path object; we convert to string to get the relative path
    # We filter with .is_file() to exclude subfolders
    files = sorted([(str(f), labeler(index)) for index, f in enumerate(path.iterdir()) if f.is_file()])
    print("files:", files)
    return files
# ============================================================
# Geometry: min pair distance & radii
# ============================================================

def _min_pairwise_distance_sample(points_nd: np.ndarray) -> float:
    """
    points_nd: shape (N, d) numpy array.
    Returns the minimum pairwise Euclidean distance.
    """
    N = points_nd.shape[0]
    if N < 2:
        return 0.0
    best = np.inf
    for i in range(N):
        xi = points_nd[i]
        for j in range(i + 1, N):
            dx = xi - points_nd[j]
            d = float(np.sqrt(np.dot(dx, dx)))
            if d < best:
                best = d
    return float(best if np.isfinite(best) else 0.0)

def compute_radii2(tensor_data):
    min_dists = calculate_min_sep(tensor_data, len(tensor_data), chunk=128)
    return {"radii": 0.5*min_dists.numpy(), "min_distances": min_dists.numpy()}

def compute_radii(tensor_data):
    """
    tensor_data: expected shapes (M,d,N) or (M,N,d) or torch.Tensor.
    Returns dict with per-sample:
      - 'radii':          effective radius = 0.5 * min pairwise distance
      - 'min_distances':  min pairwise distance itself
    """
    if isinstance(tensor_data, torch.Tensor):
        arr = tensor_data.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor_data)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape {arr.shape}")

    # Normalize to (M, d, N)
    if arr.shape[1] < arr.shape[2]:
        data_d_n = arr  # (M, d, N)
    else:
        data_d_n = np.transpose(arr, (0, 2, 1))  # (M, d, N)

    M, d, N = data_d_n.shape
    min_dists = np.empty(M, dtype=np.float64)
    radii = np.empty(M, dtype=np.float64)

    # min_dists2 = calculate_min_sep(torch.tensor(data_d_n), M, chunk=128)
    for idx in tqdm(range(M)):
        # data_d_n[idx]: (d, N) -> (N, d)
        pts = data_d_n[idx].T.astype(np.float64, copy=False)
        md = _min_pairwise_distance_sample(pts)
        min_dists[idx] = md
        radii[idx] = 0.5 * md
    # diff = torch.tensor(min_dists)-min_dists2
    # print("diff: ", diff)
    # print(torch.sum(torch.abs(diff)))


    return {"radii": radii, "min_distances": min_dists}


# ============================================================
# Plotting utilities
# ============================================================

def plot_double_1D(data1, data2, data1_label, data2_label, x_label, y_label, title, horizontal_value=None, horizontal_label=None):

    # 2. Create figure and axes (Object-Oriented style)
    fig, ax = plt.subplots(figsize=(8, 5))

    # 3. Plot the data
    # If you only provide one array, it is used as y-values and 
    # x-values are automatically generated as indices (0, 1, 2...)
    ax.plot(data1, marker='o', linestyle='-', color='b', label=data1_label)
    ax.plot(data2, marker='o', linestyle='-', color='r', label=data2_label)

    if horizontal_value:
        ax.axhline(y=0.090490, color='g', linestyle='-', linewidth=2, label=horizontal_label)

    # 4. Customize labels and title
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # 5. Enhance visuals (2026 Best Practices)
    ax.grid(True, linestyle='--', alpha=0.7) # Add subtle grid
    ax.legend()                              # Display the legend
    plt.tight_layout()                       # Adjust layout to prevent clipping

    # 6. Save or Show
    # plt.savefig("plot_2026.png", dpi=300)  # Save at high resolution
    plt.show()

def plot(Arrays, labels, savepath,
         n_bins=100,
         title="Sphere radius distribution",
         xlabel="Radius",
         ylabel="Frequency",
         plotmode="overlay",
         n_xticks=15,
         show=True,
         filename="radii_hist.png"):
    """
    Arrays: list of 1D numpy arrays (radii per dataset)
    labels: list of labels (same length as Arrays)
    savepath: directory where to save the figure (if not None)
    plotmode: 'overlay' (default) or 'stacked' (hist)
    """
    if len(Arrays) != len(labels):
        raise ValueError("Arrays and labels must have the same length.")

    # Range across all arrays
    max_val = -np.inf
    min_val = np.inf
    for values in Arrays:
        if len(values) == 0:
            continue
        max_val = max(max_val, float(np.max(values)))
        min_val = min(min_val, float(np.min(values)))

    if not np.isfinite(max_val) or not np.isfinite(min_val):
        raise ValueError("No finite values to plot.")

    # Pad range slightly
    pad = 0.05 * (max_val - min_val if max_val > min_val else max_val or 1.0)
    max_val = max_val + pad
    min_val = max(0.0, min_val - pad)

    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    bin_widths = np.diff(bin_edges)
    bar_x_positions = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(10, 6))

    if plotmode == "overlay":
        max_freq = 0.0
        for index, values in enumerate(Arrays):
            if len(values) == 0:
                continue
            hist_counts, _ = np.histogram(values, bins=bin_edges)
            # normalize to probability (frequency)
            hist_data = hist_counts / max(1, len(values))
            color = mpl.colors.hsv_to_rgb((index / max(1, len(Arrays)), 1, 1))
            max_freq = max(max_freq, float(np.max(hist_data)))
            label = f"{labels[index]} [{len(values)} samples]"
            plt.bar(bar_x_positions,
                    hist_data,
                    width=bin_widths,
                    edgecolor='black',
                    alpha=0.4,
                    label=label,
                    color=color)
        plt.ylim(0, max_freq if max_freq > 0 else 1.0)
    else:
        datas = []
        weights = []
        labels_samples = []
        for index, values in enumerate(Arrays):
            datas.append(values)
            if len(values) == 0:
                w = np.array([1.0])
            else:
                w = np.full(len(values), 1.0 / len(values))
            weights.append(w)
            labels_samples.append(f"{labels[index]} [{len(values)} samples]")
        plt.hist(datas, bin_edges, weights=weights, label=labels_samples)

    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlim(min_val, max_val)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # optional: nicer x-ticks
    try:
        xticks = np.linspace(min_val, max_val, n_xticks)
        plt.xticks(xticks)
    except Exception:
        pass

    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        out_file = os.path.join(savepath, filename)
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved figure to {out_file}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_files_combined(files, labels, savepath, title, **kwargs):
    """
    files: list of .pt paths (or tensors)
    labels: list of labels (same length)
    savepath: directory to save plot
    kwargs: forwarded to plot(...)
    """
    arrays_radii = []
    radius_averages = []
    radius_maxs = []
    for i, file in enumerate(files):
        dataset = load_dataset(file)
        metr = compute_radii2(dataset)
        #print("metr: ", metr)
        arrays_radii.append(metr["radii"])
        radius_avg = np.mean(metr['radii'])
        radius_max = np.max(metr['radii'])
        radius_averages.append(radius_avg)
        radius_maxs.append(radius_max)
        labels[i] = f"{labels[i]}, max radius = {radius_max:.6f}, avg. radius = {radius_avg:.6f}"

    num_spheres = dataset.shape[2]
    best_cube_lengths = load_best_results()
    try:
        best_radius = 1/best_cube_lengths[num_spheres]
    except:
        best_radius = 0.0
    title = f"{title}, {num_spheres} Spheres, best radius = {best_radius:.6f}"

    plot(Arrays=arrays_radii,
         labels=labels,
         savepath=savepath,
         title=title if title else "Sphere radius distribution" ,
         xlabel="Effective radius (min distance / 2)",
         ylabel="Frequency",
         **kwargs)
    
    # avg and radius plot
    plot_double_1D(radius_maxs, radius_averages, "Max. radius", "Avg radius", "Pipeline loop", "", "Avg. and Max. Radius per Pipeline step", best_radius, "radius to beat")



# ============================================================
# 3D visualization (optional)
# ============================================================

def plot_3d(dataset, title="plot"):
    """
    Quick 3D scatter for a few configurations.
    dataset: tensor or path to .pt with shape (M, d, N) or (M, N, d)
    """
    if isinstance(dataset, str):
        dataset = load_dataset(dataset)
    if isinstance(dataset, torch.Tensor):
        arr = dataset.detach().cpu().numpy()
    else:
        arr = np.asarray(dataset)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D tensor for plotting, got shape {arr.shape}")

    # Normalize to (M, 3, N)
    if arr.shape[1] == 3:
        data_3n = arr
    elif arr.shape[2] == 3:
        data_3n = np.transpose(arr, (0, 2, 1))
    else:
        raise ValueError(f"Second or last dimension must be 3; got shape {arr.shape}")

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    for idx, data in enumerate(data_3n):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xdata, ydata, zdata = data
        ax.scatter3D(xdata, ydata, zdata, c=zdata)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"{title} (sample {idx})")
        plt.show()


# ============================================================
# Main: interactive multi-file radius plots
# ============================================================

def main(state:PipelineState=None):
    # Usage 1: pass files as command-line arguments
    #   python plot_radii.py file1.pt file2.pt ...
    # Usage 2: run with no args, enter paths interactively

    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        # files = ["flow_boost/output/spheres_in_cube_new/training_sets/srp_data_N191_2025-11-27_00-00-19.pt",
        #          "flow_boost/output/spheres_in_cube_new/generated_sets/spheres_gen_500x191_20251128_000000.pt",
        #          "flow_boost/output/spheres_in_cube_new/final_pushed/spheres_srp_pushed_N191_2025-11-28_002657.pt"]
        
        # files = ["flow_boost/output/spheres_in_cube_new/generated_sets/spheres_gen_500x191_20251128_000000.pt",
        #          "flow_boost/output/spheres_in_cube_new/generated_sets/spheres_gen_500x191_20251128_094024_w=0.pt"]
        
        # files = ["flow_boost/output/spheres_in_cube_new/training_sets/2025-11-27/srp_data_N191_2025-11-27_00-00-19.pt",
        #          "flow_boost/output/spheres_in_cube_new/final_pushed/2025-11-29/spheres_srp_pushed_N191_2025-11-29_041209.pt",
        #          "flow_boost/output/spheres_in_cube_new/final_pushed/2025-11-29/spheres_srp_pushed_N191_2025-11-29_073551.pt",
        #          "flow_boost/output/spheres_in_cube_new/final_pushed/2025-11-29/spheres_srp_pushed_N191_2025-11-29_105908.pt",
        #          "flow_boost/output/spheres_in_cube_new/final_pushed/2025-11-29/spheres_srp_pushed_N191_2025-11-29_154519.pt",
        #          "flow_boost/output/spheres_in_cube_new/final_pushed/2025-11-29/spheres_srp_pushed_N191_2025-11-29_211112.pt",]
        
        # files= ["flow_boost/output/spheres_in_cube_new/final_pushed/spheres_srp_pushed_N191_2025-11-28_002657.pt"]

        files = [("flow_boost/output/spheres_in_cube_new/training_sets/2025-12-04/srp_data_N191_2025-12-04_01-50-14.pt", "Training"),
                 ("flow_boost/output/spheres_in_cube_new/best_merge/2025-12-04/merge_20251204_064146.pt",  "Push1"),
                 ("flow_boost/output/spheres_in_cube_new/best_merge/2025-12-04/merge_20251204_100128.pt",  "Push2"),
                 ("flow_boost/output/spheres_in_cube_new/best_merge/2025-12-04/merge_20251204_131713.pt",  "Push3"),
                 ("flow_boost/output/spheres_in_cube_new/best_merge/2025-12-04/merge_20251204_163338.pt",  "Push4"),
                 ("flow_boost/output/spheres_in_cube_new/best_merge/2025-12-04/merge_20251204_203037.pt",  "Push5"),
                 ("flow_boost/output/spheres_in_cube_new/best_merge/2025-12-05/merge_20251205_045225.pt",  "Push6"),
                 ("flow_boost/output/spheres_in_cube_new/best_merge/2025-12-05/merge_20251205_073550.pt",  "Push7"),]
        
        files = [("flow_boost/output/spheres_in_cube_new/training_sets/2025-12-08/srp_data_N79_2025-12-08_10-35-35.pt", "Training"),
                 ("flow_boost/output/spheres_in_cube_new/best_merge/2025-12-08/merge_20251208_113954.pt",  "Push1"),
                 ("flow_boost/output/spheres_in_cube_new/best_merge/2025-12-08/merge_20251208_140034.pt",  "Push4")]
        
        files = [("flow_boost/spheres_in_cube_new/my_test_data.pt", "2 spheres"),
                 ("flow_boost/output/spheres_in_cube_new/final_pushed/2025-12-12/spheres_srp_pushed_N2_2025-12-12_103429.pt", "10, pushed, radius=0.1"),
                 ("flow_boost/output/spheres_in_cube_new/final_pushed/2025-12-12/spheres_srp_pushed_N2_2025-12-12_103347.pt", "10, pushed, radius=0.3"),
                 ("flow_boost/output/spheres_in_cube_new/final_pushed/2025-12-12/spheres_srp_pushed_N2_2025-12-12_104038.pt", "10, pushed, radius=0.55"),]
        
        files = [("flow_boost/spheres_in_cube_new/my_test_data2.pt", "2 spheres"),
                 ("flow_boost/output/spheres_in_cube_new/final_pushed/2025-12-12/spheres_srp_pushed_N3_2025-12-12_104658.pt", "10, pushed, radius=0.05"),
                 ("flow_boost/output/spheres_in_cube_new/final_pushed/2025-12-12/spheres_srp_pushed_N3_2025-12-12_104755.pt", "10, pushed, radius=0.4"),
                 ("flow_boost/output/spheres_in_cube_new/final_pushed/2025-12-12/spheres_srp_pushed_N3_2025-12-12_110205.pt", "10, pushed, radius=0.4 no stat"),]
        
        files = [("flow_boost/output/spheres_in_cube_new/2026_experiments/191/3/training.pt", "training"),
                 ("flow_boost/output/spheres_in_cube_new/2026_experiments/191/3/merge5.pt", "iteration 5"),
                 ("flow_boost/output/spheres_in_cube_new/2026_experiments/191/3/merge10.pt", "iteration 10"),
                 ("flow_boost/output/spheres_in_cube_new/2026_experiments/191/3/merge15.pt", "iteration 15")]
        
        files = [("flow_boost/output/spheres_in_cube_new/2026_experiments/191/1/training.pt", "training"),
                 ("flow_boost/output/spheres_in_cube_new/2026_experiments/191/1/merge1.pt", "iteration 1"),
                 ("flow_boost/output/spheres_in_cube_new/2026_experiments/191/1/merge5.pt", "iteration 5"),]
        
        merges = get_all_file_names(folder="flow_boost/output/spheres_in_cube_new/2026_experiments/191/4_100_trainsize", labeler = lambda it: f"Iteration {it+1}")
        # [merges[i] for i in [0,42]]
        files = [("flow_boost/output/spheres_in_cube_new/2026_experiments/191/4_100_trainsize/training/srp_data_N191_2026-01-18_03-57-04.pt", "training")] + [merges[-1]]
                 

        
        

    if not files:
        print("No files provided. Exiting.")
        sys.exit(0)

    title = "Sphere Radius"

    out_dir_default = "flow_boost/output/spheres_in_cube_new/distribution_plots"
    out_dir = "default" # input(f"Output directory for plots (default: {out_dir_default}): ").strip()
    if not out_dir:
        out_dir = out_dir_default
    else:
        out_dir = os.path.join(out_dir_default, out_dir)

    plot_files_combined(
        files=[file[0] for file in files],
        labels=[file[1] for file in files],
        savepath=out_dir,
        plotmode="overlay",   # or "stacked"
        n_bins=80,
        title=title,
        show=True,
        filename="sphere_radii_hist.png",
    )

    # Optional: print max radius per set
    # print("\n=== Summary: max effective radius per dataset ===")
    # for f, lab in files:
    #     td = load_dataset(f)
    #     metr = compute_radii2(td)
    #     print(f"{lab}: max radius = {np.max(metr['radii']):.6f}, avg. radius = {np.mean(metr['radii'])} ({len(metr['radii'])} samples)")


if __name__ == "__main__":
    main()
    