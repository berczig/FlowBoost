import os, platform
# If you need the quick unblock, leave this enabled on macOS.
# Disable by exporting SPHEREPACK_DISABLE_KMP_HACK=1 in your shell.
if platform.system() == "Darwin" and os.environ.get("SPHEREPACK_DISABLE_KMP_HACK") != "1":
    # Intel notes this is an unsafe workaround; prefer the environment fix below.
    # It must be set BEFORE any library initializes OpenMP.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import combinations
from datetime import datetime


def load_dataset(file):
    if isinstance(file, torch.Tensor):
        return file
    loaded = torch.load(file)
    # Some of our saved artifacts are dicts (e.g. {"pushed": tensor, ...}).
    # For plotting, pick the most relevant tensor.
    if isinstance(loaded, dict):
        preferred_keys = ["pushed", "dataset", "data", "tensor"]
        for key in preferred_keys:
            value = loaded.get(key)
            if isinstance(value, torch.Tensor):
                return value
        # Fallback: first tensor-like value in dict
        for value in loaded.values():
            if isinstance(value, torch.Tensor):
                return value
        raise ValueError(
            f"Loaded a dict from '{file}', but it contained no torch.Tensor values. Keys={list(loaded.keys())}"
        )
    return loaded


def _min_triangle_area_sample(points_2xn: np.ndarray) -> float:
    """
    points_2xn: shape (2, N) in the unit square.
    Returns the minimum triangle area among all C(N,3) triples.
    Area formula: 0.5 * | (B - A) x (C - A) | where x is 2D cross (determinant).
    """
    # (N,2)
    P = points_2xn.T.astype(np.float64, copy=False)
    N = P.shape[0]
    if N < 3:
        return 0.0
    min_area = np.inf
    # Enumerate triples; for typical N (<= ~50) this is fine
    for i, j, k in combinations(range(N), 3):
        A = P[i]; B = P[j]; C = P[k]
        BA = B - A
        CA = C - A
        # 2D "cross product" magnitude (determinant)
        cross = BA[0]*CA[1] - BA[1]*CA[0]
        area = 0.5 * abs(cross)
        if area < min_area:
            min_area = area
            # tiny short-circuit if exactly 0 (collinear)
            if min_area == 0.0:
                break
    return float(min_area if np.isfinite(min_area) else 0.0)


def compute_metrics(tensor_data):
    """
    tensor_data: expected shapes (M,2,N) or (M,N,2) or torch.Tensor.
    Returns a dict with array of per-sample minimum triangle areas (Heilbronn).
    """
    if isinstance(tensor_data, torch.Tensor):
        arr = tensor_data.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor_data)

    # Normalize to (M,2,N)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape {arr.shape}")
    if arr.shape[1] == 2:
        data_2n = arr
    elif arr.shape[2] == 2:
        data_2n = np.transpose(arr, (0, 2, 1))
    else:
        raise ValueError(f"Second or last dimension must be 2; got shape {arr.shape}")

    M = data_2n.shape[0]
    min_areas = np.empty(M, dtype=np.float64)
    for idx in range(M):
        # data_2n[idx] has shape (2, N)
        min_areas[idx] = _min_triangle_area_sample(data_2n[idx])

    return {"min_triangle_areas": min_areas, "num_points": int(data_2n.shape[2])}


def plot(Arrays, labels, savepath=None, n_bins=100,
         title="Min Triangle Area", xlabel="Min Triangle Area",
         ylabel="Frequency", plotmode="overlay", n_xticks=15,
         save_dir=None, filename=None, show=False):
    max_val = -np.inf
    min_val = np.inf
    for values in Arrays:
        if len(values) == 0:
            continue
        max_val = max(max_val, np.max(values))
        min_val = min(min_val, np.min(values))

    if not np.isfinite(max_val) or not np.isfinite(min_val):
        raise ValueError("No finite values to plot.")

    max_val = max_val * 1.1
    min_val = max(0.0, min_val * 0.9)  # areas are nonnegative

    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    bin_widths = np.diff(bin_edges)
    bar_x_positions = (bin_edges[:-1] + bin_edges[1:]) / 2

    if plotmode == "overlay":
        max_freq = 0.0
        for index, values in enumerate(Arrays):
            if len(values) == 0:
                continue
            hist_data = np.histogram(values, bins=bin_edges)[0] / max(1, len(values))
            color = mpl.colors.hsv_to_rgb((index / max(1, len(Arrays)), 1, 1))
            max_freq = max(max_freq, np.max(hist_data))
            label = f"{labels[index]}[{len(values)} samples]"
            plt.bar(bar_x_positions, hist_data, width=bin_widths,
                    edgecolor='black', alpha=0.4, label=label, color=color)
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
                w = np.full(len(values), 1 / len(values))
            weights.append(w)
            labels_samples.append(f"{labels[index]}[{len(values)} samples]")
        plt.hist(datas, bin_edges, weights=weights, label=labels_samples)

    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlim(min_val, max_val)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    out_dir = save_dir if save_dir is not None else savepath
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_name = filename if filename else "min_triangle_area_hist.png"
        out_path = os.path.join(out_dir, out_name)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_files_combined(files, labels, savepath, **kwargs):
    arrays_min_area = []
    max_of_min_areas = []
    n_points = None
    for file in files:
        dataset = load_dataset(file)
        metr = compute_metrics(dataset)
        min_areas = metr["min_triangle_areas"]
        arrays_min_area.append(min_areas)
        max_of_min_areas.append(float(np.max(min_areas)) if len(min_areas) else float("nan"))
        if n_points is None:
            n_points = metr.get("num_points")

    # Always save into the fixed Heilbronn output folder
    save_dir = os.path.join(
        "flow_boost", "output", "heilbronn_square", "fixed_gen_sets", "distribution_plots"
    )
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    n_tag = f"N{int(n_points)}" if n_points is not None else "Nunknown"
    filename = f"heilbronn_min_triangle_area_hist_{n_tag}_{stamp}.png"

    title_n = f" (N={int(n_points)})" if n_points is not None else ""
    labels_with_stats = []
    for base_label, v in zip(labels, max_of_min_areas):
        if np.isfinite(v):
            labels_with_stats.append(f"{base_label} (max minA={v:.6g})")
        else:
            labels_with_stats.append(f"{base_label} (max minA=n/a)")

    plot(
        Arrays=arrays_min_area,
        labels=labels_with_stats,
        save_dir=save_dir,
        filename=filename,
        show=False,
        title=f"Normalized Min Triangle Area Frequency{title_n}",
        xlabel="Min Triangle Area",
        ylabel="Frequency",
        **kwargs,
    )


def plot_named_datasets(datasets, **kwargs):
    """datasets: list of (path_or_tensor, name)"""
    if not datasets:
        raise ValueError("datasets list is empty")
    files = [p for (p, _) in datasets]
    labels = [name for (_, name) in datasets]
    plot_files_combined(files, labels, savepath="output", **kwargs)


def plot_3d(dataset, title="plot"):
    if isinstance(dataset, str):
        dataset = torch.load(dataset)
    for data in dataset:
        ax = plt.axes(projection='3d')
        xdata, ydata, zdata = data
        ax.scatter3D(xdata, ydata, zdata, c=zdata)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(title)
        plt.show()


if __name__ == "__main__":
    # Edit this list directly (no argparse):
    # Each entry is (path_to_pt, label)
    DATASETS = [
        (
            "flow_boost/output/heilbronn_square/training_sets/heilbronn_srp_1500x13_2025-12-15_180226.pt",
            "Training data" 
        ),
        (
            "flow_boost/output/heilbronn_square/fixed_gen_sets/heilbronn_srp_pushed_2025-12-17_092749.pt",
            "1st iteration"
        ),
        (
            "flow_boost/output/heilbronn_square/fixed_gen_sets/heilbronn_srp_pushed_2025-12-20_025749.pt","2nd iteration"
        ),
        (
            "flow_boost/output/heilbronn_square/fixed_gen_sets/heilbronn_srp_pushed_2025-12-20_210041.pt", "3rd iteration"
        ),
    ]

    plot_named_datasets(DATASETS)

    # Optional: print max(min triangle area) per dataset
    for path, label in DATASETS:
        tensor = load_dataset(path)
        metrics = compute_metrics(tensor)
        print(f"Max Min Triangle Area ({label}):", np.max(metrics["min_triangle_areas"]))