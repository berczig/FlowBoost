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


def load_dataset(file):
    if isinstance(file, torch.Tensor):
        return file
    return torch.load(file)


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

    return {"min_triangle_areas": min_areas}


def plot(Arrays, labels, savepath, n_bins=100,
         title="Min Triangle Area", xlabel="Min Triangle Area",
         ylabel="Frequency", plotmode="overlay", n_xticks=15):
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
    plt.show()
    # If you prefer files:
    # os.makedirs(savepath, exist_ok=True)
    # plt.savefig(os.path.join(savepath, "min_triangle_area_hist.png"), dpi=150, bbox_inches="tight")
    # plt.close()


def plot_files_combined(files, labels, savepath, **kwargs):
    arrays_min_area = []
    for file in files:
        dataset = load_dataset(file)
        metr = compute_metrics(dataset)
        arrays_min_area.append(metr["min_triangle_areas"])
    plot(Arrays=arrays_min_area, labels=labels, savepath=os.path.join(savepath, "min_triangle_area"),
         title="Normalized Min Triangle Area Frequency",
         xlabel="Min Triangle Area", ylabel="Frequency", **kwargs)


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
    training_data = "flow_boost/output/heilbronn_square/training_sets/heilbronn_srp_generated_2000x10_2025-10-22_120000.pt"
    gen_samples = "flow_boost/output/heilbronn_square/generated_sets/heilbronn_gen_300x10_20251022_145555.pt"
    pushed_samples = "flow_boost/output/heilbronn_square/fixed_gen_sets/heilbronn_srp_pushed_2025-10-23_103035.pt"

    # if training data is a dict, let training_data be the tensor corresponding to key "pushed"
    td_loaded = torch.load(training_data)
    if isinstance(td_loaded, dict):
        training_data = td_loaded["pushed"]

    plot_files_combined(
        [training_data, gen_samples, pushed_samples],
        ["Training data", "Samples (Flow matching)", "SRP Pushed Samples"],
        "output"
    )

    # Optional: print best min triangle area per set
    pushed_samples_data = load_dataset(pushed_samples)
    pushed_samples_metrics = compute_metrics(pushed_samples_data)
    print("Max Min Triangle Area (SRP Pushed Samples):", np.max(pushed_samples_metrics["min_triangle_areas"]))

    training_data_tensor = load_dataset(training_data)
    training_metrics = compute_metrics(training_data_tensor)
    print("Max Min Triangle Area (Training Data):", np.max(training_metrics["min_triangle_areas"]))
