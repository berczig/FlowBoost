# ---- OpenMP guard: must be FIRST (before numpy/torch/matplotlib) ----
import os
import platform
if platform.system() == "Darwin" and os.environ.get("SPHEREPACK_DISABLE_KMP_HACK") != "1":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import math
import torch
import itertools
from datetime import datetime
from numba import njit
from flow_boost import cfg
from flow_boost.spheres_in_cube.physics_push_PESC import eliminate_overlaps_box
from flow_boost.spheres_in_cube.best_results import load_best_results
from flow_boost.spheres_in_cube_12d.pipeline import PipelineState
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Config helper
# -----------------------------------------------------------------------------
EPS_SMALL = 1e-8

def _get_cfg(section, key, fallback):
    from flow_boost import cfg
    try:
        if isinstance(fallback, bool):
            return cfg.getboolean(section, key)
        if isinstance(fallback, int):
            return cfg.getint(section, key)
        if isinstance(fallback, float):
            return cfg.getfloat(section, key)
        return cfg.get(section, key)
    except Exception:
        return fallback


def _set_cfg(section, key, value):
    """Set a config value in the in-memory ConfigParser.

    This is intentionally runtime-only (does not write back to disk).
    """
    from flow_boost import cfg
    if not cfg.has_section(section):
        cfg.add_section(section)
    if isinstance(value, bool):
        cfg.set(section, key, "true" if value else "false")
    else:
        cfg.set(section, key, str(value))

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _linf_norm_with_eps(g):
    m = 0.0
    for k in range(g.size):
        a = g[k]
        if a < 0:
            a = -a
        if a > m:
            m = a
    return m + EPS_SMALL

# -----------------------------------------------------------------------------
# Min pairwise distance (flat and matrix forms)
# -----------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def min_pairwise_distance_flat(X_flat, N):
    """
    X_flat: (N*D,) in absolute coords
    returns min_{i<j} ||x_i - x_j||  (float)
    """
    if N < 2:
        return 0.0
    D = X_flat.size // N
    best = 1e300
    for i in range(N):
        for j in range(i + 1, N):
            dist_sq = 0.0
            base_i = i * D
            base_j = j * D
            for d in range(D):
                diff = X_flat[base_i + d] - X_flat[base_j + d]
                dist_sq += diff * diff
            dist = math.sqrt(dist_sq)
            if dist < best:
                best = dist
    return best

@njit(cache=True, fastmath=True)
def min_pairwise_distance(centers):
    """
    centers: (N, D)
    """
    N, D = centers.shape
    if N < 2:
        return 0.0
    best = 1e300
    for i in range(N):
        for j in range(i + 1, N):
            dist_sq = 0.0
            for d in range(D):
                diff = centers[i, d] - centers[j, d]
                dist_sq += diff * diff
            dist = math.sqrt(dist_sq)
            if dist < best:
                best = dist
    return best

# -----------------------------------------------------------------------------
# HARD CONSTRAINT PROJECTION:
# Enforce x_{i,d} in [m(X), L - m(X)], where m(X)=0.5*d_min(X)
# Projection is done via reflection (NOT clipping).
# -----------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _reflect_into_interval_inplace(x, a, b):
    """
    Reflect each coordinate into [a,b] even if far outside.
    a < b required.
    """
    length = b - a
    if length <= EPS_SMALL:
        # Degenerate interval: force to midpoint
        mid = 0.5 * (a + b)
        for k in range(x.size):
            x[k] = mid
        return

    twoL = 2.0 * length
    for k in range(x.size):
        v = x[k] - a  # shift to [0, length] target
        # bring into [0, 2*length)
        v = v - twoL * math.floor(v / twoL)
        # reflect if needed
        if v > length:
            v = twoL - v
        x[k] = a + v

@njit(cache=True, fastmath=True)
def project_minwall_inplace(X_flat, L, N, proj_iters=3):
    """
    Iteratively project into the configuration-dependent box:
      [m(X), L-m(X)]^D,  where m(X)=0.5*d_min(X).

    Because m depends on X, we do a few fixed-point iterations.
    """
    for _ in range(proj_iters):
        dmin = min_pairwise_distance_flat(X_flat, N)
        m = 0.5 * dmin

        # Keep it sane:
        if m < 0.0:
            m = 0.0
        # If m >= L/2, box collapses; keep a tiny feasible interval
        if m >= 0.5 * L:
            m = 0.5 * L - EPS_SMALL
            if m < 0.0:
                m = 0.0

        a = m
        b = L - m
        if b <= a + EPS_SMALL:
            # fall back: whole cube
            a = 0.0
            b = L

        _reflect_into_interval_inplace(X_flat, a, b)

# -----------------------------------------------------------------------------
# Objective for "true sphere packing at radius r" WITHOUT wall penalty:
# - Hard wall constraint handled by projection above.
# - Energy only penalizes sphere-sphere overlaps (dist < 2r).
# -----------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def compute_overlap_EL_grad_abs(X_flat, L, N, r):
    """
    Overlap-only energy in absolute coords:
      EL = sum_{i<j} max(0, 2r - dist)^2
    Gradient is w.r.t. X_flat.
    """
    D = X_flat.size // N
    EL = 0.0
    g = np.zeros(X_flat.size, dtype=X_flat.dtype)

    for i in range(N):
        base_i = i * D
        for j in range(i + 1, N):
            base_j = j * D
            dist_sq = 0.0
            for d in range(D):
                diff = X_flat[base_i + d] - X_flat[base_j + d]
                dist_sq += diff * diff
            dist = math.sqrt(dist_sq)

            over = 2.0 * r - dist
            if over > 0.0:
                EL += over * over
                inv = (1.0 / dist) if dist >= EPS_SMALL else 0.0
                for d in range(D):
                    u = (X_flat[base_i + d] - X_flat[base_j + d]) * inv
                    gg = 2.0 * over * u
                    g[base_i + d] -= gg
                    g[base_j + d] += gg

    return EL, g

@njit(cache=True, fastmath=True)
def compute_overlap_EL_abs(X_flat, L, N, r):
    EL, _ = compute_overlap_EL_grad_abs(X_flat, L, N, r)
    return EL

# -----------------------------------------------------------------------------
# SRP with hard minwall constraint via projection
# -----------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def SRP_hardwall(X0_flat, L, N, r, Imax, m_inner, sigma, beta, add_noise):
    """
    SRP-like annealed normalized gradient descent on overlap energy,
    with hard constraint: min distance to wall >= 0.5 * d_min(current config).
    """
    Xc = X0_flat.copy()
    project_minwall_inplace(Xc, L, N, proj_iters=3)

    eta = sigma
    for _ in range(Imax):
        if add_noise:
            Xc += np.random.uniform(-eta, eta, size=Xc.shape[0])
            project_minwall_inplace(Xc, L, N, proj_iters=3)

        for __ in range(m_inner):
            _, g = compute_overlap_EL_grad_abs(Xc, L, N, r)
            Xc -= (sigma * eta) * (g / _linf_norm_with_eps(g))
            project_minwall_inplace(Xc, L, N, proj_iters=2)

        eta *= beta

    return Xc

# -----------------------------------------------------------------------------
# Local optimizer replacement: projected gradient descent with backtracking
# (because bounds depend on X, L-BFGS-B isn't applicable)
# -----------------------------------------------------------------------------
def local_opt_projected(X0_flat, L, N, r, tol, maxiter, lr=0.05, backtrack_steps=10):
    x = X0_flat.copy()
    project_minwall_inplace(x, L, N, proj_iters=4)

    EL = float(compute_overlap_EL_abs(x, L, N, r))
    for _ in range(maxiter):
        EL_curr, g = compute_overlap_EL_grad_abs(x, L, N, r)
        g_inf = float(np.max(np.abs(g))) if g.size else 0.0
        if g_inf < tol:
            return x, float(EL_curr)

        step = lr
        improved = False
        for __ in range(backtrack_steps):
            x_try = x - step * g
            project_minwall_inplace(x_try, L, N, proj_iters=3)
            EL_try = float(compute_overlap_EL_abs(x_try, L, N, r))
            if EL_try <= float(EL_curr) + 1e-15:
                x = x_try
                EL = EL_try
                improved = True
                break
            step *= 0.5

        if not improved:
            # no progress; stop
            return x, float(EL_curr)

    return x, float(EL)

# -----------------------------------------------------------------------------
# Sampling & symmetries
# -----------------------------------------------------------------------------
def sample_uniform_points_full(dim, L, N, eps=1e-6):
    return np.random.uniform(eps, L - eps, size=(N, dim))

def get_cube_symmetry_matrices(dim):
    mats = []
    for perm in itertools.permutations(range(dim)):
        for signs in itertools.product([-1, 1], repeat=dim):
            M = np.zeros((dim, dim))
            for i in range(dim):
                M[i, perm[i]] = signs[i]
            mats.append(M)
    return mats

def apply_symmetries_to_data(data, L):
    """
    data: (M, D, N)
    """
    M, D, N = data.shape

    # The full hyperoctahedral symmetry group has size D! * 2^D.
    # This explodes very quickly (e.g., D=12 -> ~2e12 transforms) and will
    # effectively hang the pipeline. We guard and let the caller skip.
    try:
        num_syms = math.factorial(int(D)) * (2 ** int(D))
    except Exception:
        num_syms = float("inf")

    # D=6 => 46,080 (OK). D=7 => 645,120 (already heavy). D>=8 is huge.
    max_syms = 100_000
    if num_syms > max_syms:
        raise ValueError(
            f"Symmetry enrichment skipped: D={D} would generate {num_syms} transforms (> {max_syms}). "
            "Reduce dimension or disable symmetry enrichment."
        )

    mats = get_cube_symmetry_matrices(D)
    out = np.zeros((M * len(mats), D, N), dtype=data.dtype)
    center = L / 2
    idx = 0
    for i in range(M):
        coords = data[i].T  # (N,D)
        for mat in mats:
            T = (mat @ (coords - center).T).T + center
            out[idx] = T.T
            idx += 1
    return out

# -----------------------------------------------------------------------------
# Main generation: optional physics push + SRP_hardwall + local_opt_projected
# -----------------------------------------------------------------------------
def generate_dataset_push_srp(verbose=True):
    sec = "sample_generation_PP+PBTS"

    D        = _get_cfg(sec, "dimension",            3)
    L        = _get_cfg(sec, "bounding_box_width",   1.0)
    r        = _get_cfg(sec, "sphere_radius",        0.1)
    best_d   = _get_cfg(sec, "best_known_diameter",  2.0 * r)
    N        = _get_cfg(sec, "num_spheres",          10)
    M        = _get_cfg(sec, "num_samples",          100)
    dt       = _get_cfg(sec, "dt",                   1e-3)
    max_iter = _get_cfg(sec, "max_iter",             10000)
    tol      = _get_cfg(sec, "tol",                  1e-6)
    mode_bnd = _get_cfg(sec, "boundary_mode",        "reflect")

    Imax       = _get_cfg(sec, "srp_Imax",        500)
    m_inner    = _get_cfg(sec, "srp_m",           20)
    sigma_frac = _get_cfg(sec, "srp_sigma_frac",  0.2)
    sigma      = sigma_frac * L
    beta       = _get_cfg(sec, "srp_beta",        0.95)

    tol_opt     = _get_cfg(sec, "srp_tol",        1e-8)
    maxiter_opt = _get_cfg(sec, "srp_maxiter",    300)
    num_srp_restarts = _get_cfg(sec, "num_srp_restarts", 15)

    physics_push_mode = _get_cfg(sec, "physics_push_mode", True)

    print(f"[HardWall SRP] N={N}, D={D}, L={L}, r={r}, M={M}, restarts={num_srp_restarts}")
    print("[HardWall SRP] Hard constraint: dist-to-wall >= 0.5 * current min pairwise distance.")

    # Output filenames
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stamp_with_N = f"N{N}_{timestamp_str}"
    day_stamp = datetime.now().strftime("%Y-%m-%d")

    base_metrics = _get_cfg(sec, "output_filename_metrics", "srp_metrics_{DATE}.csv")
    metrics_fn = base_metrics.replace("{DATE}", stamp_with_N).replace("{DAY_DATE}", day_stamp)

    base_data = _get_cfg(sec, "output_filename", "srp_data_{DATE}.pt")
    data_fn = base_data.replace("{DATE}", stamp_with_N).replace("{DAY_DATE}", day_stamp)

    base_sym = _get_cfg(sec, "output_filename_sym", data_fn.replace(".pt", "_sym.pt"))
    sym_fn = base_sym.replace("{DATE}", stamp_with_N).replace("{DAY_DATE}", day_stamp)

    base_top = _get_cfg(sec, "output_filename_top", "srp_top_{DATE}.pt")
    top_fn = base_top.replace("{DATE}", stamp_with_N).replace("{DAY_DATE}", day_stamp)

    base_sym_top = _get_cfg(sec, "output_filename_sym_top", top_fn.replace(".pt", "_sym.pt"))
    sym_top_fn = base_sym_top.replace("{DATE}", stamp_with_N).replace("{DAY_DATE}", day_stamp)

    metrics_dir = os.path.dirname(metrics_fn)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_fn, "w") as mf:
        mf.write("sample,srp_restart,EL_before,EL_after,pre_min,post_min,excess\n")

    data = np.zeros((M * num_srp_restarts, D, N), dtype=np.float32)
    min_dists = []

    # per-sample best stats
    best_EL_before_list = [0.0] * M
    best_EL_after_list  = [0.0] * M
    best_pre_min_list   = [0.0] * M
    best_post_min_list  = [0.0] * M
    best_excess_list    = [float("inf")] * M
    best_restart_idx_list = [-1] * M

    for i in tqdm(range(M), desc=f"Generating samples (hard-wall)"):
        pts = sample_uniform_points_full(D, L, N).astype(np.float64)

        # Optional physics push (uses radius r). If enabled, we still project afterwards.
        if physics_push_mode:
            centers0, _ = eliminate_overlaps_box(
                pts, r, [L] * D,
                max_iter=max_iter, dt=dt, tol=tol,
                boundary_mode=mode_bnd, visualize=False, verbose=False
            )
            centers0 = centers0.astype(np.float64)
        else:
            centers0 = pts

        X0 = centers0.ravel().astype(np.float64)
        # enforce hard wall constraint from the beginning
        project_minwall_inplace(X0, L, N, proj_iters=6)
        centers0 = X0.reshape((N, D))

        init_min = float(min_pairwise_distance(centers0))
        best_min = init_min

        best_EL_before = math.inf
        best_EL_after  = math.inf
        best_pre_min   = init_min
        best_post_min  = init_min
        best_excess    = best_d - init_min
        best_restart   = -1

        for k in range(num_srp_restarts):
            add_noise = (k > 0)

            X_srp = SRP_hardwall(X0, L, N, r, Imax, m_inner, sigma, beta, add_noise)
            EL_before = float(compute_overlap_EL_abs(X_srp, L, N, r))

            # local projected improvement
            X_lo, EL_after = local_opt_projected(X_srp, L, N, r, tol_opt, maxiter_opt, lr=0.05)

            centers_opt = X_lo.reshape((N, D))
            pre_min = float(min_pairwise_distance(centers_opt))

            # Optional physics push again, then hard-wall projection again
            if physics_push_mode:
                centers_k, _ = eliminate_overlaps_box(
                    centers_opt, r, [L] * D,
                    max_iter=max_iter, dt=dt, tol=tol,
                    boundary_mode=mode_bnd, visualize=False, verbose=False
                )
                Xk = centers_k.astype(np.float64).ravel()
                project_minwall_inplace(Xk, L, N, proj_iters=6)
                centers_k = Xk.reshape((N, D))
            else:
                # already projected by local_opt_projected
                centers_k = centers_opt

            post_min = float(min_pairwise_distance(centers_k))
            excess = best_d - post_min

            with open(metrics_fn, "a") as mf:
                mf.write(f"{i},{k+1},{EL_before:.6f},{EL_after:.6f},{pre_min:.6f},{post_min:.6f},{excess:.6f}\n")

            if post_min > best_min:
                best_min = post_min
            if excess < best_excess:
                best_excess     = excess
                best_EL_before  = EL_before
                best_EL_after   = EL_after
                best_pre_min    = pre_min
                best_post_min   = post_min
                best_restart    = k + 1

            data[i * num_srp_restarts + k] = centers_k.T.astype(np.float32)

        min_dists.append(best_min)
        best_EL_before_list[i] = best_EL_before
        best_EL_after_list[i]  = best_EL_after
        best_pre_min_list[i]   = best_pre_min
        best_post_min_list[i]  = best_post_min
        best_excess_list[i]    = best_excess
        best_restart_idx_list[i] = best_restart

    # Save full dataset
    data_dir = os.path.dirname(data_fn)
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
    torch.save(torch.from_numpy(data), data_fn)
    print(f"[Save] Saved full dataset to {data_fn}")

    # Symmetrized full dataset
    #try:
    #    sym_data = apply_symmetries_to_data(data, L)
    #    sym_dir = os.path.dirname(sym_fn)
    #    if sym_dir:
    #        os.makedirs(sym_dir, exist_ok=True)
    #    torch.save(torch.from_numpy(sym_data), sym_fn)
    #    print(f"[Save] Saved symmetrized dataset to {sym_fn}")
    #except ValueError as e:
    #    print(f"Skipping symmetry enrichment: {e}")

    # Top subset (~25% by best min distance per sample)
    k_top = max(1, int(np.ceil(0.25 * M)))
    best_idx = np.argsort(min_dists)[-k_top:]
    top_data = data[best_idx]
    top_dir = os.path.dirname(top_fn)
    if top_dir:
        os.makedirs(top_dir, exist_ok=True)
    torch.save(torch.from_numpy(top_data), top_fn)
    print(f"[Save] Saved top {k_top} samples to {top_fn}")

    # Symmetrized top subset
    #try:
    #    sym_top = apply_symmetries_to_data(top_data, L)
    #    sym_top_dir = os.path.dirname(sym_top_fn)
    #    if sym_top_dir:
    #        os.makedirs(sym_top_dir, exist_ok=True)
    #    torch.save(torch.from_numpy(sym_top), sym_top_fn)
    #    print(f"[Save] Saved symmetrized top dataset to {sym_top_fn}")
    #except ValueError as e:
    #    print(f"Skipping symmetry enrichment for top samples: {e}")

    # Top-10 metrics (minimal excess)
    num_top10 = min(10, M)
    order_top10 = np.argsort(best_excess_list)[:num_top10]
    metrics_top10_fn = metrics_fn.replace(".csv", "_top10.csv")
    with open(metrics_top10_fn, "w") as mf:
        mf.write("sample,EL_before,EL_after,pre_min,post_min,excess,srp_restart\n")
        for idx in order_top10:
            mf.write(
                f"{idx},"
                f"{best_EL_before_list[idx]:.6f},"
                f"{best_EL_after_list[idx]:.6f},"
                f"{best_pre_min_list[idx]:.6f},"
                f"{best_post_min_list[idx]:.6f},"
                f"{best_excess_list[idx]:.6f},"
                f"{best_restart_idx_list[idx]}\n"
            )
    print(f"[Save] Saved top-10 metrics to {metrics_top10_fn}")
    return data_fn

# -----------------------------------------------------------------------------
# Multi-sphere-count training generation (unchanged logic)
# -----------------------------------------------------------------------------
def generate_dataset_push_srp_different_sphere_count():
    secmul = "sample_generation_PP+PBTS_multiple_sphere_num"
    secgen = "sample_generation_PP+PBTS"
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    num_spheres_start = _get_cfg(secmul, "num_spheres_start", 5)
    num_spheres_end   = _get_cfg(secmul, "num_spheres_end",   10)

    base_output_filename         = _get_cfg(secmul, "output_filename",           "srp_mult_{SPHERE_NUM}_{DATE}.pt").replace("{DATE}", timestamp_str)
    base_output_filename_top     = _get_cfg(secmul, "output_filename_top",       "srp_mult_top_{SPHERE_NUM}_{DATE}.pt").replace("{DATE}", timestamp_str)
    base_output_filename_metrics = _get_cfg(secmul, "output_filename_metrics",   "srp_mult_metrics_{SPHERE_NUM}_{DATE}.csv").replace("{DATE}", timestamp_str)

    box_sizes = load_best_results()
    bar = tqdm(range(num_spheres_start, num_spheres_end + 1))
    for sphere_num in bar:
        bar.set_description(f"Generating packings, N={sphere_num}", refresh=True)
        radius = 1 / box_sizes[sphere_num]

        cfg.set(secgen, "num_spheres",          str(sphere_num))
        cfg.set(secgen, "bounding_box_width",   "1.0")
        cfg.set(secgen, "sphere_radius",        str(radius))
        cfg.set(secgen, "best_known_diameter",  str(2 * radius))

        cfg.set(secgen, "output_filename",         base_output_filename.replace("{SPHERE_NUM}", str(sphere_num)))
        cfg.set(secgen, "output_filename_top",     base_output_filename_top.replace("{SPHERE_NUM}", str(sphere_num)))
        cfg.set(secgen, "output_filename_metrics", base_output_filename_metrics.replace("{SPHERE_NUM}", str(sphere_num)))

        generate_dataset_push_srp(verbose=False)

# -----------------------------------------------------------------------------
# FINAL PUSH MODE: load packings, apply SRP_hardwall + local_opt_projected
# -----------------------------------------------------------------------------
def final_push_existing_samples():
    sec = "sample_generation_PP+PBTS"

    D        = _get_cfg(sec, "dimension",            3)
    L        = _get_cfg(sec, "bounding_box_width",   1.0)
    r        = _get_cfg(sec, "sphere_radius",        0.1)
    best_d   = _get_cfg(sec, "best_known_diameter",  2.0 * r)
    N        = _get_cfg(sec, "num_spheres",          10)

    dt       = _get_cfg(sec, "dt",                   1e-3)
    max_iter = _get_cfg(sec, "max_iter",             10000)
    tol      = _get_cfg(sec, "tol",                  1e-6)
    mode_bnd = _get_cfg(sec, "boundary_mode",        "reflect")

    Imax       = _get_cfg(sec, "srp_Imax",        500)
    m_inner    = _get_cfg(sec, "srp_m",           20)
    sigma_frac = _get_cfg(sec, "srp_sigma_frac",  0.2)
    sigma      = sigma_frac * L
    beta       = _get_cfg(sec, "srp_beta",        0.95)

    tol_opt     = _get_cfg(sec, "srp_tol",        1e-8)
    maxiter_opt = _get_cfg(sec, "srp_maxiter",    300)
    num_srp_restarts = _get_cfg(sec, "num_srp_restarts", 15)

    physics_push_mode = _get_cfg(sec, "physics_push_mode", True)

    stamp      = datetime.now().strftime("%Y-%m-%d")
    out_dir    = _get_cfg(sec, "final_push_output", "./outputs_spheres_push")
    out_dir = os.path.join(out_dir, stamp)
    input_path = _get_cfg(sec, "final_push_input",  "")

    assert isinstance(input_path, str) and len(input_path) > 0 and os.path.exists(input_path), \
        "Set sample_generation_PP+PBTS.final_push_input to a valid .pt file"

    os.makedirs(out_dir, exist_ok=True)
    stamp2     = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dataset_fn = os.path.join(out_dir, f"spheres_srp_pushed_N{N}_{stamp2}.pt")
    metrics_fn = os.path.join(out_dir, f"spheres_metrics_pushed_N{N}_{stamp2}.csv")

    loaded = torch.load(input_path, map_location="cpu")
    if isinstance(loaded, torch.Tensor):
        arr = loaded.detach().cpu().numpy()
    elif hasattr(loaded, "data") and isinstance(loaded.data, torch.Tensor):
        arr = loaded.data.detach().cpu().numpy()
    else:
        raise ValueError(f"final_push_input must be a Tensor or object with .data Tensor; got {type(loaded)}")

    if arr.ndim != 3:
        raise ValueError(f"Expected shape (M,D,N) or (M,N,D), got {arr.shape}")

    if arr.shape[1] == D:
        K, _, N_in = arr.shape
    elif arr.shape[-1] == D:
        arr = np.transpose(arr, (0, 2, 1))  # to (M, D, N)
        K, _, N_in = arr.shape
    else:
        raise ValueError(f"Second or last dimension must be {D}; got {arr.shape}")

    if N_in != N:
        raise ValueError(f"Config num_spheres={N} but input samples have N={N_in}")

    with open(metrics_fn, "w") as mf:
        mf.write("sample,srp_restart,EL_before,EL_after,pre_min,post_min,excess\n")

    data_out = np.zeros((K * num_srp_restarts, D, N), dtype=np.float32)

    best_EL_before_list = [0.0] * K
    best_EL_after_list  = [0.0] * K
    best_pre_min_list   = [0.0] * K
    best_post_min_list  = [0.0] * K
    best_excess_list    = [float("inf")] * K
    best_restart_idx_list = [-1] * K

    min_dists = []

    for s in tqdm(range(K), desc="Pushing Samples (hard-wall)", unit="sample"):
        centers_in = arr[s].T.astype(np.float64)  # (N, D)

        if physics_push_mode:
            centers0, _ = eliminate_overlaps_box(
                centers_in, r, [L] * D,
                max_iter=max_iter, dt=dt, tol=tol,
                boundary_mode=mode_bnd, visualize=False, verbose=False
            )
            centers0 = centers0.astype(np.float64)
        else:
            centers0 = centers_in.copy()

        X0 = centers0.ravel()
        project_minwall_inplace(X0, L, N, proj_iters=6)

        best_post_min_sample = -1.0
        best_excess_sample = float("inf")
        best_EL_before = math.inf
        best_EL_after  = math.inf
        best_pre_min   = 0.0
        best_restart   = -1

        for k in range(num_srp_restarts):
            add_noise = (k > 0)
            X_srp = SRP_hardwall(X0, L, N, r, Imax, m_inner, sigma, beta, add_noise)
            EL_before = float(compute_overlap_EL_abs(X_srp, L, N, r))

            X_lo, EL_after = local_opt_projected(X_srp, L, N, r, tol_opt, maxiter_opt, lr=0.05)
            centers_opt = X_lo.reshape((N, D))
            pre_min = float(min_pairwise_distance(centers_opt))

            if physics_push_mode:
                centers_k, _ = eliminate_overlaps_box(
                    centers_opt, r, [L] * D,
                    max_iter=max_iter, dt=dt, tol=tol,
                    boundary_mode=mode_bnd, visualize=False, verbose=False
                )
                Xk = centers_k.astype(np.float64).ravel()
                project_minwall_inplace(Xk, L, N, proj_iters=6)
                centers_k = Xk.reshape((N, D))
            else:
                centers_k = centers_opt

            post_min = float(min_pairwise_distance(centers_k))
            excess = best_d - post_min

            with open(metrics_fn, "a") as mf:
                mf.write(f"{s},{k+1},{EL_before:.6f},{EL_after:.6f},{pre_min:.6f},{post_min:.6f},{excess:.6f}\n")

            if post_min > best_post_min_sample:
                best_post_min_sample = post_min
            if excess < best_excess_sample:
                best_excess_sample = excess
                best_EL_before = EL_before
                best_EL_after  = EL_after
                best_pre_min   = pre_min
                best_restart   = k + 1

            data_out[s * num_srp_restarts + k] = centers_k.T.astype(np.float32)

        min_dists.append(best_post_min_sample)
        best_EL_before_list[s] = best_EL_before
        best_EL_after_list[s]  = best_EL_after
        best_pre_min_list[s]   = best_pre_min
        best_post_min_list[s]  = best_post_min_sample
        best_excess_list[s]    = best_excess_sample
        best_restart_idx_list[s] = best_restart

    torch.save(torch.from_numpy(data_out), dataset_fn)
    print(f"[Save] Saved pushed dataset:  {dataset_fn}, shape {data_out.shape}")
    print(f"[Save] Saved pushed metrics:  {metrics_fn}")

    num_top10 = min(10, K)
    order_top10 = np.argsort(best_excess_list)[:num_top10]
    metrics_top10_fn = metrics_fn.replace(".csv", "_top10.csv")
    with open(metrics_top10_fn, "w") as mf:
        mf.write("sample,EL_before,EL_after,pre_min,post_min,excess,srp_restart\n")
        for idx in order_top10:
            mf.write(
                f"{idx},"
                f"{best_EL_before_list[idx]:.6f},"
                f"{best_EL_after_list[idx]:.6f},"
                f"{best_pre_min_list[idx]:.6f},"
                f"{best_post_min_list[idx]:.6f},"
                f"{best_excess_list[idx]:.6f},"
                f"{best_restart_idx_list[idx]}\n"
            )
    print(f"[Save] Saved top-10 pushed metrics: {metrics_top10_fn}")
    return dataset_fn

# -----------------------------------------------------------------------------
# Main mode switch
# -----------------------------------------------------------------------------
def main(state: PipelineState = None):
    main_sec = "sample_generation_PP+PBTS"
    mode = _get_cfg(main_sec, "mode", "training_set_gen").strip().lower()

    if mode == "training_set_gen":
        multi_sec = "sample_generation_PP+PBTS_multiple_sphere_num"
        multi_active = _get_cfg(multi_sec, "active", False)

        if multi_active:
            generate_dataset_push_srp_different_sphere_count()
        else:
            path = generate_dataset_push_srp(verbose=False)
            if state:
                state.set_samples_path(path)

    elif mode == "final_push":
        path = final_push_existing_samples()
        if state:
            state.set_pushed_samples_path(path)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'training_set_gen' or 'final_push'.")

if __name__ == "__main__":
    main()
