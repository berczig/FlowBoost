# ---- OpenMP guard: must be FIRST (before numpy/torch/matplotlib) ----
import os
import platform
if platform.system() == "Darwin" and os.environ.get("SPHEREPACK_DISABLE_KMP_HACK") != "1":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import math
import torch
import itertools
from scipy.optimize import minimize
from datetime import datetime
import numba as nb
from numba import njit
from flow_boost import cfg
from flow_boost.spheres_in_cube.physics_push_PESC import eliminate_overlaps_box
from flow_boost.spheres_in_cube.best_results import load_best_results
from flow_boost.spheres_in_cube_new.pipeline import PipelineState
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Config helper (same style)
# -----------------------------------------------------------------------------
EPS_SMALL = 1e-12

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
    cfg.set(section, key, value)
    print(f"[CFG] Overwriting '{section}.{key}' to {value}")

# -----------------------------------------------------------------------------
# Objective: maximize the largest feasible sphere radius s
#
# We optimize over (X_rel, s), where:
#   - X_rel are N*D coordinates in [-L/2, L/2]
#   - s is a scalar radius (>= 0)
#
# Constraints we want (hard sphere + walls):
#   ||xi - xj|| >= 2s
#   s <= xi_d <= L - s   (for all i,d)
#
# We implement this as a differentiable penalty + a term that pushes s upward.
# -----------------------------------------------------------------------------
MU_S = 1.0  # weight for maximizing s via minimizing (-MU_S * s)

@njit(cache=True, fastmath=True)
def _pos_part(x):
    return x if x > 0.0 else 0.0

@njit(cache=True, fastmath=True)
def _linf_norm_with_eps(g):
    m = 0.0
    for k in range(g.size):
        a = g[k]
        if a < 0:
            a = -a
        if a > m:
            m = a
    return m + 1e-12

@njit(cache=True, fastmath=True)
def compute_EL_grad(X_state, L, N):
    """
    X_state = [X_rel (N*D), s]
      - X_rel in relative coords (centered), bounds [-L/2, L/2]
      - s is optimized (radius)
    Energy:
      sum_{i<j} (max(0, 2s - dist_ij))^2
    + sum_{i,d} (max(0, s - x_id))^2 + (max(0, x_id - (L - s)))^2
    - MU_S * s
    """
    total = X_state.size
    s = X_state[total - 1]
    if s < 0.0:
        s = 0.0  # keep penalty/grad stable even if SRP steps go slightly negative

    D = (total - 1) // N
    coords_rel = X_state[:N * D].reshape((N, D))
    half = 0.5 * L
    coords_abs = coords_rel + half  # now in [0, L] if inside bounds

    EL = 0.0
    grad_abs = np.zeros((N, D), dtype=coords_abs.dtype)
    grad_s = 0.0

    # ---------------------------
    # 1) Pairwise separation: dist >= 2s
    # ---------------------------
    for i in range(N):
        for j in range(i + 1, N):
            dist_sq = 0.0
            for d in range(D):
                diff = coords_abs[i, d] - coords_abs[j, d]
                dist_sq += diff * diff
            dist = math.sqrt(dist_sq) + 1e-18

            over = 2.0 * s - dist
            if over > 0.0:
                EL += over * over

                inv = 1.0 / dist
                # d/dx: -2*over * d(dist)/dx, with d(dist)/dx = (xi-xj)/dist
                # So grad contribution is +/- 2*over*(xi-xj)/dist
                for d in range(D):
                    dx = coords_abs[i, d] - coords_abs[j, d]
                    u = dx * inv
                    g = 2.0 * over * u
                    grad_abs[i, d] += g
                    grad_abs[j, d] -= g

                # d/ds: over = 2s - dist => d(over)/ds = 2
                # d(over^2)/ds = 2*over*2 = 4*over
                grad_s += 4.0 * over

    # ---------------------------
    # 2) Wall separation: s <= x <= L - s
    # ---------------------------
    for i in range(N):
        for d in range(D):
            x = coords_abs[i, d]

            # Left wall: require x >= s
            overL = s - x
            if overL > 0.0:
                EL += overL * overL
                grad_abs[i, d] += -2.0 * overL
                grad_s += 2.0 * overL

            # Right wall: require x <= L - s
            overR = x - (L - s)  # = x - L + s
            if overR > 0.0:
                EL += overR * overR
                grad_abs[i, d] += 2.0 * overR
                grad_s += 2.0 * overR

    # ---------------------------
    # 3) Push s upward: minimize (-MU_S*s)
    # ---------------------------
    EL += -MU_S * s
    grad_s += -MU_S

    # Map gradient back to relative coords (coords_abs = coords_rel + half => same derivative)
    grad_rel = grad_abs

    g = np.empty(total, dtype=X_state.dtype)
    g[:N * D] = grad_rel.ravel()
    g[total - 1] = grad_s
    return EL, g

@njit(cache=True, fastmath=True)
def compute_EL(X_state, L, N):
    EL, _ = compute_EL_grad(X_state, L, N)
    return EL

@njit(cache=True, fastmath=True)
def min_pairwise_distance(centers):
    """
    centers: (N, D)
    """
    N, D = centers.shape
    best = 1e300
    for i in range(N):
        for j in range(i + 1, N):
            dist_sq = 0.0
            for d in range(D):
                diff = centers[i, d] - centers[j, d]
                dist_sq += diff * diff
            dval = math.sqrt(dist_sq)
            if dval < best:
                best = dval
    return best

@njit(cache=True, fastmath=True)
def min_wall_distance(centers, L):
    """
    centers: (N, D) in [0,L]^D
    returns min over i,d of min(x, L-x)
    """
    N, D = centers.shape
    best = 1e300
    for i in range(N):
        for d in range(D):
            x = centers[i, d]
            a = x
            b = L - x
            m = a if a < b else b
            if m < best:
                best = m
    return best

@njit(cache=True, fastmath=True)
def minsep_diameter(centers, L):
    """
    Effective "diameter" allowed by both constraints:
      - pairwise: diameter <= min_pairwise_distance
      - walls:    radius <= min_wall_distance  => diameter <= 2*min_wall_distance
    """
    mp = min_pairwise_distance(centers)
    mw = min_wall_distance(centers, L)
    d2 = 2.0 * mw
    return mp if mp < d2 else d2

@njit(cache=True, fastmath=True)
def SRP(X_state, L, N, Imax, m, sigma, beta, add_noise):
    """
    Stochastic Relaxation Push (SRP) on (positions, s).
    Noise is added only to position coordinates (not to s).
    """
    eta = sigma
    Xc = X_state.copy()
    total = Xc.size
    pos_len = total - 1  # last entry is s
    for _ in range(Imax):
        if add_noise:
            # Only perturb positions; keep s as-is
            for k in range(pos_len):
                Xc[k] += np.random.uniform(-eta, eta)
        for __ in range(m):
            EL, g = compute_EL_grad(Xc, L, N)
            Xc -= (sigma * eta) * (g / _linf_norm_with_eps(g))
        eta *= beta
    return Xc

def local_opt(X_state, L, N, tol, maxiter):
    """
    Local optimization on (positions, s) with L-BFGS-B bounds.
    """
    def objective(x):
        EL, _ = compute_EL_grad(x, L, N)
        return EL

    def gradient(x):
        _, grad = compute_EL_grad(x, L, N)
        return grad

    x0 = X_state.copy()
    half = 0.5 * L
    eps = 1e-9

    total = x0.size
    D = (total - 1) // N
    # Positions in [-L/2, L/2], s in [eps, L/2 - eps]
    bounds = [(-half + eps, half - eps)] * (D * N) + [(eps, half - eps)]

    res = minimize(
        fun=objective,
        x0=x0,
        method='L-BFGS-B',
        jac=gradient,
        bounds=bounds,
        options={'ftol': tol, 'gtol': tol, 'maxiter': maxiter}
    )
    return res.x, res.fun

# -----------------------------------------------------------------------------
# Sampling & symmetries
# -----------------------------------------------------------------------------
def sample_uniform_points(dim, L, N):
    return np.random.uniform(0.0, L, size=(N, dim))

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
    M, D, N = data.shape
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
# Helpers: clipping state and converting to absolute centers
# -----------------------------------------------------------------------------
def _clip_state(X_state, L):
    half = 0.5 * L
    eps = 1e-9
    X = X_state.copy()
    # clip positions
    X[:-1] = np.clip(X[:-1], -half + eps, half - eps)
    # clip s
    X[-1] = float(np.clip(X[-1], eps, half - eps))
    return X

def _state_to_centers_abs(X_state, L, N):
    half = 0.5 * L
    total = X_state.size
    D = (total - 1) // N
    coords = X_state[:N * D].reshape((N, D))
    centers = coords + half
    # numerical safety
    centers = np.minimum(np.maximum(centers, 0.0 + 1e-12), L - 1e-12)
    s = float(X_state[-1])
    return centers, s

def _init_radius_from_centers(centers, L):
    # radius = min( min_pairwise/2, min_wall )
    mp = float(min_pairwise_distance(centers))
    mw = float(min_wall_distance(centers, L))
    s0 = min(0.5 * mp, mw)
    s0 = max(1e-6, s0 * 0.9)  # small safety margin
    half = 0.5 * L
    return min(s0, half - 1e-6)

# -----------------------------------------------------------------------------
# Main generation: SRP + local_opt + optional physics push (TRAINING MODE)
# -----------------------------------------------------------------------------
def generate_dataset_push_srp(verbose=True):
    sec = "sample_generation_PP+PBTS"

    # Read parameters (NOTE: sphere_radius is read but NOT USED as an optimization parameter)
    D        = _get_cfg(sec, "dimension",            3)
    L        = _get_cfg(sec, "bounding_box_width",   1.0)
    _r_cfg   = _get_cfg(sec, "sphere_radius",        0.1)  # unused (kept for config compatibility)
    best_d   = _get_cfg(sec, "best_known_diameter",  0.2)
    N        = _get_cfg(sec, "num_spheres",          10)
    M        = _get_cfg(sec, "num_samples",          100)
    dt       = _get_cfg(sec, "dt",                   1e-3)
    max_iter = _get_cfg(sec, "max_iter",             10000)
    tol      = _get_cfg(sec, "tol",                  1e-6)
    mode_bnd = _get_cfg(sec, "boundary_mode",        "reflect")

    Imax       = _get_cfg(sec, "srp_Imax",        500)
    m          = _get_cfg(sec, "srp_m",           20)
    sigma_frac = _get_cfg(sec, "srp_sigma_frac",  0.2)
    sigma      = sigma_frac * L
    beta       = _get_cfg(sec, "srp_beta",        0.95)
    tol_opt    = _get_cfg(sec, "srp_tol",         1e-8)
    maxiter_opt= _get_cfg(sec, "srp_maxiter",     300)
    num_srp_restarts = _get_cfg(sec, "num_srp_restarts", 15)

    physics_push_mode = _get_cfg(sec, "physics_push_mode", False)

    print(f"[Data Generation] N={N}, M={M}, SRP restarts={num_srp_restarts}, physics_push_mode={physics_push_mode}")
    print(f"[Info] Config sphere_radius={_r_cfg} is ignored. Radius is optimized per-sample via max-min separation objective.")

    # Output filenames (include N in the timestamp token)
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stamp_with_N = f"N{N}_{timestamp_str}"
    day_stamp = datetime.now().strftime("%Y-%m-%d")

    base_metrics = _get_cfg(sec, "output_filename_metrics", "srp_metrics_{DATE}.csv")
    metrics_fn = base_metrics.replace("{DATE}", stamp_with_N).replace("{DAY_DATE}", day_stamp)

    base_data = _get_cfg(sec, "output_filename", "srp_data_{DATE}.pt")
    data_fn = base_data.replace("{DATE}", stamp_with_N).replace("{DAY_DATE}", day_stamp)

    base_sym = _get_cfg(sec, "output_filename_sym", data_fn.replace('.pt', '_sym.pt'))
    sym_fn = base_sym.replace("{DATE}", stamp_with_N).replace("{DAY_DATE}", day_stamp)

    base_top = _get_cfg(sec, "output_filename_top", "srp_top_{DATE}.pt")
    top_fn = base_top.replace("{DATE}", stamp_with_N).replace("{DAY_DATE}", day_stamp)

    base_sym_top = _get_cfg(sec, "output_filename_sym_top", top_fn.replace('.pt', '_sym.pt'))
    sym_top_fn = base_sym_top.replace("{DATE}", stamp_with_N).replace("{DAY_DATE}", day_stamp)

    metrics_dir = os.path.dirname(metrics_fn)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_fn, 'w') as mf:
        # Keep same header shape; interpret pre/post as "effective diameter" (minsep_diameter)
        mf.write("sample,srp_restart,EL_before,EL_after,pre_push_min,post_push_min,excess\n")

    data = np.zeros((M * num_srp_restarts, D, N), dtype=np.float32)
    best_minseps = []

    # For per-sample best stats (for top10 metrics)
    best_EL_before_list = [0.0] * M
    best_EL_after_list  = [0.0] * M
    best_pre_min_list   = [0.0] * M
    best_post_min_list  = [0.0] * M
    best_excess_list    = [float('inf')] * M
    best_restart_idx_list = [-1] * M

    for i in tqdm(range(M), desc=f"Generating sample - will be used for {num_srp_restarts} restarts"):
        pts = sample_uniform_points(D, L, N).astype(np.float64)

        # Optional physics push: uses the current estimated radius s0 (not a fixed config radius)
        if physics_push_mode:
            s0 = _init_radius_from_centers(pts, L)
            centers0, _ = eliminate_overlaps_box(
                pts, s0, [L] * D,
                max_iter=max_iter, dt=dt, tol=tol,
                boundary_mode=mode_bnd, visualize=False, verbose=False
            )
        else:
            centers0 = pts.copy()

        half = 0.5 * L
        Xpos0 = (centers0 - half).ravel()
        s0 = _init_radius_from_centers(centers0, L)
        X0 = np.concatenate([Xpos0, np.array([s0], dtype=np.float64)])

        init_minsep = float(minsep_diameter(centers0, L))
        best_centers = centers0.copy()
        best_minsep = init_minsep
        excess_sample = best_d - init_minsep

        # "best" metrics for this sample
        best_EL_before = math.inf
        best_EL_after  = math.inf
        best_pre_min   = init_minsep
        best_post_min  = init_minsep
        best_excess    = best_d - init_minsep
        best_restart   = -1

        for k in range(num_srp_restarts):
            add_noise = (k > 0)
            X_srp = SRP(X0, L, N, Imax, m, sigma, beta, add_noise)
            X_srp = _clip_state(X_srp, L)

            EL_before = float(compute_EL(X_srp, L, N))
            X_lo, EL_after = local_opt(X_srp, L, N, tol_opt, maxiter_opt)

            X_lo = _clip_state(X_lo, L)

            centers_opt, s_opt = _state_to_centers_abs(X_lo, L, N)
            pre_minsep = float(minsep_diameter(centers_opt, L))

            # Optional physics push (again using s_opt as radius estimate)
            if physics_push_mode:
                s_push = max(1e-6, min(s_opt, 0.5 * L - 1e-6))
                centers_k, _ = eliminate_overlaps_box(
                    centers_opt, s_push, [L] * D,
                    max_iter=max_iter, dt=dt, tol=tol,
                    boundary_mode=mode_bnd, visualize=False, verbose=False
                )
            else:
                centers_k = centers_opt

            post_minsep = float(minsep_diameter(centers_k, L))
            excess = best_d - post_minsep

            with open(metrics_fn, 'a') as mf:
                mf.write(f"{i},{k+1},{EL_before:.6f},{float(EL_after):.6f},{pre_minsep:.6f},{post_minsep:.6f},{excess:.6f}\n")

            # Update best over restarts
            if post_minsep > best_minsep:
                best_minsep = post_minsep
                best_centers = centers_k.copy()
            if excess < best_excess:
                best_excess    = excess
                best_EL_before = EL_before
                best_EL_after  = float(EL_after)
                best_pre_min   = pre_minsep
                best_post_min  = post_minsep
                best_restart   = k + 1

            data[i * num_srp_restarts + k] = centers_k.copy().T.astype(np.float32)

        best_minseps.append(best_minsep)
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
    try:
        sym_data = apply_symmetries_to_data(data, L)
        sym_dir = os.path.dirname(sym_fn)
        if sym_dir:
            os.makedirs(sym_dir, exist_ok=True)
        torch.save(torch.from_numpy(sym_data), sym_fn)
        print(f"[Save] Saved symmetrized dataset to {sym_fn}")
    except ValueError as e:
        print(f"[Warn] Skipping symmetry enrichment: {e}")

    # Top-k subset (by best minsep_diameter)
    k_top = max(1, int(np.ceil(0.25 * M)))
    best_idx = np.argsort(np.array(best_minseps))[-k_top:]
    top_data = data[best_idx]
    top_dir = os.path.dirname(top_fn)
    if top_dir:
        os.makedirs(top_dir, exist_ok=True)
    torch.save(torch.from_numpy(top_data), top_fn)
    print(f"[Save] Saved top {k_top} samples to {top_fn}")

    # Symmetrized top subset
    try:
        sym_top = apply_symmetries_to_data(top_data, L)
        sym_top_dir = os.path.dirname(sym_top_fn)
        if sym_top_dir:
            os.makedirs(sym_top_dir, exist_ok=True)
        torch.save(torch.from_numpy(sym_top), sym_top_fn)
        print(f"[Save] Saved symmetrized top dataset to {sym_top_fn}")
    except ValueError as e:
        print(f"[Warn] Skipping symmetry enrichment for top samples: {e}")

    # Save metrics of TOP 10 samples (minimal excess)
    num_top10 = min(10, M)
    order_top10 = np.argsort(best_excess_list)[:num_top10]  # smaller excess is better
    metrics_top10_fn = metrics_fn.replace(".csv", "_top10.csv")
    with open(metrics_top10_fn, "w") as mf:
        mf.write("sample,EL_before,EL_after,pre_push_min,post_push_min,excess,srp_restart\n")
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
# Multi-sphere-count training generation (kept; uses load_best_results for radius,
# but the actual optimizer does NOT take a fixed r parameter).
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
    base_output_filename_metrics_excess = _get_cfg(secmul, "output_filename_metrics_excess", "srp_mult_metrics_excess_{SPHERE_NUM}_{DATE}.csv").replace("{DATE}", timestamp_str)

    print("base_output_filename: ", base_output_filename)
    print(f"generate multiple packing starting from {num_spheres_start} spheres, up to {num_spheres_end} spheres per packing")
    box_sizes = load_best_results()
    bar = tqdm(range(num_spheres_start, num_spheres_end + 1))
    for sphere_num in bar:
        bar.set_description(f"Generating packings from {num_spheres_start} to {num_spheres_end}, Currently at {sphere_num} spheres", refresh=True)

        # Keep writing these for compatibility with existing tooling (not used by optimizer)
        radius = 1 / box_sizes[sphere_num]

        cfg.set(secgen, "num_spheres",          str(sphere_num))
        cfg.set(secgen, "bounding_box_width",   "1.0")
        cfg.set(secgen, "sphere_radius",        str(radius))      # ignored by optimizer
        cfg.set(secgen, "best_known_diameter",  str(2 * radius))  # used only for "excess" logging

        cfg.set(secgen, "output_filename",                 base_output_filename.replace("{SPHERE_NUM}", str(sphere_num)))
        cfg.set(secgen, "output_filename_top",             base_output_filename_top.replace("{SPHERE_NUM}", str(sphere_num)))
        cfg.set(secgen, "output_filename_metrics",         base_output_filename_metrics.replace("{SPHERE_NUM}", str(sphere_num)))
        cfg.set(secgen, "output_filename_metrics_excess",  base_output_filename_metrics_excess.replace("{SPHERE_NUM}", str(sphere_num)))

        generate_dataset_push_srp(verbose=False)

def load_metrics_PP_p_PBTS(filename):
    data_excess = []
    with open(filename) as m_file:
        lines = m_file.readlines()
        for data_text in lines[1:]:
            data_string = data_text.split(",")
            data_excess.append(float(data_string[-1]))
    return data_excess

# -----------------------------------------------------------------------------
# FINAL PUSH MODE: read packings from file, apply SRP + local opt + optional physics push
# -----------------------------------------------------------------------------
def final_push_existing_samples():
    sec = "sample_generation_PP+PBTS"

    # Shared geometric / physical parameters
    D        = _get_cfg(sec, "dimension",            3)
    L        = _get_cfg(sec, "bounding_box_width",   1.0)
    _r_cfg   = _get_cfg(sec, "sphere_radius",        0.1)  # ignored by optimizer
    best_d   = _get_cfg(sec, "best_known_diameter",  0.2)
    N        = _get_cfg(sec, "num_spheres",          10)
    dt       = _get_cfg(sec, "dt",                   1e-3)
    max_iter = _get_cfg(sec, "max_iter",             10000)
    tol      = _get_cfg(sec, "tol",                  1e-6)
    mode_bnd = _get_cfg(sec, "boundary_mode",        "reflect")

    # SRP / local optimization parameters
    Imax       = _get_cfg(sec, "srp_Imax",        500)
    m          = _get_cfg(sec, "srp_m",           20)
    sigma_frac = _get_cfg(sec, "srp_sigma_frac",  0.2)
    sigma      = sigma_frac * L
    beta       = _get_cfg(sec, "srp_beta",        0.95)
    tol_opt    = _get_cfg(sec, "srp_tol",         1e-8)
    maxiter_opt= _get_cfg(sec, "srp_maxiter",     300)
    num_srp_restarts = _get_cfg(sec, "num_srp_restarts", 15)

    physics_push_mode = _get_cfg(sec, "physics_push_mode", False)

    # IO paths for final push
    stamp      = datetime.now().strftime("%Y-%m-%d")
    out_dir    = _get_cfg(sec, "final_push_output", "./outputs_spheres_push")
    out_dir = os.path.join(out_dir, stamp)
    input_path = _get_cfg(sec, "final_push_input",  "")

    assert isinstance(input_path, str) and len(input_path) > 0 and os.path.exists(input_path), \
        "Set sample_generation_PP+PBTS.final_push_input to a valid .pt file"

    os.makedirs(out_dir, exist_ok=True)
    stamp      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dataset_fn = os.path.join(out_dir, f"spheres_srp_pushed_N{N}_{stamp}.pt")
    metrics_fn = os.path.join(out_dir, f"spheres_metrics_pushed_N{N}_{stamp}.csv")

    loaded = torch.load(input_path, map_location="cpu")
    arr = loaded.detach().cpu().numpy() if isinstance(loaded, torch.Tensor) else None
    if arr is None or arr.ndim != 3:
        raise ValueError(f"Expected a tensor at final_push_input, got shape {getattr(loaded, 'shape', None)}")

    # Expect either (M, D, N) or (M, N, D)
    if arr.shape[1] == D:
        M_in, d_in, N_in = arr.shape
    elif arr.shape[-1] == D:
        arr = np.transpose(arr, (0, 2, 1))  # to (M, D, N)
        M_in, d_in, N_in = arr.shape
    else:
        raise ValueError(f"Second or last dimension must be {D}; got shape {arr.shape}")

    if N_in != N:
        raise ValueError(f"Config num_spheres={N} but input samples have N={N_in}")

    K = M_in
    print(f"[Push] Pushing {K} loaded samples (N={N}, D={D}), physics_push_mode={physics_push_mode}")
    print(f"[Info] Config sphere_radius={_r_cfg} is ignored. Radius is optimized per-sample via max-min separation objective.")

    with open(metrics_fn, "w") as mf:
        mf.write("sample,srp_restart,EL_before,EL_after,pre_push_min,post_push_min,excess\n")

    data_out = np.zeros((K * num_srp_restarts, D, N), dtype=np.float32)
    half = 0.5 * L

    best_minseps = []

    # Arrays for top-10 stats
    best_EL_before_list = [0.0] * K
    best_EL_after_list  = [0.0] * K
    best_pre_min_list   = [0.0] * K
    best_post_min_list  = [0.0] * K
    best_excess_list    = [float('inf')] * K
    best_restart_idx_list = [-1] * K

    for sidx in tqdm(range(K), desc="Pushing Samples", unit="sample"):
        centers_in = arr[sidx].T.astype(np.float64)  # (N, D) in [0,L]^D (assumed)

        if physics_push_mode:
            s0 = _init_radius_from_centers(centers_in, L)
            centers0, _ = eliminate_overlaps_box(
                centers_in, s0, [L] * D,
                max_iter=max_iter, dt=dt, tol=tol,
                boundary_mode=mode_bnd, visualize=False, verbose=False
            )
        else:
            centers0 = centers_in.copy()

        Xpos0 = (centers0 - half).ravel()
        s0 = _init_radius_from_centers(centers0, L)
        X0 = np.concatenate([Xpos0, np.array([s0], dtype=np.float64)])

        # per-sample best
        best_post_min_sample = -1.0
        best_excess_sample = float('inf')
        best_EL_before = math.inf
        best_EL_after  = math.inf
        best_pre_min   = 0.0
        best_restart   = -1
        best_centers   = centers0.copy()

        for k in range(num_srp_restarts):
            add_noise = (k > 0)
            X_srp = SRP(X0, L, N, Imax, m, sigma, beta, add_noise)
            X_srp = _clip_state(X_srp, L)
            EL_before = float(compute_EL(X_srp, L, N))

            X_lo, EL_after = local_opt(X_srp, L, N, tol_opt, maxiter_opt)
            X_lo = _clip_state(X_lo, L)

            centers_opt, s_opt = _state_to_centers_abs(X_lo, L, N)
            pre_minsep = float(minsep_diameter(centers_opt, L))

            if physics_push_mode:
                s_push = max(1e-6, min(s_opt, 0.5 * L - 1e-6))
                centers_k, _ = eliminate_overlaps_box(
                    centers_opt, s_push, [L] * D,
                    max_iter=max_iter, dt=dt, tol=tol,
                    boundary_mode=mode_bnd, visualize=False, verbose=False
                )
            else:
                centers_k = centers_opt

            post_minsep = float(minsep_diameter(centers_k, L))
            excess = best_d - post_minsep

            with open(metrics_fn, "a") as mf:
                mf.write(f"{sidx},{k+1},{EL_before:.6f},{float(EL_after):.6f},{pre_minsep:.6f},{post_minsep:.6f},{excess:.6f}\n")

            if post_minsep > best_post_min_sample:
                best_post_min_sample = post_minsep
                best_centers = centers_k.copy()
            if excess < best_excess_sample:
                best_excess_sample = excess
                best_EL_before = EL_before
                best_EL_after  = float(EL_after)
                best_pre_min   = pre_minsep
                best_restart   = k + 1

            data_out[sidx * num_srp_restarts + k] = centers_k.copy().T.astype(np.float32)

        best_minseps.append(best_post_min_sample)
        best_EL_before_list[sidx] = best_EL_before
        best_EL_after_list[sidx]  = best_EL_after
        best_pre_min_list[sidx]   = best_pre_min
        best_post_min_list[sidx]  = best_post_min_sample
        best_excess_list[sidx]    = best_excess_sample
        best_restart_idx_list[sidx] = best_restart

    torch.save(torch.from_numpy(data_out), dataset_fn)
    print(f"\nSaved pushed dataset:  {dataset_fn}, shape {data_out.shape}")
    print(f"Saved pushed metrics:  {metrics_fn}")

    # Top-10 metrics (minimal excess)
    num_top10 = min(10, K)
    order_top10 = np.argsort(best_excess_list)[:num_top10]
    metrics_top10_fn = metrics_fn.replace(".csv", "_top10.csv")
    with open(metrics_top10_fn, "w") as mf:
        mf.write("sample,EL_before,EL_after,pre_push_min,post_push_min,excess,srp_restart\n")
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
    print(f"Saved top-10 pushed metrics: {metrics_top10_fn}")
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
            data_save_path = generate_dataset_push_srp(verbose=False)
            if state:
                state.set_samples_path(data_save_path)

    elif mode == "final_push":
        data_save_path = final_push_existing_samples()
        if state:
            state.set_pushed_samples_path(data_save_path)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'training_set_gen' or 'final_push'.")

if __name__ == "__main__":
    main()
