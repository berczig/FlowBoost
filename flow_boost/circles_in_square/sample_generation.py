# ---- OpenMP guard: must be FIRST (before numpy/torch/matplotlib) ----
import os, platform
if platform.system() == "Darwin" and os.environ.get("SPHEREPACK_DISABLE_KMP_HACK") != "1":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from scipy.optimize import minimize

import numba as nb
from numba import njit

# Optional config support
try:
    from flow_boost import cfg
    HAS_CFG = True
except Exception:
    HAS_CFG = False

# =============================================================================
# Utilities / small constants
# =============================================================================
EPS = 1e-12

def _get_cfg(section, key, fallback):
    if HAS_CFG:
        try:
            if isinstance(fallback, int):
                return cfg.getint(section, key, fallback=fallback)
            if isinstance(fallback, float):
                return cfg.getfloat(section, key, fallback=fallback)
            if isinstance(fallback, bool):
                return cfg.getboolean(section, key, fallback=fallback)
            return cfg.get(section, key, fallback=fallback)
        except Exception:
            return fallback
    return fallback

def _set_cfg(section, key, value):
    """
    Minimal setter used by pipeline scripts.
    Safe no-op if cfg is unavailable.
    """
    if HAS_CFG:
        try:
            if not cfg.has_section(section):
                cfg.add_section(section)
            cfg.set(section, key, str(value))
        except Exception:
            pass

# =============================================================================
# Random initialization
# =============================================================================
def sample_uniform_centers(N):
    return np.random.uniform(0.0, 1.0, size=(N, 2)).astype(np.float64)

def init_radii(N, r0=0.01, jitter=0.005):
    r = r0 + jitter * (np.random.rand(N) - 0.5) * 2.0
    r = np.maximum(r, 1e-4)
    return r.astype(np.float64)

# =============================================================================
# Objective: overlap + wall penalties - alpha * sum r
# X = [x0,y0,...,x_{N-1},y_{N-1}, r0,...,r_{N-1}]
# =============================================================================
@njit(cache=True, fastmath=True)
def _unit(dx, dy):
    d = math.sqrt(dx*dx + dy*dy) + EPS
    return dx/d, dy/d, d

@njit(cache=True, fastmath=True)
def loss_and_grad(X, N, w_overlap, w_wall, alpha):
    centers = X[:2*N].reshape((N, 2))
    radii   = X[2*N:2*N+N]

    L = 0.0
    g = np.zeros_like(X)

    # Walls
    for i in range(N):
        ri = radii[i]
        xi = centers[i, 0]
        yi = centers[i, 1]
        # x-left
        v = ri - xi
        if v > 0.0:
            L += w_wall * (v*v)
            g[2*i + 0] += -2.0 * w_wall * v
            g[2*N + i] +=  2.0 * w_wall * v
        # x-right
        v = ri - (1.0 - xi)
        if v > 0.0:
            L += w_wall * (v*v)
            g[2*i + 0] +=  2.0 * w_wall * v
            g[2*N + i] +=  2.0 * w_wall * v
        # y-bottom
        v = ri - yi
        if v > 0.0:
            L += w_wall * (v*v)
            g[2*i + 1] += -2.0 * w_wall * v
            g[2*N + i] +=  2.0 * w_wall * v
        # y-top
        v = ri - (1.0 - yi)
        if v > 0.0:
            L += w_wall * (v*v)
            g[2*i + 1] +=  2.0 * w_wall * v
            g[2*N + i] +=  2.0 * w_wall * v

    # Pair overlaps
    for i in range(N):
        xi0, yi0, ri = centers[i,0], centers[i,1], radii[i]
        for j in range(i+1, N):
            dx = xi0 - centers[j,0]
            dy = yi0 - centers[j,1]
            ux, uy, d = _unit(dx, dy)
            over = (ri + radii[j]) - d
            if over > 0.0:
                L += w_overlap * (over * over)
                c = 2.0 * w_overlap * over
                # i center  (NOTE the minus signs)
                g[2*i + 0] += -c * ux
                g[2*i + 1] += -c * uy
                # j center  (opposite)
                g[2*j + 0] +=  c * ux
                g[2*j + 1] +=  c * uy
                # radii
                g[2*N + i] +=  c
                g[2*N + j] +=  c

    # Maximize sum radii (minimize -sum radii)
    L += -alpha * np.sum(radii)
    for i in range(N):
        g[2*N + i] += -alpha

    return L, g

# =============================================================================
# SRP (adaptive) with backtracking
# =============================================================================
@njit(cache=True, fastmath=True)
def _l2_norm(v):
    s = 0.0
    for k in range(v.size):
        s += v[k]*v[k]
    return math.sqrt(s) + 1e-18

@njit(cache=True, fastmath=True)
def srp_adaptive(X, N, Imax, m, step_center, step_radius, beta, backtrack,
                 w_overlap, w_wall, alpha):
    Xc = X.copy()
    eta = 1.0
    for _ in range(Imax):
        # jitter
        noise = np.zeros_like(Xc)
        for i in range(N):
            noise[2*i+0] = (np.random.rand()*2.0 - 1.0) * (eta * step_center)
            noise[2*i+1] = (np.random.rand()*2.0 - 1.0) * (eta * step_center)
        for i in range(N):
            noise[2*N + i] = (np.random.rand()*2.0 - 1.0) * (eta * step_radius)
        Xtrial = Xc + noise

        # m gradient steps with simple backtracking
        for __ in range(m):
            L0, g = loss_and_grad(Xtrial, N, w_overlap, w_wall, alpha)

            step = np.empty_like(g)
            for i in range(N):
                step[2*i+0] = eta * step_center
                step[2*i+1] = eta * step_center
            for i in range(N):
                step[2*N + i] = eta * step_radius

            gn = _l2_norm(g)
            if gn > 0.0:
                Xprop = Xtrial - step * (g / gn)
                Lprop, _ = loss_and_grad(Xprop, N, w_overlap, w_wall, alpha)

                bt = 0
                while Lprop > L0 and bt < backtrack:
                    for i in range(N):
                        step[2*i+0] *= 0.5
                        step[2*i+1] *= 0.5
                    for i in range(N):
                        step[2*N + i] *= 0.5
                    Xprop = Xtrial - step * (g / gn)
                    Lprop, _ = loss_and_grad(Xprop, N, w_overlap, w_wall, alpha)
                    bt += 1

                Xtrial = Xprop

        # clip to domain for stability
        for i in range(N):
            if Xtrial[2*i+0] < 0.0: Xtrial[2*i+0] = 0.0
            if Xtrial[2*i+0] > 1.0: Xtrial[2*i+0] = 1.0
            if Xtrial[2*i+1] < 0.0: Xtrial[2*i+1] = 0.0
            if Xtrial[2*i+1] > 1.0: Xtrial[2*i+1] = 1.0
        for i in range(N):
            if Xtrial[2*N + i] < 0.0: Xtrial[2*N + i] = 0.0
            if Xtrial[2*N + i] > 0.5: Xtrial[2*N + i] = 0.5

        Xc = Xtrial
        eta *= beta
    return Xc

# =============================================================================
# Local optimization (L-BFGS-B)
# =============================================================================
def local_optimize(X0, N, w_overlap, w_wall, alpha, gtol=1e-8, ftol=1e-12,
                   maxiter=1000, maxcor=20):
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N

    def fun(x):
        L, _ = loss_and_grad(x, N, w_overlap, w_wall, alpha)
        return L

    def jac(x):
        _, g = loss_and_grad(x, N, w_overlap, w_wall, alpha)
        return g

    res = minimize(
        fun=fun, x0=X0, method='L-BFGS-B', jac=jac, bounds=bounds,
        options={'gtol': gtol, 'ftol': ftol, 'maxiter': maxiter, 'maxcor': maxcor}
    )
    return res.x, res.fun

# =============================================================================
# Metrics
# =============================================================================
@njit(cache=True, fastmath=True)
def min_wall_clearance(centers, radii):
    N = centers.shape[0]
    best = 1e9
    for i in range(N):
        xi, yi = centers[i,0], centers[i,1]
        ri = radii[i]
        c1 = xi - ri
        if c1 < best: best = c1
        c2 = 1.0 - xi - ri
        if c2 < best: best = c2
        c3 = yi - ri
        if c3 < best: best = c3
        c4 = 1.0 - yi - ri
        if c4 < best: best = c4
    return best

@njit(cache=True, fastmath=True)
def min_pair_clearance(centers, radii):
    N = centers.shape[0]
    best = 1e9
    for i in range(N):
        xi, yi, ri = centers[i,0], centers[i,1], radii[i]
        for j in range(i+1, N):
            dx = xi - centers[j,0]
            dy = yi - centers[j,1]
            d  = math.sqrt(dx*dx + dy*dy)
            clr = d - (ri + radii[j])
            if clr < best:
                best = clr
    return best

def hard_project_max_sum_radii(centers, radii, safety=1e-9, pair_safety_mul=1.0):
    """
    Given centers (N,2) in [0,1]^2 and current radii (N,),
    compute radii' that:
      maximize sum(r_i)
      subject to:
        0 <= r_i <= wall_i
        r_i + r_j <= d_ij
    and then shrink by a tiny `safety` to avoid visual tangency.
    Returns: radii_proj (N,), info dict with diagnostics.
    """
    import numpy as np
    from scipy.optimize import linprog

    # --- diagnostics helpers ---
    def _min_wall_clear(cent, rr):
        xi, yi = cent[:, 0], cent[:, 1]
        return float(np.min([xi - rr, 1.0 - xi - rr, yi - rr, 1.0 - yi - rr]))

    def _min_pair_clear(cent, rr):
        dx = cent[:, None, 0] - cent[None, :, 0]
        dy = cent[:, None, 1] - cent[None, :, 1]
        D = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(D, np.inf)
        S = rr[:, None] + rr[None, :]
        return float(np.min(D - S))

    N = centers.shape[0]
    x = centers[:, 0]
    y = centers[:, 1]

    # Wall upper bounds for each circle
    wall_ub = np.minimum.reduce([x, 1.0 - x, y, 1.0 - y])
    wall_ub = np.clip(wall_ub - safety, 0.0, None)  # tiny safety

    # Pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    D = np.sqrt(dx * dx + dy * dy)

    # Build LP:
    # maximize sum r  <=>  minimize -sum r
    c = -np.ones(N, dtype=float)

    # Bounds: 0 <= r_i <= wall_ub[i]
    bounds = [(0.0, float(wall_ub[i])) for i in range(N)]

    # Inequalities A_ub @ r <= b_ub
    rows = []
    rhs = []
    for i in range(N):
        for j in range(i + 1, N):
            dij = D[i, j]
            rhs_ij = max(dij * pair_safety_mul - safety, 0.0)
            row = np.zeros(N, dtype=float)
            row[i] = 1.0
            row[j] = 1.0
            rows.append(row)
            rhs.append(rhs_ij)

    if rows:
        A_ub = np.vstack(rows)
        b_ub = np.asarray(rhs, dtype=float)
    else:
        A_ub = None
        b_ub = None

    # Solve LP
    res = linprog(
        c,
        A_ub=A_ub, b_ub=b_ub,
        bounds=bounds,
        method="highs"
    )

    info = {
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "sum_r": None,
        "violations": None,
    }

    if not res.success:
        # Fallback: clamp to wall_ub (feasible w.r.t. walls; pairs may violate)
        r_proj = np.minimum(radii, wall_ub).copy()
        info["sum_r"] = float(np.sum(r_proj))
        info["violations"] = {
            "min_wall_clear": _min_wall_clear(centers, r_proj),
            "min_pair_clear": _min_pair_clear(centers, r_proj),
        }
        info["message"] = f"LP failed; returned clamped radii. HiGHS status {res.status}: {res.message}"
        return r_proj, info

    r_star = res.x

    # Final tiny shrink to guarantee strict feasibility in plotting
    r_proj = np.maximum(0.0, r_star - safety)

    mwc = _min_wall_clear(centers, r_proj)
    mpc = _min_pair_clear(centers, r_proj)

    info["sum_r"] = float(np.sum(r_proj))
    info["violations"] = {"min_wall_clear": mwc, "min_pair_clear": mpc}
    return r_proj, info

# =============================================================================
# Plotting
# =============================================================================
def plot_first_k_samples(data_tensor, k, out_dir, filename_prefix="sample"):
    """
    Draws the first k samples from data tensor of shape (M, 3, N) with rows [x, y, r].
    Creates one figure per sample and saves PNGs to out_dir.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    os.makedirs(out_dir, exist_ok=True)

    M = data_tensor.shape[0]
    k = min(k, M)

    for s in range(k):
        xy = data_tensor[s, :2, :].T  # (N,2)
        rr = data_tensor[s, 2, :]     # (N,)
        fig, ax = plt.subplots(figsize=(5,5))
        # unit square boundary
        ax.plot([0,1,1,0,0], [0,0,1,1,0])
        # draw circles
        for i in range(xy.shape[0]):
            c = Circle((xy[i,0], xy[i,1]), rr[i], fill=False)
            ax.add_patch(c)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # Put sum of radii to title
        sum_r = float(np.sum(rr))
        ax.set_title(f"Sample #{s}, num_circles= {xy.shape[0]}, sum_r = {sum_r:.4f})")
        out_path = os.path.join(out_dir, f"{filename_prefix}_{xy.shape[0]}_{sum_r:.4f}_{s}.png")
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

# =============================================================================
# Main generator
# =============================================================================
def generate_circle_packing_dataset():
    sec = "circle_packing_SRP"

    # Config / defaults
    N   = _get_cfg(sec, "num_circles",  50)
    M   = _get_cfg(sec, "num_samples",  5000)

    # SRP hyperparams
    Imax       = _get_cfg(sec, "srp_Imax",        400)
    m          = _get_cfg(sec, "srp_m",           30)
    beta       = _get_cfg(sec, "srp_beta",        0.985)
    backtrack  = _get_cfg(sec, "srp_backtrack",   3)
    step_center= _get_cfg(sec, "srp_step_center", 0.05)
    step_radius= _get_cfg(sec, "srp_step_radius", 0.01)

    # Penalty/objective weights
    w_overlap  = _get_cfg(sec, "w_overlap",       1.0)
    w_wall     = _get_cfg(sec, "w_wall",          1.0)
    alpha      = _get_cfg(sec, "alpha_sum_r",     1.0)

    # Local opt
    gtol       = _get_cfg(sec, "lbfgs_gtol",      1e-8)
    ftol       = _get_cfg(sec, "lbfgs_ftol",      1e-12)
    maxiter    = _get_cfg(sec, "lbfgs_maxiter",   1000)
    maxcor     = _get_cfg(sec, "lbfgs_maxcor",    20)

    # Initialization
    r0         = _get_cfg(sec, "init_r0",         0.01)
    rj         = _get_cfg(sec, "init_r_jitter",   0.005)

    # I/O + plotting
    out_dir    = _get_cfg(sec, "output_dir",      "./outputs_circle_packing")
    plot_k     = _get_cfg(sec, "plot_k",          0)
    plot_dir   = os.path.join(out_dir, "plots")

    os.makedirs(out_dir, exist_ok=True)
    stamp      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dataset_fn = os.path.join(out_dir, f"circle_srp_generated_{M}x{N}_{stamp}.pt")
    metrics_fn = os.path.join(out_dir, f"circle_srp_metrics_{M}x{N}_{stamp}.csv")

    print(f"Generating {M} circle packings in unit square (N={N})")
    with open(metrics_fn, "w") as mf:
        mf.write("sample,sum_r,min_wall_clear,min_pair_clear,loss_after\n")

    data = np.zeros((M, 3, N), dtype=np.float32)

    bar = tqdm(range(M), desc="Generating circle packings")
    for s in bar:
        # Random start
        C0 = sample_uniform_centers(N)
        R0 = init_radii(N, r0=r0, jitter=rj)
        X0 = np.concatenate([C0.ravel(), R0], axis=0)

        # SRP exploration
        X_srp = srp_adaptive(
            X0, N,
            Imax=Imax, m=m,
            step_center=step_center,
            step_radius=step_radius,
            beta=beta, backtrack=backtrack,
            w_overlap=w_overlap, w_wall=w_wall, alpha=alpha
        )

        # Local refinement
        X_fin, L_fin = local_optimize(
            X_srp, N,
            w_overlap=w_overlap, w_wall=w_wall, alpha=alpha,
            gtol=gtol, ftol=ftol, maxiter=maxiter, maxcor=maxcor
        )

        centers = X_fin[:2*N].reshape(N, 2)
        radii   = X_fin[2*N:2*N+N]

        # Project to hard-feasible radii with max sum
        r_proj, info = hard_project_max_sum_radii(centers, radii, safety=1e-9)

        # Replace radii for saving / plotting
        radii = r_proj

        # Optional: log diagnostics (guarded)
        viol = info.get("violations") or {}
        print(
            f"[hard-project] success={info.get('success', False)} sum_r={info.get('sum_r', float(np.sum(radii))):.6f} "
            f"min_wall={viol.get('min_wall_clear', float('nan')):.3e} "
            f"min_pair={viol.get('min_pair_clear', float('nan')):.3e}"
        )

        # Metrics
        sum_r   = float(np.sum(radii))
        mwc     = float(min_wall_clearance(centers, radii))
        mpc     = float(min_pair_clearance(centers, radii))

        with open(metrics_fn, "a") as mf:
            mf.write(f"{s},{sum_r:.8f},{mwc:.8f},{mpc:.8f},{L_fin:.8e}\n")

        data[s, 0, :] = centers[:,0].astype(np.float32)
        data[s, 1, :] = centers[:,1].astype(np.float32)
        data[s, 2, :] = radii.astype(np.float32)

        bar.set_postfix(sum_r=f"{sum_r:.3f}", clr=f"{min(mwc, mpc):.4f}")

    # Sort the samples by sum of radii
    sum_radii = data[:, 2, :].sum(axis=1)
    sorted_indices = np.argsort(-sum_radii)  # descending
    data = data[sorted_indices]

    # Save dataset
    torch.save(torch.from_numpy(data), dataset_fn)
    print(f"\nSaved dataset:  {dataset_fn}")
    print(f"Saved metrics:  {metrics_fn}")

    # Save plots for the first plot_k samples
    if plot_k > 0:
        plot_first_k_samples(data, plot_k, plot_dir, filename_prefix="circle_packing")
        print(f"Saved plots:   {plot_dir}")

    return dataset_fn, metrics_fn

# =============================================================================
# Final-push pass over an existing dataset tensor (M,3,N)
# =============================================================================
def main_final_push(input_path=None):
    """
    Read an existing tensor of shape (M, 3, N) with rows [x, y, r],
    run SRP + local L-BFGS-B from those initial conditions per sample,
    hard-project radii to a max-sum feasible solution, and save the
    pushed tensor + metrics into <output_dir>/final_push/.
    """
    sec = "circle_packing_SRP"

    # Core config (re-use the same knobs as generator)
    Imax       = _get_cfg(sec, "srp_Imax",        400)
    m          = _get_cfg(sec, "srp_m",           30)
    beta       = _get_cfg(sec, "srp_beta",        0.985)
    backtrack  = _get_cfg(sec, "srp_backtrack",   3)
    step_center= _get_cfg(sec, "srp_step_center", 0.05)
    step_radius= _get_cfg(sec, "srp_step_radius", 0.01)

    w_overlap  = _get_cfg(sec, "w_overlap",       1.0)
    w_wall     = _get_cfg(sec, "w_wall",          1.0)
    alpha      = _get_cfg(sec, "alpha_sum_r",     1.0)

    gtol       = _get_cfg(sec, "lbfgs_gtol",      1e-8)
    ftol       = _get_cfg(sec, "lbfgs_ftol",      1e-12)
    maxiter    = _get_cfg(sec, "lbfgs_maxiter",   1000)
    maxcor     = _get_cfg(sec, "lbfgs_maxcor",    20)

    out_dir    = _get_cfg(sec, "output_dir",      "./outputs_circle_packing")
    plot_k     = _get_cfg(sec, "plot_k",          0)

    # Input path (can come from cfg or argument)
    if input_path is None:
        input_path = _get_cfg(sec, "final_push_input", None)
        if input_path is None:
            raise ValueError(
                "main_final_push: No input_path provided and "
                "circle_packing_SRP.final_push_input not set."
            )

    # Load input tensor
    tin = torch.load(input_path)
    if isinstance(tin, np.ndarray):
        data_in = tin
    else:
        data_in = tin.detach().cpu().numpy()
    if data_in.ndim != 3 or data_in.shape[1] != 3:
        raise ValueError(f"Expected tensor of shape (M, 3, N); got {data_in.shape}.")

    M, _, N = data_in.shape
    print(f"Final-push on {M} samples, N={N} each, from: {input_path}")

    # I/O setup
    final_dir  = os.path.join(out_dir, "final_push")
    plot_dir   = os.path.join(final_dir, "plots")
    os.makedirs(final_dir, exist_ok=True)

    stamp         = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    pushed_fn     = os.path.join(final_dir, f"circle_final_push_{M}x{N}_{stamp}.pt")
    metrics_fn    = os.path.join(final_dir, f"circle_final_push_metrics_{M}x{N}_{stamp}.csv")

    with open(metrics_fn, "w") as mf:
        mf.write("sample,sum_r,min_wall_clear,min_pair_clear,loss_after\n")

    data_out = np.zeros((M, 3, N), dtype=np.float32)

    bar = tqdm(range(M), desc="Final-push SRP")
    for s in bar:
        centers0 = data_in[s, :2, :].T.astype(np.float64)  # (N,2)
        radii0   = data_in[s, 2, :].astype(np.float64)     # (N,)

        X0 = np.concatenate([centers0.ravel(), radii0], axis=0)

        X_srp = srp_adaptive(
            X0, N,
            Imax=Imax, m=m,
            step_center=step_center,
            step_radius=step_radius,
            beta=beta, backtrack=backtrack,
            w_overlap=w_overlap, w_wall=w_wall, alpha=alpha
        )

        X_fin, L_fin = local_optimize(
            X_srp, N,
            w_overlap=w_overlap, w_wall=w_wall, alpha=alpha,
            gtol=gtol, ftol=ftol, maxiter=maxiter, maxcor=maxcor
        )

        centers = X_fin[:2*N].reshape(N, 2)
        radii   = X_fin[2*N:2*N+N]

        r_proj, info = hard_project_max_sum_radii(centers, radii, safety=1e-9)
        radii = r_proj

        sum_r   = float(np.sum(radii))
        mwc     = float(min_wall_clearance(centers, radii))
        mpc     = float(min_pair_clearance(centers, radii))

        with open(metrics_fn, "a") as mf:
            mf.write(f"{s},{sum_r:.8f},{mwc:.8f},{mpc:.8f},{L_fin:.8e}\n")

        data_out[s, 0, :] = centers[:, 0].astype(np.float32)
        data_out[s, 1, :] = centers[:, 1].astype(np.float32)
        data_out[s, 2, :] = radii.astype(np.float32)

        bar.set_postfix(sum_r=f"{sum_r:.3f}", clr=f"{min(mwc, mpc):.4f}")

    sum_radii = data_out[:, 2, :].sum(axis=1)
    sorted_idx = np.argsort(-sum_radii)
    data_out = data_out[sorted_idx]

    torch.save(torch.from_numpy(data_out), pushed_fn)
    print(f"\nSaved pushed tensor: {pushed_fn}")
    print(f"Saved metrics:       {metrics_fn}")

    if plot_k > 0:
        plot_first_k_samples(data_out, plot_k, plot_dir, filename_prefix="final_push")
        print(f"Saved plots:         {plot_dir}")

    return pushed_fn, metrics_fn

# =============================================================================
# Final-push modified (two initializations, pick best)
# =============================================================================
def main_final_push_modified(input_path=None, output_path=None):
    """
    Read an existing tensor of shape (M, 3, N) with rows [x, y, r] (centers+radii from flow-model),
    and for each sample:
      - keep the centers, option A: keep the original radii
      - option B: reset to small radii
      - run SRP+local+projection for both initializations
      - pick the one with the larger sum of radii
    Returns (pushed_tensor_path, metrics_path).
    """
    sec = "circle_packing_SRP"

    # Core SRP / local config
    Imax        = _get_cfg(sec, "srp_Imax",        400)
    m           = _get_cfg(sec, "srp_m",           30)
    beta        = _get_cfg(sec, "srp_beta",        0.985)
    backtrack   = _get_cfg(sec, "srp_backtrack",   3)
    step_center = _get_cfg(sec, "srp_step_center", 0.05)
    step_radius = _get_cfg(sec, "srp_step_radius", 0.01)

    w_overlap   = _get_cfg(sec, "w_overlap",       1.0)
    w_wall      = _get_cfg(sec, "w_wall",          1.0)
    alpha       = _get_cfg(sec, "alpha_sum_r",     1.0)

    gtol        = _get_cfg(sec, "lbfgs_gtol",      1e-8)
    ftol        = _get_cfg(sec, "lbfgs_ftol",      1e-12)
    maxiter     = _get_cfg(sec, "lbfgs_maxiter",   1000)
    maxcor      = _get_cfg(sec, "lbfgs_maxcor",    20)

    # Output dir (pipeline-friendly):
    # prefer explicit output_path arg, else prefer push_output_dir, else final_push_output, else output_dir.
    out_dir = output_path
    if out_dir is None:
        out_dir = _get_cfg(sec, "push_output_dir", None)
    if out_dir is None:
        out_dir = _get_cfg(sec, "final_push_output", None)
    if out_dir is None:
        out_dir = _get_cfg(sec, "output_dir", "./outputs_circle_packing")

    plot_k      = _get_cfg(sec, "plot_k",          0)

    # Input path
    if input_path is None:
        input_path = _get_cfg(sec, "final_push_input", None)
        if input_path is None:
            raise ValueError("main_final_push_modified: No input_path provided.")

    tin = torch.load(input_path)
    if isinstance(tin, np.ndarray):
        data_in = tin
    else:
        data_in = tin.detach().cpu().numpy()
    if data_in.ndim != 3 or data_in.shape[1] != 3:
        raise ValueError(f"Expected tensor of shape (M,3,N); got {data_in.shape}.")

    M, _, N = data_in.shape
    print(f"Final-push modified on {M} samples, N={N}, from: {input_path}")

    final_dir  = out_dir
    plot_dir   = os.path.join(final_dir, "plots")
    os.makedirs(final_dir, exist_ok=True)

    stamp       = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    pushed_fn   = os.path.join(final_dir, f"circle_final_push_mod_{M}x{N}_{stamp}.pt")
    metrics_fn  = os.path.join(final_dir, f"circle_final_push_mod_metrics_{M}x{N}_{stamp}.csv")

    with open(metrics_fn, "w") as mf:
        mf.write("sample,sum_r_old,sum_r_flowr,sum_r_smallr,sum_r_chosen,which,clr_old_wall,clr_old_pair,loss_after\n")

    data_out = np.zeros((M, 3, N), dtype=np.float32)

    bar = tqdm(range(M), desc="Final-push SRP (modified)")
    for s in bar:
        centers0 = data_in[s, :2, :].T.astype(np.float64)  # (N,2)
        radii_flow = data_in[s, 2, :].astype(np.float64)   # original radii from flow model

        clr_old_pair = float(min_pair_clearance(centers0, radii_flow))
        clr_old_wall = float(min_wall_clearance(centers0, radii_flow))
        sum_r_old    = float(np.sum(radii_flow))

        # --- Option A: keep original radii
        X0a = np.concatenate([centers0.ravel(), radii_flow], axis=0)
        X_srp_a = srp_adaptive(
            X0a, N,
            Imax=Imax, m=m,
            step_center=step_center,
            step_radius=step_radius,
            beta=beta, backtrack=backtrack,
            w_overlap=w_overlap, w_wall=w_wall, alpha=alpha
        )
        X_fin_a, L_fin_a = local_optimize(
            X_srp_a, N,
            w_overlap=w_overlap, w_wall=w_wall, alpha=alpha,
            gtol=gtol, ftol=ftol, maxiter=maxiter, maxcor=maxcor
        )
        centers_a = X_fin_a[:2*N].reshape((N,2))
        radii_a   = X_fin_a[2*N:2*N+N]
        r_proj_a, _ = hard_project_max_sum_radii(centers_a, radii_a, safety=1e-9)
        sum_r_a = float(np.sum(r_proj_a))

        # --- Option B: reset to small radii
        r0_small  = _get_cfg(sec, "init_r0_small",       0.01)
        rj_small  = _get_cfg(sec, "init_rjitter_small",  0.005)
        radii0_b  = init_radii(N, r0=r0_small, jitter=rj_small)
        X0b = np.concatenate([centers0.ravel(), radii0_b], axis=0)

        # optionally use smaller SRP step sizes
        step_center_b = step_center * 0.2
        step_radius_b = step_radius * 0.2
        X_srp_b = srp_adaptive(
            X0b, N,
            Imax=Imax, m=m,
            step_center=step_center_b,
            step_radius=step_radius_b,
            beta=beta, backtrack=backtrack,
            w_overlap=w_overlap, w_wall=w_wall, alpha=alpha
        )
        X_fin_b, L_fin_b = local_optimize(
            X_srp_b, N,
            w_overlap=w_overlap, w_wall=w_wall, alpha=alpha,
            gtol=gtol, ftol=ftol, maxiter=maxiter, maxcor=maxcor
        )
        centers_b = X_fin_b[:2*N].reshape((N,2))
        radii_b   = X_fin_b[2*N:2*N+N]
        r_proj_b, _ = hard_project_max_sum_radii(centers_b, radii_b, safety=1e-9)
        sum_r_b = float(np.sum(r_proj_b))

        # Choose better
        if sum_r_b > sum_r_a:
            chosen = "smallr"
            centers_chosen = centers_b
            radii_chosen   = r_proj_b
            sum_r_chosen   = sum_r_b
            loss_after     = float(L_fin_b)
        else:
            chosen = "flowr"
            centers_chosen = centers_a
            radii_chosen   = r_proj_a
            sum_r_chosen   = sum_r_a
            loss_after     = float(L_fin_a)

        with open(metrics_fn, "a") as mf:
            mf.write(
                f"{s},{sum_r_old:.8f},{sum_r_a:.8f},{sum_r_b:.8f},{sum_r_chosen:.8f},"
                f"{chosen},{clr_old_wall:.6e},{clr_old_pair:.6e},{loss_after:.8e}\n"
            )

        data_out[s, 0, :] = centers_chosen[:,0].astype(np.float32)
        data_out[s, 1, :] = centers_chosen[:,1].astype(np.float32)
        data_out[s, 2, :] = radii_chosen.astype(np.float32)

        bar.set_postfix(sum_r=f"{sum_r_chosen:.3f}", which=chosen)

    # Sort by descending sum_r
    sum_radii = data_out[:, 2, :].sum(axis=1)
    sorted_idx = np.argsort(-sum_radii)
    data_out = data_out[sorted_idx]

    torch.save(torch.from_numpy(data_out), pushed_fn)
    print(f"\nSaved pushed tensor: {pushed_fn}")
    print(f"Saved metrics:       {metrics_fn}")

    if plot_k > 0:
        plot_first_k_samples(data_out, plot_k, plot_dir, filename_prefix="final_push_mod")
        print(f"Saved plots:         {plot_dir}")

    return pushed_fn, metrics_fn

# =============================================================================
# Pipeline-friendly entrypoint
# =============================================================================
def main(state=None):
    """
    Pipeline-friendly entrypoint.

    Reads circle_packing_SRP.mode:
      - training_set_gen : generate fresh SRP dataset -> state.samples_path
      - push_only        : push an existing dataset  -> state.pushed_samples_path

    For push_only, reads:
      - circle_packing_SRP.push_input (preferred)
      - circle_packing_SRP.final_push_input (legacy fallback)

    Writes paths into `state` if provided:
      state.set_samples_path(...)
      state.set_pushed_samples_path(...)
    """
    sec = "circle_packing_SRP"
    mode = str(_get_cfg(sec, "mode", "training_set_gen")).strip().lower()

    if mode in ("training_set_gen", "training_set_generation", "generate", "gen"):
        dataset_fn, _ = generate_circle_packing_dataset()
        if state is not None:
            state.set_samples_path(dataset_fn)
        return dataset_fn

    if mode in ("push_only", "push", "final_push", "final-push"):
        in_path = _get_cfg(sec, "push_input", None)
        if in_path is None:
            in_path = _get_cfg(sec, "final_push_input", None)
        if in_path is None:
            raise ValueError("push_only requires circle_packing_SRP.push_input (or legacy final_push_input).")

        # allow cfg-driven output dir for pushed results
        out_dir = _get_cfg(sec, "push_output_dir", None)
        if out_dir is None:
            out_dir = _get_cfg(sec, "final_push_output", None)

        pushed_fn, _ = main_final_push_modified(input_path=in_path, output_path=out_dir)
        if state is not None:
            state.set_pushed_samples_path(pushed_fn)
        return pushed_fn

    raise ValueError(
        f"Unknown circle_packing_SRP.mode='{mode}'. Expected: training_set_gen / push_only."
    )

# =============================================================================
# Entrypoint (config-driven)
# =============================================================================
if __name__ == "__main__":
    main(state=None)
