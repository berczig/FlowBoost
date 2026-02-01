# ---- OpenMP guard: must be FIRST (before numpy/torch/matplotlib) ----
import os
import platform
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

# =============================================================================
# Utilities / small constants
# =============================================================================
EPS = 1e-12

def _set_cfg(section, key, value):
    """
    Mutate flow_boost.cfg in-place (spheres-style).
    Stores value as a string (ConfigParser semantics).
    """
    try:
        from flow_boost import cfg
    except Exception as e:
        raise RuntimeError("flow_boost.cfg not available; cannot _set_cfg") from e

    if not cfg.has_section(section):
        cfg.add_section(section)

    cfg.set(section, key, str(value))


def _get_cfg(section, key, fallback):
    """
    Config getter compatible with your flow_boost.cfg pattern.
    """
    try:
        from flow_boost import cfg
        if isinstance(fallback, int):
            return cfg.getint(section, key, fallback=fallback)
        if isinstance(fallback, float):
            return cfg.getfloat(section, key, fallback=fallback)
        if isinstance(fallback, bool):
            return cfg.getboolean(section, key, fallback=fallback)
        return cfg.get(section, key, fallback=fallback)
    except Exception:
        return fallback

def sample_uniform_points(N):
    return np.random.uniform(0.0, 1.0, size=(N, 2)).astype(np.float64)

@njit(cache=True, fastmath=True)
def _l2_norm(v):
    s = 0.0
    for k in range(v.size):
        s += v[k] * v[k]
    return math.sqrt(s) + 1e-18

# =============================================================================
# Smooth star discrepancy surrogate (2D)
# =============================================================================

@njit(cache=True, fastmath=True)
def _sigmoid(z):
    # stable-ish sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

@njit(cache=True, fastmath=True)
def _smooth_abs(u, eps_abs=1e-12):
    return math.sqrt(u * u + eps_abs)

def _sigmoid_np(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

def _smooth_abs_np(u, eps=1e-12):
    return np.sqrt(u * u + eps)

def critical_grid_from_points(pts, pad_eps=0.0, include_one=True):
    """
    pts: (N,2) in [0,1]. Return Ax, Ay as unique coordinate grids (+1).
    """
    x = np.clip(pts[:, 0], 0.0, 1.0)
    y = np.clip(pts[:, 1], 0.0, 1.0)
    xs = np.unique(x)
    ys = np.unique(y)
    if include_one:
        xs = np.unique(np.concatenate([xs, np.array([1.0])]))
        ys = np.unique(np.concatenate([ys, np.array([1.0])]))
    if pad_eps > 0.0:
        xs = np.clip(xs + pad_eps, 0.0, 1.0)
        ys = np.clip(ys + pad_eps, 0.0, 1.0)
    return xs.astype(np.float64), ys.astype(np.float64)

def uniform_grid(Gx, Gy):
    # avoid 0 (trivial anchored boxes)
    Ax = np.linspace(1.0 / float(Gx), 1.0, int(Gx)).astype(np.float64)
    Ay = np.linspace(1.0 / float(Gy), 1.0, int(Gy)).astype(np.float64)
    return Ax, Ay

def star_discrepancy_loss_and_grad(
    X, N, Ax, Ay,
    beta_softmax,
    tau_sigmoid,
    w_wall=1e-8,
    eps_abs=1e-12,
    topk_boxes=0,
):
    """
    Vectorized smooth 2D star discrepancy surrogate with optional Top-K boxes.

    L = (1/beta) log sum_{(a,b) in Ax×Ay} exp(beta * | C(a,b) - a*b | )
    where C(a,b) ≈ (1/N) sum_i sigmoid((a-x_i)/tau)*sigmoid((b-y_i)/tau).
    """
    pts = X[:2 * N].reshape(N, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    g = np.zeros_like(X, dtype=np.float64)
    L = 0.0

    # --- tiny wall penalty (bounds exist too, but SRP can step out) ---
    v = -x
    mask = v > 0
    if np.any(mask):
        g[0::2][mask] += -2.0 * w_wall * v[mask]
        L += w_wall * float(np.sum(v[mask] ** 2))

    v = x - 1.0
    mask = v > 0
    if np.any(mask):
        g[0::2][mask] += 2.0 * w_wall * v[mask]
        L += w_wall * float(np.sum(v[mask] ** 2))

    v = -y
    mask = v > 0
    if np.any(mask):
        g[1::2][mask] += -2.0 * w_wall * v[mask]
        L += w_wall * float(np.sum(v[mask] ** 2))

    v = y - 1.0
    mask = v > 0
    if np.any(mask):
        g[1::2][mask] += 2.0 * w_wall * v[mask]
        L += w_wall * float(np.sum(v[mask] ** 2))

    U = Ax.size
    V = Ay.size

    # Sx[u,i] = sigmoid((a_u - x_i)/tau), Sy[v,i] similarly
    Sx = _sigmoid_np((Ax[:, None] - x[None, :]) / tau_sigmoid)  # (U,N)
    Sy = _sigmoid_np((Ay[:, None] - y[None, :]) / tau_sigmoid)  # (V,N)

    # C[u,v] = (1/N) sum_i Sx[u,i]*Sy[v,i] = (Sx @ Sy^T) / N
    C = (Sx @ Sy.T) / float(N)  # (U,V)

    Agrid, Bgrid = np.meshgrid(Ax, Ay, indexing="ij")  # (U,V)
    Delta = C - (Agrid * Bgrid)
    Dabs = _smooth_abs_np(Delta, eps_abs)

    # derivatives of gates wrt x and y
    dSx_dx = -(Sx * (1.0 - Sx)) / tau_sigmoid  # (U,N)
    dSy_dy = -(Sy * (1.0 - Sy)) / tau_sigmoid  # (V,N)

    # --- Top-K path (optional) ---
    if topk_boxes and topk_boxes < Dabs.size:
        K = int(topk_boxes)
        flat_idx = np.argpartition(Dabs.ravel(), -K)[-K:]
        u_idx, v_idx = np.unravel_index(flat_idx, Dabs.shape)

        Dsel = Dabs[u_idx, v_idx]
        Delta_sel = Delta[u_idx, v_idx]

        Xlog = beta_softmax * Dsel
        m = float(np.max(Xlog))
        ex = np.exp(Xlog - m)
        Z = float(ex.sum())
        L += (m + math.log(Z)) / beta_softmax

        W = ex / Z  # (K,)
        dabs_ddelta = Delta_sel / (Dsel + 1e-18)  # (K,)
        coeff_k = (W * dabs_ddelta) / float(N)    # (K,)

        grad_x = np.zeros(N, dtype=np.float64)
        grad_y = np.zeros(N, dtype=np.float64)
        for kk, (u, v) in enumerate(zip(u_idx, v_idx)):
            c = coeff_k[kk]
            grad_x += c * dSx_dx[u, :] * Sy[v, :]
            grad_y += c * dSy_dy[v, :] * Sx[u, :]

        g[0::2] += grad_x
        g[1::2] += grad_y
        return float(L), g

    # --- Full path ---
    Xlog = beta_softmax * Dabs
    m = float(np.max(Xlog))
    ex = np.exp(Xlog - m)
    Z = float(ex.sum())
    L += (m + math.log(Z)) / beta_softmax

    W = ex / Z  # (U,V)
    dabs_ddelta = Delta / (Dabs + 1e-18)
    Coeff = (W * dabs_ddelta) / float(N)  # (U,V)

    # x-gradient: sum_u dSx_dx[u,i] * sum_v Coeff[u,v] * Sy[v,i]
    T_x = Coeff @ Sy  # (U,N)
    grad_x = np.sum(dSx_dx * T_x, axis=0)

    # y-gradient: sum_v dSy_dy[v,i] * sum_u Coeff[u,v] * Sx[u,i]
    T_y = (Coeff.T @ Sx)  # (V,N)
    grad_y = np.sum(dSy_dy * T_y, axis=0)

    g[0::2] += grad_x
    g[1::2] += grad_y
    return float(L), g

# =============================================================================
# Exact 2D star discrepancy (critical grid)
# =============================================================================
def exact_star_discrepancy_2d(pts):
    """
    Exact 2D L_infinity star discrepancy on the critical grid.
    Checks open [0,a)×[0,b) and closed [0,a]×[0,b].
    """
    pts = np.asarray(pts, dtype=np.float64)
    assert pts.ndim == 2 and pts.shape[1] == 2
    N = pts.shape[0]
    if N == 0:
        return 0.0, {"open_max": 0.0, "closed_max": 0.0}

    x = np.clip(pts[:, 0], 0.0, 1.0)
    y = np.clip(pts[:, 1], 0.0, 1.0)

    Ax = np.unique(np.concatenate([x, [1.0]]))
    Ay = np.unique(np.concatenate([y, [1.0]]))
    U, V = Ax.size, Ay.size

    # OPEN (<,<)
    iu = np.searchsorted(Ax, x, side="left")
    iv = np.searchsorted(Ay, y, side="left")
    M_open = np.zeros((U, V), dtype=np.int64)
    for i, j in zip(iu, iv):
        M_open[i, j] += 1
    C_open = M_open.cumsum(axis=0).cumsum(axis=1)
    frac_open = C_open / float(N)

    Agrid, Bgrid = np.meshgrid(Ax, Ay, indexing="ij")
    D_minus = Agrid * Bgrid - frac_open
    D_minus_max = float(D_minus.max())
    u_minus, v_minus = np.unravel_index(np.argmax(D_minus), D_minus.shape)

    # CLOSED (<=,<=)
    iu2 = np.searchsorted(Ax, x, side="right") - 1
    iv2 = np.searchsorted(Ay, y, side="right") - 1
    M_closed = np.zeros((U, V), dtype=np.int64)
    for i, j in zip(iu2, iv2):
        M_closed[i, j] += 1
    C_closed = M_closed.cumsum(axis=0).cumsum(axis=1)
    frac_closed = C_closed / float(N)

    D_plus = frac_closed - Agrid * Bgrid
    D_plus_max = float(D_plus.max())
    u_plus, v_plus = np.unravel_index(np.argmax(D_plus), D_plus.shape)

    D = max(D_minus_max, D_plus_max)
    details = {
        "open_max": D_minus_max,
        "open_box": (float(Ax[u_minus]), float(Ay[v_minus])),
        "open_count": int(C_open[u_minus, v_minus]),
        "closed_max": D_plus_max,
        "closed_box": (float(Ax[u_plus]), float(Ay[v_plus])),
        "closed_count": int(C_closed[u_plus, v_plus]),
        "grid_sizes": (int(U), int(V)),
    }
    return D, details

# =============================================================================
# SRP (adaptive) for star discrepancy (mirrors Heilbronn structure)
# =============================================================================
def srp_adaptive_points_star(
    X, N,
    Imax, m, step_center, beta_sched_decay, backtrack,
    beta_softmax_start, beta_softmax_final,
    tau_sigmoid,
    w_wall, eps_abs,
    grid_mode="critical",         # "critical" or "uniform"
    grid_x=64, grid_y=64,
    grid_refresh_frac=0.10,       # refresh critical grid every ~Imax*frac iters
    critical_pad_eps=0.0,
    topk_boxes=0,
):
    Xc = X.copy()
    eta = 1.0

    grid_mode = str(grid_mode).strip().lower()

    if grid_mode == "uniform":
        Ax, Ay = uniform_grid(grid_x, grid_y)
        refresh_every = None
    else:
        pts0 = Xc[:2*N].reshape(N, 2)
        Ax, Ay = critical_grid_from_points(pts0, pad_eps=critical_pad_eps)
        refresh_every = max(1, int(round(Imax * float(grid_refresh_frac))))

    for it in range(Imax):
        tfrac = (it + 0.0) / max(Imax - 1, 1)
        beta_t = beta_softmax_start * math.pow(beta_softmax_final / beta_softmax_start, tfrac)

        if refresh_every is not None and (it % refresh_every) == 0:
            pts = Xc[:2*N].reshape(N, 2)
            Ax, Ay = critical_grid_from_points(pts, pad_eps=critical_pad_eps)

        # SRP noise step
        noise = (np.random.rand(Xc.size) * 2.0 - 1.0) * (eta * step_center)
        Xtrial = Xc + noise

        # m gradient steps with backtracking
        for _ in range(m):
            L0, g = star_discrepancy_loss_and_grad(
                Xtrial, N, Ax, Ay, beta_t, tau_sigmoid,
                w_wall=w_wall, eps_abs=eps_abs, topk_boxes=topk_boxes
            )
            step = np.full_like(g, eta * step_center)
            gn = float(np.linalg.norm(g) + 1e-18)
            Xprop = Xtrial - step * (g / gn)

            Lprop, _ = star_discrepancy_loss_and_grad(
                Xprop, N, Ax, Ay, beta_t, tau_sigmoid,
                w_wall=w_wall, eps_abs=eps_abs, topk_boxes=topk_boxes
            )

            bt = 0
            while Lprop > L0 and bt < backtrack:
                step *= 0.5
                Xprop = Xtrial - step * (g / gn)
                Lprop, _ = star_discrepancy_loss_and_grad(
                    Xprop, N, Ax, Ay, beta_t, tau_sigmoid,
                    w_wall=w_wall, eps_abs=eps_abs, topk_boxes=topk_boxes
                )
                bt += 1

            Xtrial = Xprop

        # clip + decay
        Xtrial[0::2] = np.clip(Xtrial[0::2], 0.0, 1.0)
        Xtrial[1::2] = np.clip(Xtrial[1::2], 0.0, 1.0)
        Xc = Xtrial
        eta *= beta_sched_decay

    return Xc

# =============================================================================
# Local optimization (L-BFGS-B) on [0,1]^{2N}
# =============================================================================
def local_optimize_points_star(
    X0, N, Ax, Ay,
    beta_softmax_final, tau_sigmoid,
    w_wall, eps_abs,
    topk_boxes=0,
    gtol=1e-8, ftol=1e-12, maxiter=500, maxcor=20
):
    bounds = [(0.0, 1.0)] * (2 * N)

    def fun(x):
        L, _ = star_discrepancy_loss_and_grad(
            x, N, Ax, Ay, beta_softmax_final, tau_sigmoid,
            w_wall=w_wall, eps_abs=eps_abs, topk_boxes=topk_boxes
        )
        return L

    def jac(x):
        _, g = star_discrepancy_loss_and_grad(
            x, N, Ax, Ay, beta_softmax_final, tau_sigmoid,
            w_wall=w_wall, eps_abs=eps_abs, topk_boxes=topk_boxes
        )
        return g

    res = minimize(fun=fun, x0=X0, method="L-BFGS-B", jac=jac, bounds=bounds,
                   options={"gtol": gtol, "ftol": ftol, "maxiter": maxiter, "maxcor": maxcor})
    return res.x, float(res.fun), res

# =============================================================================
# Metrics: surrogate value (at evaluation params)
# =============================================================================
@njit(cache=True, fastmath=True)
def star_disc_surrogate_value(pts, Ax, Ay, beta_softmax, tau_sigmoid, eps_abs=1e-12):
    N = pts.shape[0]
    Z = 0.0
    for ui in range(Ax.size):
        a = Ax[ui]
        for vi in range(Ay.size):
            b = Ay[vi]
            c = 0.0
            for i in range(N):
                sx = _sigmoid((a - pts[i, 0]) / tau_sigmoid)
                sy = _sigmoid((b - pts[i, 1]) / tau_sigmoid)
                c += sx * sy
            c /= N
            delta = c - a * b
            dabs = _smooth_abs(delta, eps_abs)
            Z += math.exp(beta_softmax * dabs)
    return (1.0 / beta_softmax) * math.log(Z)

# =============================================================================
# Plotting (optional)
# =============================================================================
def plot_point_sets_star(data_tensor, k, out_dir, filename_prefix="stardisc"):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    M = data_tensor.shape[0]

    disc = np.zeros(M, dtype=np.float64)
    for s in range(M):
        pts = data_tensor[s].T.astype(np.float64)
        D, _ = exact_star_discrepancy_2d(pts)
        disc[s] = D

    order = np.argsort(disc)  # smaller is better
    for rank in range(min(k, M)):
        s = int(order[rank])
        pts = data_tensor[s].T.astype(np.float64)
        D, info = exact_star_discrepancy_2d(pts)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="black")
        ax.scatter(pts[:, 0], pts[:, 1], s=12)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"Rank {rank+1} (sample {s}), N={pts.shape[0]}, star D={D:.8f}")
        out_path = os.path.join(
            out_dir,
            f"{filename_prefix}_n={pts.shape[0]}_rank={rank+1}_D={D:.8f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

# =============================================================================
# Main generator (training_set_gen) — metrics written incrementally per sample
# =============================================================================
def generate_star_discrepancy_dataset():
    sec = "star_SRP"

    # ---- dataset sizes ----
    N = _get_cfg(sec, "num_points", 50)
    M = _get_cfg(sec, "num_samples", 2000)

    # ---- SRP hyperparams ----
    Imax        = _get_cfg(sec, "srp_Imax",        400)
    m           = _get_cfg(sec, "srp_m",           30)
    beta_sched  = _get_cfg(sec, "srp_beta",        0.985)
    backtrack   = _get_cfg(sec, "srp_backtrack",   3)
    step_center = _get_cfg(sec, "srp_step_center", 0.05)

    # ---- loss params ----
    beta0       = _get_cfg(sec, "beta_softmax_start", 10.0)
    betaF       = _get_cfg(sec, "beta_softmax_final", 200.0)
    tau_sigmoid = _get_cfg(sec, "tau_sigmoid",        0.01)
    w_wall      = _get_cfg(sec, "w_wall",             1e-8)
    eps_abs     = _get_cfg(sec, "abs_eps",            1e-12)
    topk_boxes  = _get_cfg(sec, "topk_boxes",         0)

    # ---- grid policy (SRP) ----
    grid_mode         = str(_get_cfg(sec, "grid_mode", "critical")).strip().lower()
    grid_x            = _get_cfg(sec, "grid_x", 64)
    grid_y            = _get_cfg(sec, "grid_y", 64)
    grid_refresh_frac = _get_cfg(sec, "grid_refresh_frac", 0.10)
    critical_pad_eps  = _get_cfg(sec, "critical_pad_eps", 0.0)

    # ---- L-BFGS params ----
    gtol        = _get_cfg(sec, "lbfgs_gtol",      1e-8)
    ftol        = _get_cfg(sec, "lbfgs_ftol",      1e-12)
    maxiter     = _get_cfg(sec, "lbfgs_maxiter",   500)
    maxcor      = _get_cfg(sec, "lbfgs_maxcor",    20)
    lbfgs_grid_mode = str(_get_cfg(sec, "lbfgs_grid_mode", "uniform")).strip().lower()

    # ---- evaluation params ----
    eval_grid_mode = str(_get_cfg(sec, "eval_grid_mode", "uniform")).strip().lower()
    eval_grid_x    = _get_cfg(sec, "eval_grid_x", grid_x)
    eval_grid_y    = _get_cfg(sec, "eval_grid_y", grid_y)
    beta_eval      = float(_get_cfg(sec, "beta_eval", betaF))
    tau_eval       = float(_get_cfg(sec, "tau_eval", tau_sigmoid))

    # ---- I/O + plotting ----
    out_dir  = _get_cfg(sec, "output_dir", "./outputs_star")
    plot_k   = _get_cfg(sec, "plot_k", 0)  # (rename from your old plot_k_dis)
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dataset_fn = os.path.join(out_dir, f"star_srp_{M}x{N}_{stamp}.pt")
    metrics_fn = os.path.join(out_dir, f"star_metrics_{M}x{N}_{stamp}.csv")

    print(f"Generating {M} star-discrepancy point sets (N={N})")
    print(f"SRP grid_mode={grid_mode}, topk_boxes={topk_boxes}, tau={tau_sigmoid}, beta: {beta0}->{betaF}")
    print(f"Writing metrics incrementally to: {metrics_fn}")
    print(f"Saving dataset to: {dataset_fn}")

    # precompute eval grid if uniform
    if eval_grid_mode == "uniform":
        Ax_eval, Ay_eval = uniform_grid(eval_grid_x, eval_grid_y)
    else:
        Ax_eval, Ay_eval = None, None

    data = np.zeros((M, 2, N), dtype=np.float32)
    exact_scores = np.zeros(M, dtype=np.float64)  # for final sorting only (dataset sort)

    # open metrics file once; write header; flush every sample
    with open(metrics_fn, "w") as mf:
        mf.write("sample,exact_discr,open_max,closed_max,star_surr,beta_eval,tau_eval,loss_after\n")
        mf.flush()

        bar = tqdm(range(M), desc="Generating point sets")
        for s in bar:
            P0 = sample_uniform_points(N)
            X0 = P0.ravel()

            # SRP exploration
            X_srp = srp_adaptive_points_star(
                X0, N,
                Imax=Imax, m=m,
                step_center=step_center,
                beta_sched_decay=beta_sched, backtrack=backtrack,
                beta_softmax_start=beta0, beta_softmax_final=betaF,
                tau_sigmoid=tau_sigmoid,
                w_wall=w_wall, eps_abs=eps_abs,
                grid_mode=grid_mode, grid_x=grid_x, grid_y=grid_y,
                grid_refresh_frac=grid_refresh_frac,
                critical_pad_eps=critical_pad_eps,
                topk_boxes=topk_boxes,
            )

            # L-BFGS refinement grid
            if lbfgs_grid_mode == "critical":
                pts_tmp = X_srp.reshape(N, 2)
                Ax_lbfgs, Ay_lbfgs = critical_grid_from_points(pts_tmp, pad_eps=critical_pad_eps)
            else:
                Ax_lbfgs, Ay_lbfgs = uniform_grid(grid_x, grid_y)

            X_fin, L_fin, _res = local_optimize_points_star(
                X_srp, N, Ax_lbfgs, Ay_lbfgs,
                beta_softmax_final=betaF, tau_sigmoid=tau_sigmoid,
                w_wall=w_wall, eps_abs=eps_abs,
                topk_boxes=topk_boxes,
                gtol=gtol, ftol=ftol, maxiter=maxiter, maxcor=maxcor
            )

            pts = X_fin.reshape(N, 2)

            # exact discrepancy
            D_exact, info = exact_star_discrepancy_2d(pts)
            exact_scores[s] = D_exact

            # surrogate metric
            if eval_grid_mode == "uniform":
                Ax_m, Ay_m = Ax_eval, Ay_eval
            else:
                Ax_m, Ay_m = critical_grid_from_points(pts, pad_eps=critical_pad_eps)
            star_surr = float(star_disc_surrogate_value(pts, Ax_m, Ay_m, beta_eval, tau_eval, eps_abs))

            # store sample
            data[s, 0, :] = pts[:, 0].astype(np.float32)
            data[s, 1, :] = pts[:, 1].astype(np.float32)

            # write metrics row immediately (so you can tail -f the file)
            mf.write(
                f"{s},{D_exact:.12f},{info['open_max']:.12f},{info['closed_max']:.12f},"
                f"{star_surr:.12f},{beta_eval:.6f},{tau_eval:.8f},{L_fin:.8e}\n"
            )
            mf.flush()

            bar.set_postfix(exact=f"{D_exact:.6f}", surr=f"{star_surr:.6f}")

    # sort dataset by exact discrepancy (ascending; smaller is better)
    order = np.argsort(exact_scores)
    data = data[order]

    

    # save dataset
    torch.save(torch.from_numpy(data), dataset_fn)
    print(f"\nSaved dataset:  {dataset_fn}")
    print(f"Saved metrics:  {metrics_fn}")

    if plot_k > 0:
        plot_point_sets_star(data, plot_k, plot_dir, filename_prefix="star_srp_best")
        print(f"Saved plots:   {plot_dir}")

    return dataset_fn, metrics_fn

# =============================================================================
# push_only mode (load .pt, improve each sample, save new .pt + metrics)
# =============================================================================
def push_existing_samples_star():
    sec = "star_SRP"

    # ---- IO ----
    N = _get_cfg(sec, "num_points", 50)
    input_path = _get_cfg(sec, "push_input", "")
    out_dir = _get_cfg(sec, "push_output_dir", "./outputs_star_push")
    plot_k = _get_cfg(sec, "plot_k", 0)
    plot_dir = os.path.join(out_dir, "plots")

    assert isinstance(input_path, str) and len(input_path) > 0 and os.path.exists(input_path), \
        "Set star_SRP.push_input to a valid .pt file"

    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dataset_fn = os.path.join(out_dir, f"star_srp_pushed_{stamp}.pt")
    metrics_fn = os.path.join(out_dir, f"star_metrics_pushed_{stamp}.csv")

    # ---- SRP hyperparams ----
    Imax        = _get_cfg(sec, "srp_Imax",        400)
    m           = _get_cfg(sec, "srp_m",           30)
    beta_sched  = _get_cfg(sec, "srp_beta",        0.985)
    backtrack   = _get_cfg(sec, "srp_backtrack",   3)
    step_center = _get_cfg(sec, "srp_step_center", 0.05)

    # ---- loss params ----
    beta0       = _get_cfg(sec, "beta_softmax_start", 10.0)
    betaF       = _get_cfg(sec, "beta_softmax_final", 200.0)
    tau_sigmoid = _get_cfg(sec, "tau_sigmoid",        0.01)
    w_wall      = _get_cfg(sec, "w_wall",             1e-8)
    eps_abs     = _get_cfg(sec, "abs_eps",            1e-12)
    topk_boxes  = _get_cfg(sec, "topk_boxes",         0)

    # ---- grid policy ----
    grid_mode         = str(_get_cfg(sec, "grid_mode", "critical")).strip().lower()
    grid_x            = _get_cfg(sec, "grid_x", 64)
    grid_y            = _get_cfg(sec, "grid_y", 64)
    grid_refresh_frac = _get_cfg(sec, "grid_refresh_frac", 0.10)
    critical_pad_eps  = _get_cfg(sec, "critical_pad_eps", 0.0)

    # ---- L-BFGS ----
    gtol        = _get_cfg(sec, "lbfgs_gtol",      1e-8)
    ftol        = _get_cfg(sec, "lbfgs_ftol",      1e-12)
    maxiter     = _get_cfg(sec, "lbfgs_maxiter",   500)
    maxcor      = _get_cfg(sec, "lbfgs_maxcor",    20)
    lbfgs_grid_mode = str(_get_cfg(sec, "lbfgs_grid_mode", "uniform")).strip().lower()

    # ---- evaluation params ----
    eval_grid_mode = str(_get_cfg(sec, "eval_grid_mode", "uniform")).strip().lower()
    eval_grid_x    = _get_cfg(sec, "eval_grid_x", grid_x)
    eval_grid_y    = _get_cfg(sec, "eval_grid_y", grid_y)
    beta_eval      = float(_get_cfg(sec, "beta_eval", betaF))
    tau_eval       = float(_get_cfg(sec, "tau_eval", tau_sigmoid))

    if eval_grid_mode == "uniform":
        Ax_eval, Ay_eval = uniform_grid(eval_grid_x, eval_grid_y)
    else:
        Ax_eval, Ay_eval = None, None

    # ---- load ----
    loaded = torch.load(input_path, map_location="cpu")
    arr = loaded.detach().cpu().numpy() if isinstance(loaded, torch.Tensor) else None
    if arr is None or arr.ndim != 3:
        raise ValueError(f"Expected a tensor at star_SRP.push_input, got shape {getattr(loaded, 'shape', None)}")

    # accept either (M,2,N) or (M,N,2)
    if arr.shape[1] == 2:
        M_in, d_in, N_in = arr.shape
    elif arr.shape[-1] == 2:
        arr = np.transpose(arr, (0, 2, 1))
        M_in, d_in, N_in = arr.shape
    else:
        raise ValueError(f"Second or last dimension must be 2; got shape {arr.shape}")

    if N_in != N:
        raise ValueError(f"Config N={N} but input samples have N={N_in}")

    K = int(_get_cfg(sec, "push_count", M_in))
    K = min(K, M_in)

    print(f"Pushing {K}/{M_in} loaded samples (N={N})")
    print(f"Writing metrics incrementally to: {metrics_fn}")
    print(f"Saving pushed dataset to: {dataset_fn}")

    data = np.zeros((K, 2, N), dtype=np.float32)
    exact_scores_after = np.zeros(K, dtype=np.float64)

    with open(metrics_fn, "w") as mf:
        mf.write("sample,exact_before,exact_after,star_surr_after,beta_eval,tau_eval,loss_after\n")
        mf.flush()

        for s in tqdm(range(K), desc="[Pushing Samples]"):
            pts0 = arr[s].T.astype(np.float64)   # (N,2)
            X0   = pts0.ravel()

            D_before, _ = exact_star_discrepancy_2d(pts0)

            X_srp = srp_adaptive_points_star(
                X0, N,
                Imax=Imax, m=m,
                step_center=step_center,
                beta_sched_decay=beta_sched, backtrack=backtrack,
                beta_softmax_start=beta0, beta_softmax_final=betaF,
                tau_sigmoid=tau_sigmoid,
                w_wall=w_wall, eps_abs=eps_abs,
                grid_mode=grid_mode, grid_x=grid_x, grid_y=grid_y,
                grid_refresh_frac=grid_refresh_frac,
                critical_pad_eps=critical_pad_eps,
                topk_boxes=topk_boxes,
            )

            if lbfgs_grid_mode == "critical":
                pts_tmp = X_srp.reshape(N, 2)
                Ax_lbfgs, Ay_lbfgs = critical_grid_from_points(pts_tmp, pad_eps=critical_pad_eps)
            else:
                Ax_lbfgs, Ay_lbfgs = uniform_grid(grid_x, grid_y)

            X_fin, L_fin, _res = local_optimize_points_star(
                X_srp, N, Ax_lbfgs, Ay_lbfgs,
                beta_softmax_final=betaF, tau_sigmoid=tau_sigmoid,
                w_wall=w_wall, eps_abs=eps_abs,
                topk_boxes=topk_boxes,
                gtol=gtol, ftol=ftol, maxiter=maxiter, maxcor=maxcor
            )

            pts = X_fin.reshape(N, 2)
            D_after, _ = exact_star_discrepancy_2d(pts)
            exact_scores_after[s] = D_after

            if eval_grid_mode == "uniform":
                Ax_m, Ay_m = Ax_eval, Ay_eval
            else:
                Ax_m, Ay_m = critical_grid_from_points(pts, pad_eps=critical_pad_eps)
            star_surr = float(star_disc_surrogate_value(pts, Ax_m, Ay_m, beta_eval, tau_eval, eps_abs))

            data[s, 0, :] = pts[:, 0].astype(np.float32)
            data[s, 1, :] = pts[:, 1].astype(np.float32)

            mf.write(
                f"{s},{D_before:.12f},{D_after:.12f},{star_surr:.12f},"
                f"{beta_eval:.6f},{tau_eval:.8f},{L_fin:.8e}\n"
            )
            mf.flush()

    # sort by exact_after
    order = np.argsort(exact_scores_after)
    data = data[order]

    torch.save(torch.from_numpy(data), dataset_fn)
    print(f"\nSaved pushed dataset:  {dataset_fn}")
    print(f"Saved metrics:         {metrics_fn}")

    if plot_k > 0:
        plot_point_sets_star(data, plot_k, plot_dir, filename_prefix="star_srp_pushed_best")
        print(f"Saved plots:          {plot_dir}")

    return dataset_fn, metrics_fn

# =============================================================================
# Entrypoint
# =============================================================================
def main(state=None):
    sec = "star_SRP"
    mode = str(_get_cfg(sec, "mode", "training_set_gen")).strip().lower()
    if mode == "training_set_gen":
        dataset_fn, metrics_fn = generate_star_discrepancy_dataset()   # make it return paths
        if state is not None:
            state.set_samples_path(dataset_fn)
            # optional: state.set_metrics_path(metrics_fn)
        return dataset_fn, metrics_fn
    elif mode in ("push_only", "final_push"):
        dataset_fn, metrics_fn = push_existing_samples_star()          # make it return paths
        if state is not None:
            state.set_pushed_samples_path(dataset_fn)
            # optional: state.set_metrics_path(metrics_fn)
        return dataset_fn, metrics_fn
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'training_set_gen' or 'push_only'.")

if __name__ == "__main__":
    main()
