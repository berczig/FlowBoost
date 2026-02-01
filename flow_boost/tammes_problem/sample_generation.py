# ---- OpenMP guard: must be FIRST (before numpy/torch/matplotlib) ----
import os, platform
if platform.system() == "Darwin" and os.environ.get("SPHEREPACK_DISABLE_KMP_HACK") != "1":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

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

def _get_cfg_list_floats(section, key, fallback_csv):
    txt = _get_cfg(section, key, fallback_csv)
    if isinstance(txt, str):
        parts = [p.strip() for p in txt.split(",") if p.strip() != ""]
        return [float(p) for p in parts]
    if isinstance(txt, (list, tuple)):
        return [float(x) for x in txt]
    return [float(txt)]

# =============================================================================
# Random initialization on the sphere S^{n-1}
# =============================================================================
def sample_uniform_on_sphere(N, dim):
    X = np.random.normal(size=(N, dim)).astype(np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-18
    return (X / norms).astype(np.float64)

# =============================================================================
# Spherical geometry helpers
# =============================================================================
@njit(cache=True, fastmath=True)
def _tangent_projection(g, x):
    dot = 0.0
    for k in range(x.size):
        dot += g[k] * x[k]
    out = g.copy()
    for k in range(x.size):
        out[k] -= dot * x[k]
    return out

@njit(cache=True, fastmath=True)
def _expmap_sphere(x, u):
    un = 0.0
    for k in range(u.size):
        un += u[k]*u[k]
    un = math.sqrt(un)
    if un < 1e-18:
        y = x + u
        yn = 0.0
        for k in range(y.size):
            yn += y[k]*y[k]
        yn = math.sqrt(yn) + 1e-18
        for k in range(y.size):
            y[k] /= yn
        return y
    c = math.cos(un)
    s = math.sin(un) / un
    y = np.empty_like(x)
    for k in range(x.size):
        y[k] = c * x[k] + s * u[k]
    yn = 0.0
    for k in range(y.size):
        yn += y[k]*y[k]
    yn = math.sqrt(yn) + 1e-18
    for k in range(y.size):
        y[k] /= yn
    return y

@njit(cache=True, fastmath=True)
def _l2_norm_vec(v):
    s = 0.0
    for k in range(v.size):
        s += v[k]*v[k]
    return math.sqrt(s) + 1e-18

# =============================================================================
# Losses and gradients
# mode = 0  -> inverse-power repulsion   L = sum_{i<j} 1 / ||x_i - x_j||^p
# mode = 1  -> log-sum-exp of dot prods  L = tau * log( sum_{i<j} exp( (xi·xj)/tau ) )
# mode = 2  -> hinge on dot prods        L = sum_{i<j} max(0, xi·xj - t_cap)^2   (active-set via margin)
# =============================================================================
@njit(cache=True, fastmath=True)
def spherical_loss_and_grad(X, N, dim, mode, p_power, lse_tau, t_cap, hinge_margin):
    P = X.reshape((N, dim))
    G = np.zeros_like(P)
    L = 0.0

    if mode == 0:
        for i in range(N):
            xi = P[i]
            for j in range(i+1, N):
                xj = P[j]
                s = 0.0
                for k in range(dim):
                    v = xi[k] - xj[k]
                    s += v*v
                d = math.sqrt(s) + EPS
                inv = 1.0 / (d**p_power)
                L += inv
                coeff = -p_power / (d**(p_power + 2.0))
                for k in range(dim):
                    vij = (xi[k] - xj[k])
                    gi = coeff * vij
                    G[i, k] += gi
                    G[j, k] -= gi
        return L, G.ravel()

    if mode == 1:
        maxz = -1e300
        for i in range(N):
            xi = P[i]
            for j in range(i+1, N):
                xj = P[j]
                dot = 0.0
                for k in range(dim):
                    dot += xi[k]*xj[k]
                z = dot / lse_tau
                if z > maxz:
                    maxz = z
        S = 0.0
        for i in range(N):
            xi = P[i]
            for j in range(i+1, N):
                xj = P[j]
                dot = 0.0
                for k in range(dim):
                    dot += xi[k]*xj[k]
                z = (dot / lse_tau) - maxz
                ez = math.exp(z)
                S += ez
                for k in range(dim):
                    G[i, k] += (ez / lse_tau) * xj[k]
                    G[j, k] += (ez / lse_tau) * xi[k]
        if S <= 0.0:
            return 0.0, G.ravel()
        L = lse_tau * (math.log(S) + maxz)
        return L, G.ravel()

    # mode == 2: hinge on dot products with active-set margin
    t_eff = t_cap - hinge_margin
    for i in range(N):
        xi = P[i]
        for j in range(i+1, N):
            xj = P[j]
            dot = 0.0
            for k in range(dim):
                dot += xi[k]*xj[k]
            # quick skip if far below cap - margin
            if dot <= t_eff:
                continue
            s = dot - t_cap
            if s > 0.0:
                L += s*s
                coeff = 2.0*s
                for k in range(dim):
                    G[i,k] += coeff * xj[k]
                    G[j,k] += coeff * xi[k]
    return L, G.ravel()

# =============================================================================
# SRP on sphere with backtracking + seeding
# =============================================================================
@njit(cache=True, fastmath=True)
def srp_sphere(
    X0, N, dim,
    Imax, m, step_pos, beta, backtrack,
    mode, p_power, lse_tau, t_cap, hinge_margin,
    noise_scale,
    seed
):
    # Seed Numba RNG per run
    np.random.seed(np.int64(seed))

    Xc = X0.copy()
    eta = 1.0

    for _ in range(Imax):
        # 1) jitter in tangent spaces
        Xtrial = Xc.copy()
        P = Xtrial.reshape((N, dim))
        for i in range(N):
            z = np.random.normal(0.0, 1.0, size=dim)
            # project to tangent
            dot = 0.0
            for k in range(dim):
                dot += z[k] * P[i, k]
            for k in range(dim):
                z[k] = z[k] - dot * P[i, k]
            # scale
            zn = _l2_norm_vec(z)
            if zn > 0.0:
                for k in range(dim):
                    z[k] = z[k] * (eta * noise_scale * step_pos / zn)
            P[i, :] = _expmap_sphere(P[i, :], z)

        # 2) m inner gradient steps with backtracking
        for __ in range(m):
            L0, g = spherical_loss_and_grad(Xtrial, N, dim, mode, p_power, lse_tau, t_cap, hinge_margin)
            G = g.reshape((N, dim))
            GT = np.zeros_like(G)
            for i in range(N):
                GT[i, :] = _tangent_projection(G[i, :], P[i, :])

            Pprop = P.copy()
            for i in range(N):
                gn = _l2_norm_vec(GT[i, :])
                if gn > 0.0:
                    u = - (eta * step_pos / gn) * GT[i, :]
                else:
                    u = np.zeros(dim)
                Pprop[i, :] = _expmap_sphere(P[i, :], u)

            Xprop = Pprop.ravel()
            Lprop, _ = spherical_loss_and_grad(Xprop, N, dim, mode, p_power, lse_tau, t_cap, hinge_margin)

            bt = 0
            cur_step = step_pos
            while Lprop > L0 and bt < backtrack:
                cur_step *= 0.5
                for i in range(N):
                    gn = _l2_norm_vec(GT[i, :])
                    if gn > 0.0:
                        u = - (eta * cur_step / gn) * GT[i, :]
                    else:
                        u = np.zeros(dim)
                    Pprop[i, :] = _expmap_sphere(P[i, :], u)
                Xprop = Pprop.ravel()
                Lprop, _ = spherical_loss_and_grad(Xprop, N, dim, mode, p_power, lse_tau, t_cap, hinge_margin)
                bt += 1

            P[:, :] = Pprop[:, :]

        Xc = P.ravel()
        eta *= beta

    return Xc

# =============================================================================
# Polishing pass: short Riemannian GD with backtracking
# =============================================================================
@njit(cache=True, fastmath=True)
def polish_sphere(
    X0, N, dim,
    iters, step_pos, backtrack,
    mode, p_power, lse_tau, t_cap, hinge_margin
):
    X = X0.copy()
    P = X.reshape((N, dim))

    for _ in range(iters):
        L0, g = spherical_loss_and_grad(X, N, dim, mode, p_power, lse_tau, t_cap, hinge_margin)
        G = g.reshape((N, dim))
        GT = np.zeros_like(G)
        for i in range(N):
            GT[i, :] = _tangent_projection(G[i, :], P[i, :])

        Pprop = P.copy()
        for i in range(N):
            gn = _l2_norm_vec(GT[i, :])
            if gn > 0.0:
                u = - (step_pos / gn) * GT[i, :]
            else:
                u = np.zeros(dim)
            Pprop[i, :] = _expmap_sphere(P[i, :], u)

        Xprop = Pprop.ravel()
        Lprop, _ = spherical_loss_and_grad(Xprop, N, dim, mode, p_power, lse_tau, t_cap, hinge_margin)

        bt = 0
        cur_step = step_pos
        while Lprop > L0 and bt < backtrack:
            cur_step *= 0.5
            for i in range(N):
                gn = _l2_norm_vec(GT[i, :])
                if gn > 0.0:
                    u = - (cur_step / gn) * GT[i, :]
                else:
                    u = np.zeros(dim)
                Pprop[i, :] = _expmap_sphere(P[i, :], u)
            Xprop = Pprop.ravel()
            Lprop, _ = spherical_loss_and_grad(Xprop, N, dim, mode, p_power, lse_tau, t_cap, hinge_margin)
            bt += 1

        P[:, :] = Pprop[:, :]
        X = P.ravel()

    return X

# =============================================================================
# Metrics / helpers (Python-level)
# =============================================================================
def min_pairwise_chord_py(P):
    N = P.shape[0]
    best = 1e9
    for i in range(N):
        diffs = P[i+1:] - P[i]
        ds = np.linalg.norm(diffs, axis=1)
        if ds.size > 0:
            b = ds.min()
            if b < best: best = b
    return float(best)

def min_pairwise_angle_py(P):
    # angle = arccos(dot), units radians
    dots = np.clip(P @ P.T, -1.0, 1.0)
    N = P.shape[0]
    mask = np.triu(np.ones((N,N), dtype=bool), k=1)
    d = dots[mask]
    ang = np.arccos(np.max(d))  # min angle corresponds to max dot
    return float(ang)

def max_pairwise_dot_py(P):
    dots = np.clip(P @ P.T, -1.0, 1.0)
    N = P.shape[0]
    mask = np.triu(np.ones((N,N), dtype=bool), k=1)
    return float(np.max(dots[mask]))

# =============================================================================
# Plotting (only for dim == 3)
# =============================================================================
def plot_first_k_spherical_samples(data_tensor, k, out_dir, filename_prefix="tammes"):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    M, dim, _ = data_tensor.shape
    if dim != 3:
        return

    k = min(k, M)
    for s in range(k):
        X = data_tensor[s, :, :].T  # (N,3)
        N = X.shape[0]
        dots = np.clip(X @ X.T, -1.0, 1.0)
        mask = ~np.eye(N, dtype=bool)
        min_ang = np.min(np.arccos(dots[mask]))
        min_ang_deg = np.degrees(min_ang)

        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111, projection='3d')
        u = np.linspace(0, 2*np.pi, 100)
        v = np.linspace(0, np.pi, 50)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(xs, ys, zs, linewidth=0.3, alpha=0.3)
        ax.scatter(X[:,0], X[:,1], X[:,2], s=20)
        ax.set_title(f"Sample #{s}, N={N}, min angle={min_ang_deg:.2f}°")
        ax.set_box_aspect([1,1,1])
        ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)

        out_path = os.path.join(out_dir, f"{filename_prefix}_N{N}_minang_{min_ang_deg:.2f}.png")
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

# =============================================================================
# Main generator with multi-stage schedule: inv_p -> LSE schedule -> hinge drive
# =============================================================================
def generate_tammes_dataset():
    sec = "tammes_SRP"

    # ---------- Problem ----------
    N        = _get_cfg(sec, "num_points",   50)
    dim      = _get_cfg(sec, "dim",          3)
    M        = _get_cfg(sec, "num_samples",  1000)

    # ---------- Seeding / rotations ----------
    base_seed  = _get_cfg(sec, "base_seed",       123456)
    rotate_fin = _get_cfg(sec, "random_rotate_final", True)

    # ---------- Output & plotting ----------
    out_dir    = _get_cfg(sec, "output_dir",      "./outputs_tammes")
    plot_k     = _get_cfg(sec, "plot_k",          0)
    plot_dir   = os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    stamp      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dataset_fn = os.path.join(out_dir, f"tammes_srp_generated_{M}x{N}d{dim}_{stamp}.pt")
    metrics_fn = os.path.join(out_dir, f"tammes_srp_metrics_{M}x{N}d{dim}_{stamp}.csv")

    # ---------- Common SRP parameters ----------
    backtrack     = _get_cfg(sec, "srp_backtrack",   3)
    beta          = _get_cfg(sec, "srp_beta",        0.985)

    # ---------- Stage A: inverse-power (optional) ----------
    use_inv_p     = _get_cfg(sec, "invp_enable",     True)
    invp_Imax     = _get_cfg(sec, "invp_Imax",       250)
    invp_m        = _get_cfg(sec, "invp_m",          20)
    invp_step     = _get_cfg(sec, "invp_step_pos",   0.04)
    invp_noise    = _get_cfg(sec, "invp_noise_scale",0.5)
    invp_p        = _get_cfg(sec, "invp_p_power",    12)
    invp_polish   = _get_cfg(sec, "invp_polish_iters", 100)
    invp_polstep  = _get_cfg(sec, "invp_polish_step",  0.02)

    # ---------- Stage B: LSE schedule ----------
    lse_enable    = _get_cfg(sec, "lse_enable", True)
    # Comma-separated temperatures, e.g. "0.08,0.04,0.02,0.01"
    lse_schedule  = _get_cfg_list_floats(sec, "lse_tau_schedule", "0.08,0.04,0.02,0.01")
    lse_Imax      = _get_cfg(sec, "lse_Imax_per_stage", 150)
    lse_m         = _get_cfg(sec, "lse_m",              20)
    lse_step0     = _get_cfg(sec, "lse_step_pos_start", 0.035)
    lse_step_min  = _get_cfg(sec, "lse_step_pos_final", 0.015)
    lse_noise     = _get_cfg(sec, "lse_noise_scale",    0.4)
    lse_polish    = _get_cfg(sec, "lse_polish_iters",   180)
    lse_polstep   = _get_cfg(sec, "lse_polish_step",    0.015)

    # ---------- Stage C: Hinge finishing ----------
    hinge_enable  = _get_cfg(sec, "hinge_enable", True)
    hinge_Imax    = _get_cfg(sec, "hinge_Imax_per_stage", 120)
    hinge_m       = _get_cfg(sec, "hinge_m",             20)
    hinge_step    = _get_cfg(sec, "hinge_step_pos",      0.015)
    hinge_noise   = _get_cfg(sec, "hinge_noise_scale",   0.3)
    hinge_margin  = _get_cfg(sec, "hinge_active_margin", 0.02)  # pairs below (t - margin) skipped
    hinge_reduce  = _get_cfg(sec, "hinge_reduce_frac",   0.01)  # 1% tighter per stage
    hinge_stages  = _get_cfg(sec, "hinge_max_stages",    12)
    hinge_polish  = _get_cfg(sec, "hinge_polish_iters",  300)
    hinge_polstep = _get_cfg(sec, "hinge_polish_step",   0.012)
    hinge_tol     = _get_cfg(sec, "hinge_loss_tol",      1e-8)  # consider "feasible" if below

    print(f"Generating {M} Tammes samples on S^{dim-1} (N={N})")
    with open(metrics_fn, "w") as mf:
        mf.write("sample,min_chord,min_angle_rad,min_angle_deg,cosine_min_degree,loss\n")

    data = np.zeros((M, dim, N), dtype=np.float32)
    per_sample_min = np.empty(M, dtype=np.float64)

    bar = tqdm(range(M), desc="Generating Tammes packings")
    for s in bar:
        # Start
        P = sample_uniform_on_sphere(N, dim)
        X = P.ravel()

        seed_base = int(base_seed + s * 1000)

        # ---- Stage A: inverse-power spread
        if use_inv_p:
            X = srp_sphere(
                X, N, dim,
                Imax=invp_Imax, m=invp_m, step_pos=invp_step, beta=beta, backtrack=backtrack,
                mode=0, p_power=invp_p, lse_tau=0.0, t_cap=0.0, hinge_margin=0.0,
                noise_scale=invp_noise,
                seed=np.int64(seed_base + 11)
            )
            X = polish_sphere(
                X, N, dim,
                iters=invp_polish, step_pos=invp_polstep, backtrack=backtrack,
                mode=0, p_power=invp_p, lse_tau=0.0, t_cap=0.0, hinge_margin=0.0
            )

        # ---- Stage B: LSE annealing
        if lse_enable and len(lse_schedule) > 0:
            # interpolate step from start -> final across schedule
            for idx, tau in enumerate(lse_schedule):
                # linear interpolation of step size over schedule
                if len(lse_schedule) > 1:
                    t = idx / (len(lse_schedule) - 1.0)
                else:
                    t = 1.0
                step_here = (1.0 - t) * lse_step0 + t * lse_step_min

                X = srp_sphere(
                    X, N, dim,
                    Imax=lse_Imax, m=lse_m, step_pos=step_here, beta=beta, backtrack=backtrack,
                    mode=1, p_power=0.0, lse_tau=tau, t_cap=0.0, hinge_margin=0.0,
                    noise_scale=lse_noise,
                    seed=np.int64(seed_base + 101 + idx)
                )
                X = polish_sphere(
                    X, N, dim,
                    iters=lse_polish, step_pos=lse_polstep, backtrack=backtrack,
                    mode=1, p_power=0.0, lse_tau=tau, t_cap=0.0, hinge_margin=0.0
                )

        # ---- Stage C: Hinge finishing (drive t down)
        if hinge_enable:
            # current max dot sets initial cap
            Ptmp = X.reshape(N, dim)
            Ptmp = Ptmp / (np.linalg.norm(Ptmp, axis=1, keepdims=True) + 1e-18)
            t_cap = max_pairwise_dot_py(Ptmp)

            for h in range(hinge_stages):
                t_try = t_cap * (1.0 - hinge_reduce)
                # SRP at hinge
                Xcand = srp_sphere(
                    X, N, dim,
                    Imax=hinge_Imax, m=hinge_m, step_pos=hinge_step, beta=beta, backtrack=backtrack,
                    mode=2, p_power=0.0, lse_tau=0.0, t_cap=t_try, hinge_margin=hinge_margin,
                    noise_scale=hinge_noise,
                    seed=np.int64(seed_base + 200 + h)
                )
                # polish at hinge
                Xcand = polish_sphere(
                    Xcand, N, dim,
                    iters=hinge_polish, step_pos=hinge_polstep, backtrack=backtrack,
                    mode=2, p_power=0.0, lse_tau=0.0, t_cap=t_try, hinge_margin=hinge_margin
                )

                # Evaluate hinge loss to decide accept / backtrack
                L_hinge, _ = spherical_loss_and_grad(
                    Xcand, N, dim, 2, 0.0, 0.0, t_try, hinge_margin
                )
                if L_hinge <= hinge_tol:
                    # Accept tighter cap
                    X = Xcand
                    t_cap = t_try
                else:
                    # Too tight; relax slightly (halfway back) and continue polishing once
                    t_cap = 0.5 * (t_cap + t_try)
                    X = polish_sphere(
                        X, N, dim,
                        iters=max(hinge_polish // 2, 50), step_pos=hinge_polstep, backtrack=backtrack,
                        mode=2, p_power=0.0, lse_tau=0.0, t_cap=t_cap, hinge_margin=hinge_margin
                    )

        # ---- Final projection & optional rotation
        P = X.reshape((N, dim))
        P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-18)

        if rotate_fin and dim == 3:
            A = np.random.normal(size=(dim, dim))
            Q, _ = np.linalg.qr(A)
            if np.linalg.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]
            P = P @ Q

        # ---- Metrics
        mc  = min_pairwise_chord_py(P)
        ma  = min_pairwise_angle_py(P)             # radians
        mad = float(np.degrees(ma))                # degrees
        cos_min = float(math.cos(ma))              # requested metric (equals max pairwise dot)

        # Report the final loss value for info (use hinge if enabled else LSE else inv_p)
        if hinge_enable:
            L, _ = spherical_loss_and_grad(P.ravel(), N, dim, 2, 0.0, 0.0, max_pairwise_dot_py(P), hinge_margin)
        elif lse_enable and len(lse_schedule) > 0:
            L, _ = spherical_loss_and_grad(P.ravel(), N, dim, 1, 0.0, lse_schedule[-1], 0.0, 0.0)
        elif use_inv_p:
            L, _ = spherical_loss_and_grad(P.ravel(), N, dim, 0, invp_p, 0.0, 0.0, 0.0)
        else:
            L = 0.0

        with open(metrics_fn, "a") as mf:
            mf.write(f"{s},{mc:.8f},{ma:.8f},{mad:.8f},{cos_min:.8f},{L:.8e}\n")

        data[s, :, :] = P.T.astype(np.float32)
        per_sample_min[s] = mc
        bar.set_postfix(min_chord=f"{mc:.4f}", min_angle_deg=f"{mad:.2f}")

    # Sort by min chord distance (descending = better)
    sorted_indices = np.argsort(-per_sample_min)
    data = data[sorted_indices]

    torch.save(torch.from_numpy(data), dataset_fn)
    print(f"\nSaved dataset:  {dataset_fn}")
    print(f"Saved metrics:  {metrics_fn}")

    if plot_k > 0 and int(dim) == 3:
        plot_first_k_spherical_samples(data, plot_k, plot_dir, filename_prefix="tammes")
        print(f"Saved plots:   {plot_dir}")

# =============================================================================
# Entrypoint
# =============================================================================
if __name__ == "__main__":
    generate_tammes_dataset()
