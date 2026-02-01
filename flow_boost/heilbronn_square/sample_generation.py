# ---- OpenMP guard: must be FIRST (before numpy/torch/matplotlib) ----
import os
import platform
if platform.system() == "Darwin" and os.environ.get("SPHEREPACK_DISABLE_KMP_HACK") != "1":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

# Ensure repo root is importable when launching file directly from VS Code sub-workspaces
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import math
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from scipy.optimize import minimize, Bounds, NonlinearConstraint

import numba as nb
from numba import njit

# =============================================================================
# Utilities / small constants
# =============================================================================
EPS = 1e-12

def _get_cfg(section, key, fallback):
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

def _triangle_area_exact(p1, p2, p3):
    ux, uy = p2[0] - p1[0], p2[1] - p1[1]
    vx, vy = p3[0] - p1[0], p3[1] - p1[1]
    return 0.5 * abs(ux * vy - uy * vx)

def find_min_area_and_triangles(pts, atol=1e-12, rtol=1e-9):
    N = pts.shape[0]
    if N < 3:
        return 0.0, []
    A_min = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                A = _triangle_area_exact(pts[i], pts[j], pts[k])
                if A < A_min:
                    A_min = A
    tol = max(atol, rtol * max(A_min, 1.0))
    tris = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                A = _triangle_area_exact(pts[i], pts[j], pts[k])
                if abs(A - A_min) <= tol:
                    tris.append((i, j, k))
    return A_min, tris

@njit(cache=True, fastmath=True)
def _l2_norm(v):
    s = 0.0
    for k in range(v.size):
        s += v[k]*v[k]
    return math.sqrt(s) + 1e-18

@njit(cache=True, fastmath=True)
def _triangle_area_and_grads(p1x, p1y, p2x, p2y, p3x, p3y, eps_abs=1e-12):
    ux = p2x - p1x
    uy = p2y - p1y
    vx = p3x - p1x
    vy = p3y - p1y
    cross = ux*vy - uy*vx
    denom = math.sqrt(cross*cross + eps_abs)
    dA_dcross = 0.5 * cross / denom
    dc_dp1x = (p2y - p3y)
    dc_dp1y = (p3x - p2x)
    dc_dp2x = (p3y - p1y)
    dc_dp2y = (p1x - p3x)
    dc_dp3x = (p1y - p2y)
    dc_dp3y = (p2x - p1x)
    dA_dp1x = dA_dcross * dc_dp1x
    dA_dp1y = dA_dcross * dc_dp1y
    dA_dp2x = dA_dcross * dc_dp2x
    dA_dp2y = dA_dcross * dc_dp2y
    dA_dp3x = dA_dcross * dc_dp3x
    dA_dp3y = dA_dcross * dc_dp3y
    A = 0.5 * denom
    return A, dA_dp1x, dA_dp1y, dA_dp2x, dA_dp2y, dA_dp3x, dA_dp3y

@njit(cache=True, fastmath=True)
def heilbronn_loss_and_grad(X, N, w_wall, beta_softmin,
                             eps_abs=1e-12, topk_K=0, topk_tol=1e-12):
    pts = X[:2*N].reshape((N, 2))
    g = np.zeros_like(X)
    L = 0.0
    # wall penalties
    for i in range(N):
        xi = pts[i,0]; yi = pts[i,1]
        v = -xi
        if v > 0.0:
            L += w_wall * (v*v)
            g[2*i + 0] += -2.0 * w_wall * v
        v = xi - 1.0
        if v > 0.0:
            L += w_wall * (v*v)
            g[2*i + 0] += 2.0 * w_wall * v
        v = -yi
        if v > 0.0:
            L += w_wall * (v*v)
            g[2*i + 1] += -2.0 * w_wall * v
        v = yi - 1.0
        if v > 0.0:
            L += w_wall * (v*v)
            g[2*i + 1] += 2.0 * w_wall * v
    T = N*(N-1)*(N-2)//6
    if T == 0:
        return L, g
    A_thresh = 1e99
    use_topk = (topk_K > 0) and (topk_K < T)
    if use_topk:
        K = topk_K
        buf = np.empty(K, dtype=np.float64)
        for kk in range(K):
            buf[kk] = 1e99
        def _buf_worst_val(arr):
            worst = -1.0
            worst_i = -1
            for ii in range(arr.size):
                if arr[ii] > worst:
                    worst = arr[ii]
                    worst_i = ii
            return worst, worst_i
        worst, worst_i = _buf_worst_val(buf)
        for i in range(N):
            xi, yi = pts[i,0], pts[i,1]
            for j in range(i+1, N):
                xj, yj = pts[j,0], pts[j,1]
                for k in range(j+1, N):
                    xk, yk = pts[k,0], pts[k,1]
                    A, _, _, _, _, _, _ = _triangle_area_and_grads(xi, yi, xj, yj, xk, yk, eps_abs)
                    if A < worst:
                        buf[worst_i] = A
                        worst, worst_i = _buf_worst_val(buf)
        A_thresh = worst
    Z = 0.0
    if use_topk:
        for i in range(N):
            xi, yi = pts[i,0], pts[i,1]
            for j in range(i+1, N):
                xj, yj = pts[j,0], pts[j,1]
                for k in range(j+1, N):
                    xk, yk = pts[k,0], pts[k,1]
                    A, _, _, _, _, _, _ = _triangle_area_and_grads(xi, yi, xj, yj, xk, yk, eps_abs)
                    if A <= A_thresh + topk_tol:
                        Z += math.exp(-beta_softmin * A)
    else:
        for i in range(N):
            xi, yi = pts[i,0], pts[i,1]
            for j in range(i+1, N):
                xj, yj = pts[j,0], pts[j,1]
                for k in range(j+1, N):
                    xk, yk = pts[k,0], pts[k,1]
                    A, _, _, _, _, _, _ = _triangle_area_and_grads(xi, yi, xj, yj, xk, yk, eps_abs)
                    Z += math.exp(-beta_softmin * A)
    if Z <= 0.0:
        return L, g
    neg_softmin = (math.log(Z) / beta_softmin)
    L += neg_softmin
    invZ = 1.0 / Z
    if use_topk:
        for i in range(N):
            xi, yi = pts[i,0], pts[i,1]
            for j in range(i+1, N):
                xj, yj = pts[j,0], pts[j,1]
                for k in range(j+1, N):
                    xk, yk = pts[k,0], pts[k,1]
                    A, d1x, d1y, d2x, d2y, d3x, d3y = _triangle_area_and_grads(xi, yi, xj, yj, xk, yk, eps_abs)
                    if A <= A_thresh + topk_tol:
                        w_t = math.exp(-beta_softmin * A) * invZ
                        coeff = -w_t
                        g[2*i+0] += coeff * d1x; g[2*i+1] += coeff * d1y
                        g[2*j+0] += coeff * d2x; g[2*j+1] += coeff * d2y
                        g[2*k+0] += coeff * d3x; g[2*k+1] += coeff * d3y
    else:
        for i in range(N):
            xi, yi = pts[i,0], pts[i,1]
            for j in range(i+1, N):
                xj, yj = pts[j,0], pts[j,1]
                for k in range(j+1, N):
                    xk, yk = pts[k,0], pts[k,1]
                    A, d1x, d1y, d2x, d2y, d3x, d3y = _triangle_area_and_grads(xi, yi, xj, yj, xk, yk, eps_abs)
                    w_t = math.exp(-beta_softmin * A) * invZ
                    coeff = -w_t
                    g[2*i+0] += coeff * d1x; g[2*i+1] += coeff * d1y
                    g[2*j+0] += coeff * d2x; g[2*j+1] += coeff * d2y
                    g[2*k+0] += coeff * d3x; g[2*k+1] += coeff * d3y
    return L, g

@njit(cache=True, fastmath=True)
def srp_adaptive_points(X, N, Imax, m, step_center, beta_sched_decay, backtrack,
                         w_wall,
                         beta_softmin_start, beta_softmin_final,
                         eps_abs,
                         topk_K, topk_tol):
    Xc = X.copy()
    eta = 1.0
    for it in range(Imax):
        tfrac = (it + 0.0) / max(Imax - 1, 1)
        beta_t = beta_softmin_start * math.pow(beta_softmin_final / beta_softmin_start, tfrac)
        noise = np.zeros_like(Xc)
        for i in range(N):
            noise[2*i+0] = (np.random.rand()*2.0 - 1.0)*(eta * step_center)
            noise[2*i+1] = (np.random.rand()*2.0 - 1.0)*(eta * step_center)
        Xtrial = Xc + noise
        for __ in range(m):
            L0, g = heilbronn_loss_and_grad(
                Xtrial, N, w_wall, beta_t, eps_abs=eps_abs, topk_K=topk_K, topk_tol=topk_tol
            )
            step = np.empty_like(g)
            for i in range(N):
                step[2*i+0] = eta * step_center
                step[2*i+1] = eta * step_center
            gn = _l2_norm(g)
            if gn > 0.0:
                Xprop = Xtrial - step * (g / gn)
                Lprop, _ = heilbronn_loss_and_grad(
                    Xprop, N, w_wall, beta_t, eps_abs=eps_abs, topk_K=topk_K, topk_tol=topk_tol
                )
                bt = 0
                while Lprop > L0 and bt < backtrack:
                    for i in range(N):
                        step[2*i+0] *= 0.5
                        step[2*i+1] *= 0.5
                    Xprop = Xtrial - step * (g / gn)
                    Lprop, _ = heilbronn_loss_and_grad(
                        Xprop, N, w_wall, beta_t, eps_abs=eps_abs, topk_K=topk_K, topk_tol=topk_tol
                    )
                    bt += 1
                Xtrial = Xprop
        for i in range(N):
            if Xtrial[2*i+0] < 0.0: Xtrial[2*i+0] = 0.0
            if Xtrial[2*i+0] > 1.0: Xtrial[2*i+0] = 1.0
            if Xtrial[2*i+1] < 0.0: Xtrial[2*i+1] = 0.0
            if Xtrial[2*i+1] > 1.0: Xtrial[2*i+1] = 1.0
        Xc = Xtrial
        eta *= beta_sched_decay
    return Xc

def local_optimize_points(X0, N, w_wall,
                          beta_softmin_final, eps_abs,
                          topk_K, topk_tol,
                          gtol=1e-8, ftol=1e-12, maxiter=1000, maxcor=20):
    bounds = [(0.0, 1.0)] * (2*N)
    def fun(x):
        L, _ = heilbronn_loss_and_grad(x, N, w_wall, beta_softmin_final,
                                       eps_abs=eps_abs, topk_K=topk_K, topk_tol=topk_tol)
        return L
    def jac(x):
        _, g = heilbronn_loss_and_grad(x, N, w_wall, beta_softmin_final,
                                       eps_abs=eps_abs, topk_K=topk_K, topk_tol=topk_tol)
        return g
    res = minimize(fun=fun, x0=X0, method='L-BFGS-B', jac=jac, bounds=bounds,
                   options={'gtol': gtol, 'ftol': ftol, 'maxiter': maxiter, 'maxcor': maxcor})
    return res.x, res.fun

def local_optimize_points_maxmin(X0, N,
                                gtol=1e-9, ftol=1e-15, maxiter=500):
    # pack x (2N) + t
    def pack(x, t):
        return np.concatenate([x, np.array([t])])
    def unpack(z):
        return z[:-1], z[-1]
    def areas_of_x(x):
        pts = x.reshape(N,2)
        arr = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    arr.append(_triangle_area_exact(pts[i], pts[j], pts[k]))
        return np.array(arr, dtype=np.float64)
    def obj(z):
        x, t = unpack(z)
        return -t
    def cons_fun(z):
        x, t = unpack(z)
        return areas_of_x(x) - t
    x0 = np.clip(X0.copy(), 0.0, 1.0)
    A0 = np.min(areas_of_x(x0))
    t0 = A0 * 0.9
    z0 = pack(x0, t0)
    lb = np.concatenate([np.zeros(2*N), np.array([0.0])])
    ub = np.concatenate([np.ones(2*N), np.array([0.5])])
    bounds = Bounds(lb, ub)
    nlc = NonlinearConstraint(cons_fun, 0.0, np.inf)
    res = minimize(obj, z0, method="SLSQP", bounds=bounds,
                   constraints=[nlc],
                   options={'ftol': ftol, 'maxiter': maxiter, 'eps':1e-12})
    x_star, t_star = unpack(res.x)
    return x_star, t_star, res

@njit(cache=True, fastmath=True)
def min_wall_clearance_points(pts):
    N = pts.shape[0]
    best = 1e9
    for i in range(N):
        xi, yi = pts[i,0], pts[i,1]
        best = min(best, xi, 1.0-xi, yi, 1.0-yi)
    return best

@njit(cache=True, fastmath=True)
def min_pair_distance(pts):
    N = pts.shape[0]
    best = 1e9
    for i in range(N):
        xi, yi = pts[i,0], pts[i,1]
        for j in range(i+1, N):
            dx = xi - pts[j,0]
            dy = yi - pts[j,1]
            d  = math.sqrt(dx*dx + dy*dy)
            if d < best:
                best = d
    return best

@njit(cache=True, fastmath=True)
def min_triangle_area(pts, eps_abs=1e-12):
    N = pts.shape[0]
    best = 1e18
    for i in range(N):
        x1, y1 = pts[i,0], pts[i,1]
        for j in range(i+1, N):
            x2, y2 = pts[j,0], pts[j,1]
            for k in range(j+1, N):
                A, _, _, _, _, _, _ = _triangle_area_and_grads(x1, y1, x2, y2,
                                                               pts[k,0], pts[k,1], eps_abs)
                if A < best:
                    best = A
    return best

def plot_top_k_minarea_samples(data_tensor, k, out_dir, filename_prefix="heilbronn_top"):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    M = data_tensor.shape[0]
    min_areas = np.zeros(M, dtype=np.float64)
    for s in range(M):
        pts = data_tensor[s].T.astype(np.float64)
        min_areas[s], _ = find_min_area_and_triangles(pts)
    order = np.argsort(-min_areas)
    for rank in range(min(k, M)):
        s = int(order[rank])
        pts = data_tensor[s].T.astype(np.float64)
        A_min, tris = find_min_area_and_triangles(pts)
        fig, ax = plt.subplots(figsize=(5,5))
        ax.plot([0,1,1,0,0], [0,0,1,1,0], color='black')
        ax.scatter(pts[:,0], pts[:,1], s=12)
        for (i,j,k) in tris:
            poly = np.array([pts[i], pts[j], pts[k], pts[i]])
            ax.plot(poly[:,0], poly[:,1], linewidth=1.0)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_title(f"Top #{rank+1} (sample {s}), N={pts.shape[0]}, min area={A_min:.10f}, triangles={len(tris)}")
        # out_path contain date and time too
        out_path = os.path.join(out_dir, f"{filename_prefix}_n={pts.shape[0]}_Amin_{A_min:.10f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

# =============================================================================
# Max-Min local optimization (SLSQP) on top K_active smallest triangles only
# =============================================================================
def local_optimize_points_maxmin_active(X0, N, K_active,
                                        gtol=1e-9, ftol=1e-15, maxiter=500):
    """
    Solve: maximize t subject to area_{ijk}(x) ≥ t for the K_active smallest-area triangles (initially),
    x in [0,1]^{2N}.  
    Use SLSQP with constraints on only those K_active triangles.
    """
    def areas_of_x(x):
        pts = x.reshape(N,2)
        # compute all triangle areas
        arr = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    arr.append((i,j,k, _triangle_area_exact(pts[i], pts[j], pts[k])))
        # sort by area
        arr.sort(key=lambda item: item[3])
        # pick smallest K_active
        sel = arr[:K_active]
        # return list of (i,j,k,A)
        return sel

    # pack/unpack
    def pack(x, t):
        return np.concatenate([x, np.array([t])])
    def unpack(z):
        return z[:-1], z[-1]

    # objective: -t
    def obj(z):
        x, t = unpack(z)
        return -t

    # constraint fun for selected triangles
    def cons_fun(z):
        x, t = unpack(z)
        sel = areas_of_x(x)
        # we require for each selected triangle: A >= t
        return np.array([ item[3] - t for item in sel ], dtype=np.float64)

    x0 = np.clip(X0.copy(), 0.0, 1.0)
    sel0 = areas_of_x(x0)
    A0 = sel0[0][3]  # smallest area
    t0 = A0 * 0.9
    z0 = pack(x0, t0)

    lb = np.concatenate([np.zeros(2*N), np.array([0.0])])
    ub = np.concatenate([np.ones(2*N), np.array([0.5])])
    bounds = Bounds(lb, ub)

    nlc = NonlinearConstraint(cons_fun, 0.0, np.inf)

    res = minimize(obj, z0, method="SLSQP", bounds=bounds,
                   constraints=[nlc],
                   options={'ftol': ftol, 'maxiter': maxiter, 'eps':1e-12})
    x_star, t_star = unpack(res.x)
    return x_star, t_star, res

# =============================================================================
# Main generator, and push modes (unchanged except for calling new function)
# =============================================================================
def generate_heilbronn_dataset():
    sec = "heilbronn_SRP"
    N   = _get_cfg(sec, "num_points",   50)
    M   = _get_cfg(sec, "num_samples",  5000)
    Imax        = _get_cfg(sec, "srp_Imax",        400)
    m           = _get_cfg(sec, "srp_m",           30)
    beta_sched  = _get_cfg(sec, "srp_beta",        0.985)
    backtrack   = _get_cfg(sec, "srp_backtrack",   3)
    step_center = _get_cfg(sec, "srp_step_center", 0.05)
    w_wall      = _get_cfg(sec, "w_wall",          1.0)
    beta_softmin0    = _get_cfg(sec, "beta_softmin_start", 40.0)
    beta_softminF    = _get_cfg(sec, "beta_softmin_final", 300.0)
    eps_abs          = _get_cfg(sec, "area_eps_abs",       1e-12)
    topk_K           = _get_cfg(sec, "topk_K",             0)
    topk_tol         = _get_cfg(sec, "topk_tol",           1e-12)
    gtol        = _get_cfg(sec, "lbfgs_gtol",      1e-8)
    ftol        = _get_cfg(sec, "lbfgs_ftol",      1e-12)
    maxiter     = _get_cfg(sec, "lbfgs_maxiter",   1000)
    maxcor      = _get_cfg(sec, "lbfgs_maxcor",    20)
    maxmin_top_k = _get_cfg(sec, "maxmin_top_k",    100)
    K_active       = _get_cfg(sec, "K_active",         50)
    out_dir     = _get_cfg(sec, "output_dir",      "./outputs_heilbronn")
    plot_k      = _get_cfg(sec, "plot_k",          0)
    plot_dir    = os.path.join(out_dir, "plots")

    os.makedirs(out_dir, exist_ok=True)
    stamp      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dataset_fn = os.path.join(out_dir, f"heilbronn_srp_{M}x{N}_{stamp}.pt")
    metrics_fn = os.path.join(out_dir, f"heilbronn_metrics_{M}x{N}_{stamp}.csv")

    print(f"Generating {M} Heilbronn point sets in unit square (N={N})")
    with open(metrics_fn, "w") as mf:
        mf.write("sample,min_triangle_area,min_wall_clear,min_pair_dist,loss_after\n")

    data = np.zeros((M, 2, N), dtype=np.float32)

    for s in tqdm(range(M), desc="Generating point sets"):
        P0 = sample_uniform_points(N)
        X0 = P0.ravel()

        X_srp = srp_adaptive_points(
            X0, N,
            Imax=Imax, m=m,
            step_center=step_center,
            beta_sched_decay=beta_sched, backtrack=backtrack,
            w_wall=w_wall,
            beta_softmin_start=beta_softmin0, beta_softmin_final=beta_softminF,
            eps_abs=eps_abs,
            topk_K=topk_K, topk_tol=topk_tol
        )

        X_fin_l, L_fin_l = local_optimize_points(X_srp, N, w_wall,
                                                  beta_softminF, eps_abs,
                                                  topk_K, topk_tol,
                                                  gtol, ftol, maxiter, maxcor)

        pts = X_fin_l.reshape(N,2)
        A_min = float(min_triangle_area(pts, eps_abs))
        mwc   = float(min_wall_clearance_points(pts))
        mpd   = float(min_pair_distance(pts))
        with open(metrics_fn, "a") as mf:
            mf.write(f"{s},{A_min:.10f},{mwc:.10f},{mpd:.10f},{L_fin_l:.8e}\n")

        data[s, 0, :] = pts[:,0].astype(np.float32)
        data[s, 1, :] = pts[:,1].astype(np.float32)

    # Sort and apply max–min pass on top ones
    scores = np.array([ min_triangle_area(data[s].T.astype(np.float64), eps_abs) for s in range(M) ])
    sorted_indices = np.argsort(-scores)
    data = data[sorted_indices]
    print(f"Applying max–min active solver to top {maxmin_top_k}/{M} samples (K_active={K_active} triangles per sample)")
    for rank in tqdm(range(min(maxmin_top_k, M)), desc="Max–min optimization"):
        pts0 = data[rank].T.astype(np.float64)
        X0_  = pts0.ravel()
        x_star, t_star, res = local_optimize_points_maxmin_active(X0_, N, K_active,
                                                                  gtol=gtol, ftol=ftol, maxiter=500)
        pts_star = x_star.reshape(N,2)
        A_min_star = float(min_triangle_area(pts_star, eps_abs))
        if A_min_star > scores[rank]:
            data[rank,0,:] = pts_star[:,0].astype(np.float32)
            data[rank,1,:] = pts_star[:,1].astype(np.float32)
            scores[rank] = A_min_star
        with open(metrics_fn, "a") as mf:
            mf.write(f"maxmin_pass,rank={rank},orig_A={scores[rank]:.10f},new_A={A_min_star:.10f},t_star={t_star:.10f},success={res.success}\n")

    # Final resort after improvements
    sorted_indices_final = np.argsort(-scores)
    data = data[sorted_indices_final]

    torch.save(torch.from_numpy(data), dataset_fn)
    print(f"\nSaved dataset:  {dataset_fn}")
    print(f"Saved metrics:  {metrics_fn}")

    if plot_k > 0:
        plot_top_k_minarea_samples(data, plot_k, plot_dir, filename_prefix="heilbronn_mintriangles")

def final_push_existing_samples():
    sec = "heilbronn_SRP"
    N   = _get_cfg(sec, "num_points",   50)
    Imax        = _get_cfg(sec, "srp_Imax",        400)
    m           = _get_cfg(sec, "srp_m",           30)
    beta_sched  = _get_cfg(sec, "srp_beta",        0.985)
    backtrack   = _get_cfg(sec, "srp_backtrack",   3)
    step_center = _get_cfg(sec, "srp_step_center", 0.05)
    w_wall      = _get_cfg(sec, "w_wall",          1.0)
    beta_softmin0    = _get_cfg(sec, "beta_softmin_start", 40.0)
    beta_softminF    = _get_cfg(sec, "beta_softmin_final", 300.0)
    eps_abs          = _get_cfg(sec, "area_eps_abs",       1e-12)
    topk_K           = _get_cfg(sec, "topk_K",             0)
    topk_tol         = _get_cfg(sec, "topk_tol",           1e-12)
    gtol        = _get_cfg(sec, "lbfgs_gtol",      1e-8)
    ftol        = _get_cfg(sec, "lbfgs_ftol",      1e-12)
    maxiter     = _get_cfg(sec, "lbfgs_maxiter",   1000)
    maxcor      = _get_cfg(sec, "lbfgs_maxcor",    20)
    maxmin_top_k = _get_cfg(sec, "maxmin_top_k",    100)
    K_active       = _get_cfg(sec, "K_active",         50)
    out_dir     = _get_cfg(sec, "final_push_output", "./outputs_heilbronn_push")
    input_path  = _get_cfg(sec, "final_push_input", "")
    plot_k      = _get_cfg(sec, "plot_k",          0)
    plot_dir    = os.path.join(out_dir, "plots")

    assert isinstance(input_path, str) and len(input_path) > 0 and os.path.exists(input_path), \
           "Set heilbronn_SRP.final_push_input to a valid .pt file"

    os.makedirs(out_dir, exist_ok=True)
    stamp      = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dataset_fn = os.path.join(out_dir, f"heilbronn_srp_pushed_{stamp}.pt")
    metrics_fn = os.path.join(out_dir, f"heilbronn_metrics_pushed_{stamp}.csv")

    loaded = torch.load(input_path, map_location="cpu")
    arr = loaded.detach().cpu().numpy() if isinstance(loaded, torch.Tensor) else None
    if arr is None or arr.ndim != 3:
        raise ValueError(f"Expected a tensor at final_push_input, got shape {loaded.shape}")

    if arr.shape[1] == 2:
        M_in, d_in, N_in = arr.shape
    elif arr.shape[-1] == 2:
        arr = np.transpose(arr, (0,2,1))
        M_in, d_in, N_in = arr.shape
    else:
        raise ValueError(f"Second or last dimension must be 2; got shape {arr.shape}")

    if N_in != N:
        raise ValueError(f"Config N={N} but input samples have N={N_in}")

    K = M_in
    if K <= 0:
        raise ValueError(f"final_push_input contains no samples (M={M_in}).")
    print(f"Pushing {K} of loaded {M_in} samples (N={N}) using K_active={K_active}")
    with open(metrics_fn, "w") as mf:
        mf.write("sample,min_triangle_area_before,min_triangle_area_after,t_star,success\n")

    data = np.zeros((K,2,N), dtype=np.float32)

    with tqdm(total=K, desc="[Pushing Samples]", unit="sample") as pbar:
        for s in range(K):
            pts0 = arr[s].T.astype(np.float64)
            X0_  = pts0.ravel()

            X_srp = srp_adaptive_points(
                X0_, N,
                Imax=Imax, m=m,
                step_center=step_center,
                beta_sched_decay=beta_sched, backtrack=backtrack,
                w_wall=w_wall,
                beta_softmin_start=beta_softmin0, beta_softmin_final=beta_softminF,
                eps_abs=eps_abs,
                topk_K=topk_K, topk_tol=topk_tol
            )

            x_star, t_star, res = local_optimize_points_maxmin_active(
                X_srp, N, K_active,
                gtol=gtol, ftol=ftol, maxiter=500
            )
            pts_star = x_star.reshape(N,2)
            A_before = float(min_triangle_area(pts0, eps_abs))
            A_after  = float(min_triangle_area(pts_star, eps_abs))

            data[s,0,:] = pts_star[:,0].astype(np.float32)
            data[s,1,:] = pts_star[:,1].astype(np.float32)

            with open(metrics_fn, "a") as mf:
                mf.write(f"{s},{A_before:.10f},{A_after:.10f},{t_star:.10f},{res.success}\n")

            pbar.update(1)

    torch.save(torch.from_numpy(data), dataset_fn)
    print(f"\nSaved pushed dataset:  {dataset_fn}")
    print(f"Saved metrics:         {metrics_fn}")

    if plot_k > 0:
        plot_top_k_minarea_samples(data, plot_k, plot_dir, filename_prefix="heilbronn_pushed_mintriangles")


def main(state=None):
    sec  = "heilbronn_SRP"
    mode = _get_cfg(sec, "mode", "training_set_gen").strip().lower()

    if mode == "training_set_gen":
        generate_heilbronn_dataset()
        out_dir = _get_cfg(sec, "output_dir", "./outputs_heilbronn")
    elif mode == "final_push":
        final_push_existing_samples()
        out_dir = _get_cfg(sec, "final_push_output", "./outputs_heilbronn_push")
    else:
        raise ValueError(f"Unknown mode '{mode}' (use training_set_gen|final_push)")

    # pick newest .pt in out_dir and store in state.pushed_samples_path
    if state is not None and os.path.isdir(out_dir):
        pts = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".pt")]
        if pts:
            newest = max(pts, key=os.path.getmtime)
            state.set_pushed_samples_path(newest)

if __name__ == "__main__":
    main()