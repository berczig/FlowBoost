# -*- coding: utf-8 -*-
"""
Flow-Matching model for packing circles in a square, maximizing sum of radii.

Includes:
  - Model (FlowSetTransformer) + FM training loop with circle-aware penalties
  - Physics-Constrained sampling (GAS):
        ODE step -> center projection -> LP projection of radii (max sum r) -> proximal relaxation -> final polish
  - Dataset wrapper for tensors of shape (M, 3, N) with rows [x, y, r]
  - Validator (batch metrics + CSV)
  - Plotters (single + batch)

Notes:
  * scipy.optimize.linprog (HiGHS) is used for the LP projection. See SciPy docs.
  * Circles plotted via matplotlib.patches.Circle.
  * Pairwise distances via torch.cdist.
  * Flow Matching background: Lipman et al., "Flow Matching for Generative Modeling" (2022).
"""
import os
import math
import time
import numpy as np
import re
from datetime import datetime
from typing import Dict, Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from tqdm import tqdm

# --- Optional config (reads from [flow_matching_sumradii]) ---
try:
    from flow_boost import cfg
    HAS_CFG = True
except Exception:
    HAS_CFG = False

import schedulefree
from x_transformers import Encoder
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.utils import ModelWrapper
from flow_matching.solver import ODESolver

# =============================
# Flow-Matching path (CondOT)
# =============================
FM_PATH = AffineProbPath(scheduler=CondOTScheduler())

# =============================
# Transformer Model
# =============================
class FlowSetTransformer(nn.Module):
    class TimeEmbedFourier(nn.Module):
        def __init__(self, out_dim: int, fourier_dim: int, hidden: int, sigma: float = 1.0, include_poly: bool = True):
            super().__init__()
            fourier_dim = max(16, fourier_dim)
            self.register_buffer("freqs", torch.randn(fourier_dim) * sigma, persistent=False)
            self.include_poly = include_poly
            in_dim = 2 * fourier_dim
            if include_poly:
                in_dim += 4  # [t, t^2, t^3, log(1+t)]
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, out_dim),
            )

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            if t.ndim == 2:
                assert t.shape[1] == 1, f"t must be (B,) or (B,1); got {t.shape}"
                t = t[:, 0]
            t = t.contiguous()
            angles = (t[:, None] * self.freqs[None, :]) * (2.0 * math.pi)
            sin_feat = torch.sin(angles)
            cos_feat = torch.cos(angles)
            feats = [sin_feat, cos_feat]
            if self.include_poly:
                t1 = t
                t2 = t1 * t1
                t3 = t2 * t1
                tlog = torch.log1p(t1.clamp_min(0))
                feats += [t1[:, None], t2[:, None], t3[:, None], tlog[:, None]]
            h = torch.cat(feats, dim=-1)
            return self.net(h)

    def __init__(self, d: int = 3, **st_kwargs):
        super().__init__()
        self.d = d
        model_dim = int(st_kwargs.get("dim_hidden", 512))
        heads = int(st_kwargs.get("num_heads", 8))
        depth = int(st_kwargs.get("num_isab", st_kwargs.get("depth", 6)))
        attn_do = float(st_kwargs.get("attn_dropout", 0.1))
        ff_do = float(st_kwargs.get("ff_dropout", 0.1))
        dim_time = int(st_kwargs.get("dim_time", max(64, 4 * d)))
        time_F = int(st_kwargs.get("time_fourier_dim", max(16, dim_time // 2)))
        time_hid = int(st_kwargs.get("time_hidden", 2 * dim_time))
        sigma = float(st_kwargs.get("time_fourier_sigma", 1.0))

        # conditioning
        cond_dim_in  = int(st_kwargs.get("cond_dim_in", 4))
        cond_hidden  = int(st_kwargs.get("cond_hidden", max(64, dim_time)))
        self.uses_cond = cond_dim_in > 0

        self.time_emb = self.TimeEmbedFourier(out_dim=dim_time, fourier_dim=time_F, hidden=time_hid,
                                              sigma=sigma, include_poly=True)

        if self.uses_cond:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim_in, cond_hidden),
                nn.SiLU(),
                nn.Linear(cond_hidden, cond_hidden)
            )
            dim_cond = cond_hidden
        else:
            self.cond_mlp = None
            dim_cond = 0

        self.token_in = nn.Linear(d + dim_time + dim_cond, model_dim)

        self.film_in_dim = dim_time + dim_cond
        self.timecond_to_film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.film_in_dim, 2 * model_dim)
        )

        self.encoder = Encoder(
            dim=model_dim,
            depth=depth,
            heads=heads,
            layer_dropout=0.1,
            attn_dropout=attn_do,
            ff_dropout=ff_do,
            use_rmsnorm=True,
            ff_glu=True,
            ff_no_bias=True,
            attn_flash=True
        )

        self.token_out = nn.Linear(model_dim, d)
        nn.init.zeros_(self.token_out.weight)
        nn.init.zeros_(self.token_out.bias)

    def forward(self, t: torch.Tensor, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        B, d, N = x.shape
        assert d == self.d
        device = x.device
        t = t.to(device=device, dtype=x.dtype)
        if t.ndim == 2:
            assert t.shape[1] == 1, f"t must be (B,) or (B,1); got {t.shape}"
            t_1d = t[:, 0]
        else:
            t_1d = t
        t_embed = self.time_emb(t_1d)
        if self.uses_cond:
            if cond is None:
                cond = torch.zeros(B, self.cond_mlp[0].in_features, device=device, dtype=x.dtype)
            else:
                cond = cond.to(device=device, dtype=x.dtype)
            c_embed = self.cond_mlp(cond)
        else:
            c_embed = None

        tokens = x.permute(0, 2, 1).contiguous()  # (B,N,d)
        if c_embed is not None:
            tc = torch.cat([t_embed, c_embed], dim=-1)
            tc_rep = tc[:, None, :].expand(B, N, -1)
            h = torch.cat([tokens, tc_rep], dim=-1)
            film_in = tc
        else:
            t_rep = t_embed[:, None, :].expand(B, N, -1)
            h = torch.cat([tokens, t_rep], dim=-1)
            film_in = t_embed
        h = self.token_in(h)
        gamma_beta = self.timecond_to_film(film_in)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma[:, None, :]
        beta  = beta[:, None, :]
        h = h * (1 + gamma) + beta

        h = self.encoder(h)
        out = self.token_out(h).permute(0, 2, 1).contiguous()
        return out

# =============================
# Helpers for circles (unit square)
# =============================
_TRI_CACHE: Dict[Any, Any] = {}

def sample_t(B, device, small_t_weight=0.5, gamma=2.0):
    s = torch.rand(B, device=device)
    t_small = s**gamma
    use_small = (torch.rand(B, device=device) < small_t_weight).float()
    return use_small * t_small + (1 - use_small) * s

def _wall_upper_bounds_xy(xy: torch.Tensor, L: float = 1.0) -> torch.Tensor:
    x = xy[:, 0, :]
    y = xy[:, 1, :]
    wall_ub = torch.minimum(torch.minimum(x, L - x), torch.minimum(y, L - y))
    return wall_ub.clamp_min(0.0)

def _clamp_box_circles(xyr: torch.Tensor, L: float = 1.0) -> torch.Tensor:
    B, d, N = xyr.shape
    assert d == 3
    xy = xyr[:, :2, :]
    r  = xyr[:, 2, :].clamp_min(0.0)
    wall_ub = _wall_upper_bounds_xy(xy, L=L)
    r = torch.minimum(r, wall_ub)
    x = xy[:, 0, :].clamp(r, L - r)
    y = xy[:, 1, :].clamp(r, L - r)
    out = torch.stack([x, y, r], dim=1)
    return out

def circle_wall_penalty(xyr, beta=80.0, p=2, margin=0.0, q=0.1, L: float = 1.0):
    B, _, N = xyr.shape
    xy = xyr[:, :2, :]
    r  = xyr[:, 2, :].clamp_min(0.0)
    wall_ub = _wall_upper_bounds_xy(xy, L=L) - margin
    gap = r - wall_ub
    v = F.softplus(beta * gap) / beta
    if p != 1:
        v = v.pow(p)
    k = max(1, int(q * N))
    topk, _ = torch.topk(v, k=k, dim=1, largest=True, sorted=False)
    return topk.mean()

def circle_pair_penalty(xyr, beta=80.0, p=2, margin=0.0, q=0.2, eps=1e-12):
    B, _, N = xyr.shape
    P = xyr[:, :2, :].permute(0, 2, 1).contiguous()
    D = torch.cdist(P, P).clamp_min(eps)
    r = xyr[:, 2, :]
    S = r[:, :, None] + r[:, None, :]
    tri = _TRI_CACHE.get((xyr.device, N))
    if tri is None:
        tri = torch.triu_indices(N, N, offset=1, device=xyr.device)
        _TRI_CACHE[(xyr.device, N)] = tri
    gap = (S - D)[..., tri[0], tri[1]] + margin
    v = F.softplus(beta * gap) / beta
    if p != 1:
        v = v.pow(p)
    k = max(1, int(q * v.size(1)))
    topk, _ = torch.topk(v, k=k, dim=1, largest=True, sorted=False)
    return topk.mean()

def _x0_projection_like(u_pred, x_1, t_in, path):
    eps_t = 1e-3
    sch0 = path.scheduler(t_in)
    sch1 = path.scheduler((t_in + eps_t).clamp_max(1.0))
    alpha_dot = ((sch1.alpha_t - sch0.alpha_t) / eps_t).view(-1,1,1).to(u_pred.dtype)
    sigma_dot = ((sch1.sigma_t - sch0.sigma_t) / eps_t).view(-1,1,1).to(u_pred.dtype)
    alpha_dot = alpha_dot.sign() * alpha_dot.abs().clamp_min(1e-6)
    x0_proj = (u_pred - sigma_dot * x_1) / alpha_dot
    return x0_proj

# =============================
# Dataset wrapper (M,3,N)
# =============================
class CirclePackingDataset(Dataset):
    def __init__(self, path, scale_N=128):
        data = torch.load(path)  # (M,3,N) rows [x,y,r]
        assert data.ndim == 3 and data.size(1) == 3, f"expected (M,3,N); got {tuple(data.shape)}"
        self.data = data.contiguous().float()
        self.M, self.d, self.N = self.data.shape

        # simple conditioning: [N/scale, sum_r, min_pair_clear, min_wall_clear]
        xy = self.data[:, :2, :]
        r  = self.data[:, 2, :]
        sum_r = r.sum(dim=1)

        wall_ub = torch.minimum(torch.minimum(xy[:,0,:], 1-xy[:,0,:]),
                                torch.minimum(xy[:,1,:], 1-xy[:,1,:]))
        min_wall = (wall_ub - r).amin(dim=1)

        P = xy.permute(0,2,1).contiguous()
        D = torch.cdist(P, P)
        S = r[:, :, None] + r[:, None, :]
        eye = torch.eye(self.N, device=D.device, dtype=torch.bool)[None]
        D = D.masked_fill(eye, float('inf'))
        min_pair = (D - S).amin(dim=-1).amin(dim=-1)

        N_scaled = torch.full((self.M,), float(self.N)/float(scale_N), dtype=self.data.dtype)
        self.cond = torch.stack([N_scaled, sum_r, min_pair, min_wall], dim=1)
        print(f"Loaded {path}, shape {tuple(self.data.shape)} | cond {tuple(self.cond.shape)}")

    def __len__(self): return self.M
    def __getitem__(self, idx): return self.data[idx], self.cond[idx]

# =============================
# Training
# =============================
def train_flow_model(
    model, optimizer, loader, num_epochs,
    mse_strength, dist_strength,
    device, params, save_path
):
    model.train().to(device)
    optimizer.train()
    mse = nn.MSELoss()
    history = []
    # Supervised anchor for sum of radii (from config)
    sumr_strength = _get_cfg("flow_matching_sumradii", "sumr_strength", 0.2)

    for epoch in tqdm(range(num_epochs), desc="Training"):
        ep_losses = []
        ratio = 0.5 * (1 - np.cos(np.pi * min(1.0, epoch / (0.5 * num_epochs))))
        for x_0, cond in loader:
            x_0 = x_0.to(device)      # (B,3,N) [x,y,r]
            cond = cond.to(device)
            B = x_0.size(0)
            cond_in = cond

            # source x_1: random centers + small positive radii
            x_1 = torch.empty_like(x_0)
            x_1[:, 0:2, :] = torch.rand(B, 2, x_0.size(-1), device=device)
            x_1[:, 2, :] = (0.01 + 0.005 * torch.randn(B, x_0.size(-1), device=device)).clamp_min(1e-4)
            x_1 = _clamp_box_circles(x_1)
        
            t_in = sample_t(B, device=device, small_t_weight=0.5, gamma=2.0)

            path_sample = FM_PATH.sample(t=t_in, x_0=x_0, x_1=x_1)
            x_t  = path_sample.x_t
            dx_t = path_sample.dx_t

            u_pred = model(t_in.squeeze(-1), _clamp_box_circles(x_t), cond=cond_in)
            loss_fm = mse(u_pred, dx_t)

            # penalties on an x0-like reconstruction
            x0_like = _x0_projection_like(u_pred, x_1, t_in, FM_PATH)
            x0_like = _clamp_box_circles(x0_like)

            pen_pairs = circle_pair_penalty(x0_like, beta=160.0, p=2, margin=0.0, q=0.30)
            pen_walls = circle_wall_penalty(x0_like, beta=160.0, p=2, margin=0.0, q=0.15)

            # Sum-of-radii regression anchor (target from cond[:,1])
            target_sumr = cond[:, 1]                  # (B,)
            pred_sumr   = x0_like[:, 2, :].sum(dim=1) # (B,)
            loss_sumr   = F.mse_loss(pred_sumr, target_sumr)

            loss = (
                mse_strength * loss_fm
                + dist_strength * ratio * (pen_pairs + pen_walls)
                + float(sumr_strength) * loss_sumr
            )
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            ep_losses.append([loss_fm.item(), (pen_pairs+pen_walls).item(), loss_sumr.item(), loss.item()])

        avg = np.mean(ep_losses, axis=0)
        history.append(avg)
        print(f"Epoch {epoch+1}/{num_epochs} | FM={avg[0]:.4e} Pen={avg[1]:.4e} SumR={avg[2]:.4e} Tot={avg[3]:.4e}")

    history = np.array(history)

    # save (store st_kwargs and dim for reliable reloads)
    os.makedirs(save_path, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_path, f"flow_circles_{ts}.pth")
    torch.save({
        "model": model.state_dict(),
        "params": {
            "st_kwargs": params,
            "dim": int(getattr(model, "d", 3))
        },
        "history": history
    }, model_path)

    plt.figure(figsize=(12,6))
    plt.plot(history[:,0], label="FM MSE")
    plt.plot(history[:,1], label="Geom Penalty")
    plt.plot(history[:,2], label="SumR Anchor")
    plt.plot(history[:,3], label="Total")
    plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid()
    plt.savefig(os.path.join(save_path, f"train_losses_{ts}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return model, history

# =============================
# LP radii projection (maximize sum r) with robust fallback
# =============================
def _lp_project_radii_max_sum(xyr: torch.Tensor, L: float = 1.0, safety: float = 1e-9, pair_safety_mul: float = 0.995) -> torch.Tensor:
    """
    Maximize sum r subject to:
      0 <= r_i <= wall_ub_i
      r_i + r_j <= pair_safety_mul * dist(i,j)
    Returns radii (B,N). Uses SciPy HiGHS when available; otherwise a fast vectorized fallback.
    """
    # Try SciPy first
    try:
        import numpy as _np
        from scipy.optimize import linprog as _linprog

        xyr_np = xyr.detach().cpu().numpy()
        B, _, N = xyr_np.shape
        out_r = _np.empty((B, N), dtype=_np.float64)

        for b in range(B):
            C = xyr_np[b, :2, :].T  # (N,2)
            x = C[:,0]; y = C[:,1]
            wall_ub = _np.minimum.reduce([x, L - x, y, L - y])
            wall_ub = _np.clip(wall_ub - safety, 0.0, None)

            dx = x[:,None] - x[None,:]
            dy = y[:,None] - y[None,:]
            D  = _np.sqrt(dx*dx + dy*dy) * float(pair_safety_mul)

            c = -_np.ones(N, dtype=float)
            bounds = [(0.0, float(wall_ub[i])) for i in range(N)]

            rows, rhs = [], []
            for i in range(N):
                for j in range(i+1, N):
                    row = _np.zeros(N, dtype=float); row[i]=1.0; row[j]=1.0
                    rows.append(row)
                    rhs.append(max(D[i,j] - safety, 0.0))

            A_ub = _np.vstack(rows) if rows else None
            b_ub = _np.asarray(rhs, dtype=float) if rows else None

            res = _linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            if res.success:
                out_r[b] = _np.maximum(0.0, res.x - safety)
            else:
                raise RuntimeError("HiGHS failed")
        return torch.from_numpy(out_r).to(xyr.device, dtype=xyr.dtype)
    except Exception:
        pass

    # Fallback: vectorized equal-split projection (fast, near-optimal)
    with torch.no_grad():
        B, _, N = xyr.shape
        xy = xyr[:, :2, :]
        x, y = xy[:, 0, :], xy[:, 1, :]
        wall_ub = torch.minimum(torch.minimum(x, L - x), torch.minimum(y, L - y)).clamp_min(0.0)
        r = wall_ub.clone()

        P = xy.permute(0, 2, 1).contiguous()  # (B,N,2)
        for _ in range(80):
            D = torch.cdist(P, P).clamp_min(1e-12) * float(pair_safety_mul)
            S = r[:, :, None] + r[:, None, :]
            O = (S - D).clamp_min(0.0)  # overlap matrix (B,N,N)
            eye = torch.eye(N, device=xyr.device, dtype=torch.bool)[None]
            O = O.masked_fill(eye, 0.0)
            if float(O.max().item()) <= 1e-10:
                break
            # Equal split: each circle reduces half of its total overlaps
            reduce_i = 0.5 * O.sum(dim=2)  # (B,N)
            r = (r - reduce_i).clamp_min(0.0)
            r = torch.minimum(r, wall_ub)
        return r

# =============================
# GAS Sampling for circles
# =============================

def _sanitize_ode_method(val: str, default="midpoint") -> str:
    if not isinstance(val, str):
        return default
    # strip inline ; or # comments and extra spaces
    s = re.split(r'[;#]', val, maxsplit=1)[0].strip().lower()
    valid = {"dopri8","dopri5","bosh3","fehlberg2","adaptive_heun","euler","midpoint","heun2","heun3","rk4","explicit_adams","implicit_adams","fixed_adams","scipy_solver"}
    return s if s in valid else default

def _infer_st_kwargs_from_state_dict(sd: dict):
    """
    Infer FlowSetTransformer kwargs and input dim from a checkpoint state_dict.
    Returns (st_kwargs: dict, d_inferred: int)
    """
    # model_dim and d (output channels of token_out)
    if "token_out.weight" in sd:
        d_out, model_dim = sd["token_out.weight"].shape
        d_inferred = int(d_out)
    else:
        model_dim, _ = sd["token_in.weight"].shape
        d_inferred = 3

    # depth from max layer index
    layer_ids = []
    pat = re.compile(r"^encoder\.layers\.(\d+)\.")
    for k in sd.keys():
        m = pat.match(k)
        if m:
            layer_ids.append(int(m.group(1)))
    depth = (max(layer_ids) + 1) if layer_ids else 6

    # time dims
    if "time_emb.net.2.weight" in sd:
        dim_time, time_hidden = sd["time_emb.net.2.weight"].shape
    else:
        dim_time, time_hidden = 64, 128

    if "time_emb.net.0.weight" in sd:
        _, in_dim0 = sd["time_emb.net.0.weight"].shape
        time_fourier_dim = max((in_dim0 - 4) // 2, 16)
    else:
        time_fourier_dim = max(dim_time // 2, 16)

    # cond mlp dims
    if "cond_mlp.0.weight" in sd:
        cond_hidden, cond_dim_in = sd["cond_mlp.0.weight"].shape
    else:
        cond_dim_in = 4
        cond_hidden = max(64, dim_time)

    # pick heads dividing model_dim
    for h in (16, 12, 8, 6, 4, 2, 1):
        if model_dim % h == 0:
            num_heads = h
            break

    st_kw = {
        "dim_hidden": int(model_dim),
        "num_heads": int(num_heads),
        "depth": int(depth),
        "attn_dropout": 0.1,
        "ff_dropout": 0.1,
        "dim_time": int(dim_time),
        "time_fourier_dim": int(time_fourier_dim),
        "time_hidden": int(time_hidden),
        "time_fourier_sigma": 1.0,
        "cond_dim_in": int(cond_dim_in),
        "cond_hidden": int(cond_hidden),
    }
    return st_kw, int(d_inferred)

def _build_model_from_ckpt(checkpoint, device, fallback_kwargs):
    # Prefer saved st_kwargs (new checkpoints saved by this script)
    if isinstance(checkpoint, dict) and "params" in checkpoint and isinstance(checkpoint["params"], dict):
        params = checkpoint["params"]
        if "st_kwargs" in params:
            st_kw = params["st_kwargs"]
            dim = int(params.get("dim", 3))
            print("[FM] Rebuilding model from checkpoint params.")
            return FlowSetTransformer(dim, **st_kw).to(device)
        # Backward-compat: params may be st_kwargs directly
        if any(k in params for k in ("dim_hidden","num_heads","depth","dim_time","time_hidden","cond_hidden")):
            print("[FM] Rebuilding model from flat params (st_kwargs).")
            return FlowSetTransformer(int(params.get("dim", 3) or 3), **params).to(device)

    # Try inferring from state_dict
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        try:
            st_kw_inf, dim_inf = _infer_st_kwargs_from_state_dict(checkpoint["model"])
            print("[FM] Rebuilding model from inferred state_dict shapes:")
            print(f"     dim={dim_inf} | dim_hidden={st_kw_inf['dim_hidden']} | depth={st_kw_inf['depth']} | "
                  f"heads={st_kw_inf['num_heads']} | dim_time={st_kw_inf['dim_time']} | "
                  f"time_fourier_dim={st_kw_inf['time_fourier_dim']} | time_hidden={st_kw_inf['time_hidden']} | "
                  f"cond_dim_in={st_kw_inf['cond_dim_in']} | cond_hidden={st_kw_inf['cond_hidden']}")
            return FlowSetTransformer(dim_inf, **st_kw_inf).to(device)
        except Exception as e:
            print(f"[FM] Inference from state_dict failed ({e}); falling back to current config.")

    # Fallback: use current config kwargs
    print("[FM] Checkpoint did not include st_kwargs; using current config widths.")
    return FlowSetTransformer(3, **fallback_kwargs).to(device)

@torch.no_grad()
def sample_flow_model(
    model, optimizer, num_samples, batch_size, num_points,
    device, clip_min, clip_max, dim, cond_loader=None,
    # Read from [flow_matching_sumradii]
    L=1.0,                         # box_len
    gas_steps=40,
    ode_method='midpoint',
    ode_step_cap=0.05,
    proj_iters=6,
    alpha_proj=0.25,
    contact_q=1.0,
    wall_weight=1.0,
    wall_margin=0.05,
    prox_iters=8,
    prox_step=0.1,
    prox_lambda=1.5,
    final_passes=4,
    tol_finish=1e-8,
    # radii limits
    r_min_init=1e-4,
    r_max_init_frac=0.25,
    # de-novo conditioning bump
    de_novo_minsep_bump=0.0,
    cond_minsep_index=3
):
    """
    Physics-Constrained FM for circles in [0,L]^2 with variable radii.
    Returns: np.ndarray (num_samples, 3, N) rows [x,y,r]
    """
    eps_nrm = 1e-6

    def _FMVF(mdl, cond):
        class VF:
            def __call__(self, x, t, **_):
                tb = torch.full((x.size(0),), float(t), device=x.device, dtype=x.dtype)
                return mdl(tb, x, cond=cond) if cond is not None else mdl(tb, x)
        return VF()

    def _ode_solve_with_model(x_init, tau_start, tau_end, cond):
        t0, t1 = 1.0 - float(tau_start), 1.0 - float(tau_end)
        solver = ODESolver(velocity_model=_FMVF(model, cond))
        T = torch.tensor([t0, t1], device=x_init.device, dtype=x_init.dtype)
        dt = abs(t1 - t0)
        step_size = min(ode_step_cap, max(1e-3, dt))
        x_end = solver.sample(time_grid=T, x_init=x_init, method=ode_method,
                              step_size=step_size, return_intermediates=False, enable_grad=False)
        return _clamp_box_circles(x_end, L=L)

    def _active_overlap_centers(xyr, for_stop=False):
        P = xyr[:, :2, :].permute(0,2,1).contiguous()  # (B,N,2)
        if not for_stop and eps_nrm > 0.0:
            P = P + (eps_nrm) * torch.randn_like(P)
        D = torch.cdist(P, P).clamp_min(1e-12)
        r = xyr[:, 2, :]
        S = r[:, :, None] + r[:, None, :]
        overlap = (S - D)
        B, N, _ = overlap.shape
        eye = torch.eye(N, device=xyr.device, dtype=torch.bool)[None]
        overlap = overlap.masked_fill(eye, 0.0)
        diff = P[:, :, None, :] - P[:, None, :, :]
        n = diff / D.unsqueeze(-1)
        return overlap, n, eye

    def _select_active(overlap, eye, q=1.0):
        pos = (overlap > 0) & (~eye)
        if q >= 1.0: return pos
        flat = overlap.clone(); flat[~pos] = float('-inf')
        thr = torch.quantile(flat.view(flat.size(0), -1), 1.0 - q, dim=1, keepdim=True)
        return pos & (overlap >= thr.view(-1,1,1))

    def _JJt_inv_h_times_Jt(overlap, n, eye, q):
        active = _select_active(overlap, eye, q)
        if not bool(active.any()):
            B, N, _ = overlap.shape
            return torch.zeros((B, 3, N), device=overlap.device, dtype=overlap.dtype)
        w = (0.5 * overlap * active.float())
        term_i = (w.unsqueeze(-1) * n).sum(dim=2)              # (B,N,2)
        term_j = (w.transpose(1,2).unsqueeze(-1) * n.transpose(1,2)).sum(dim=2)
        delta_xy = (term_i - term_j)                           # (B,N,2)
        deg = active.float().sum(dim=2, keepdim=True).clamp_min(1.0)
        delta_xy = (delta_xy / deg)
        B, N, _ = delta_xy.shape
        out = torch.zeros(B, 3, N, device=overlap.device, dtype=overlap.dtype)
        out[:, 0, :] = delta_xy[...,0]
        out[:, 1, :] = delta_xy[...,1]
        return out

    def _wall_push(xyr, margin_frac=0.05, scale=1.0):
        xy = xyr[:, :2, :]
        r  = xyr[:, 2, :]
        thr = margin_frac * 2.0 * r  # per-circle margin
        delta = torch.zeros_like(xyr)
        x = xy[:, 0, :]; y = xy[:, 1, :]
        dl = x - r; dh = (L - r) - x
        db = y - r; dt = (L - r) - y
        delta[:, 0, :] += torch.where(dl < thr, (thr - dl), torch.zeros_like(dl))
        delta[:, 0, :] -= torch.where(dh < thr, (thr - dh), torch.zeros_like(dh))
        delta[:, 1, :] += torch.where(db < thr, (thr - db), torch.zeros_like(db))
        delta[:, 1, :] -= torch.where(dt < thr, (thr - dt), torch.zeros_like(dt))
        return scale * delta

    def _project_centers(xyr):
        u = xyr
        for _ in range(proj_iters):
            overlap, n, eye = _active_overlap_centers(u, for_stop=False)
            has_pairs = bool((overlap > 0).any())
            if not has_pairs and wall_weight <= 0.0:
                break
            delta_pairs = _JJt_inv_h_times_Jt(overlap, n, eye, contact_q) if has_pairs else 0.0
            delta_walls = _wall_push(u, margin_frac=wall_margin, scale=wall_weight) if wall_weight > 0.0 else 0.0
            u = _clamp_box_circles(u + alpha_proj * (delta_pairs + delta_walls), L=L)
        return u

    def _prox_relaxed(u, u0, u_proj, tau_prime, cond):
        u_hat = (1.0 - tau_prime) * u0 + tau_prime * u_proj
        t_prime = 1.0 - float(tau_prime)
        for _ in range(max(1, prox_iters)):
            tb = torch.full((u.size(0),), t_prime, device=u.device, dtype=u.dtype)
            v = model(tb, u, cond=cond) if cond is not None else model(tb, u)
            u_next = _clamp_box_circles(u + (1.0 - tau_prime) * v, L=L)

            overlap, n, eye = _active_overlap_centers(u_next, for_stop=False)
            active = (overlap > 0) & (~eye)
            w = overlap * active.float()
            term_i = (-(w.unsqueeze(-1) * n).sum(dim=2))
            term_j = (+(w.transpose(1,2).unsqueeze(-1) * n.transpose(1,2)).sum(dim=2))
            jt_pairs_xy = (term_i + term_j).permute(0,2,1)

            jt_walls = _wall_push(u_next, margin_frac=wall_margin, scale=1.0)[:, :2, :]
            jt = torch.zeros_like(u_next)
            jt[:, :2, :] = jt_pairs_xy + jt_walls

            grad = (u - u_hat) + prox_lambda * jt
            u = _clamp_box_circles(u - prox_step * grad, L=L)
        return u

    def _final_polish(u):
        for _ in range(final_passes):
            overlap, _, _ = _active_overlap_centers(u, for_stop=True)
            if float(overlap.clamp_min(0.0).amax()) <= tol_finish:
                break
            u = _project_centers(u)
        return u

    model.eval()
    try:
        optimizer.eval()
    except Exception:
        pass

    samples, remaining = [], int(num_samples)
    cond_iter = iter(cond_loader) if cond_loader is not None else None

    while remaining > 0:
        bs = min(batch_size, remaining)
        # optional cond
        if cond_iter is not None:
            try:
                x_dummy, cond = next(cond_iter)
            except StopIteration:
                cond_iter = iter(cond_loader)
                x_dummy, cond = next(cond_iter)
            cond = cond.to(device)
            if cond.size(0) > bs: cond = cond[:bs]
            if de_novo_minsep_bump > 0.0 and cond.size(1) > cond_minsep_index:
                cond = cond.clone()
                cond[:, cond_minsep_index] = (cond[:, cond_minsep_index] + float(de_novo_minsep_bump)).clamp(max=1.0)
        else:
            cond = None

        # init: random centers in [0,L], tiny radii
        u0 = torch.empty(bs, 3, num_points, device=device)
        u0[:, :2, :] = torch.rand(bs, 2, num_points, device=device) * L
        r0 = (max(1e-6, float(r_min_init)) + float(r_max_init_frac) * 0.5 * torch.rand(bs, num_points, device=device))
        u0[:, 2, :]  = r0
        u0 = _clamp_box_circles(u0, L=L)
        u = u0.clone()

        for k in range(max(1, gas_steps)):
            tau, tau_next = k / max(1, gas_steps), (k+1) / max(1, gas_steps)
            u1 = _ode_solve_with_model(u, tau, tau_next, cond)
            u_proj = _project_centers(u1)
            # LP: maximize sum radii given current centers (respect walls)
            new_r = _lp_project_radii_max_sum(u_proj, L=L)
            u_proj = u_proj.clone()
            u_proj[:, 2, :] = torch.minimum(new_r, _wall_upper_bounds_xy(u_proj[:, :2, :], L=L))
            u = _prox_relaxed(u, u0, u_proj, tau_next, cond)

        u = _final_polish(u)
        r_final = _lp_project_radii_max_sum(u, L=L)
        u[:, 2, :] = torch.minimum(r_final, _wall_upper_bounds_xy(u[:, :2, :], L=L))
        # Diagnostics: show sum of radii for this batch
        batch_sumr = u[:, 2, :].sum(dim=1)
        print(f"[GAS] batch sum_r: mean={batch_sumr.mean().item():.6f} max={batch_sumr.max().item():.6f}")
        samples.append(u.cpu().numpy())
        remaining -= bs

    return np.concatenate(samples, axis=0)

# =============================
# Validation utilities + CSV
# =============================
def _min_wall_clearance_batch(xy: np.ndarray, r: np.ndarray, L: float = 1.0) -> np.ndarray:
    x, y = xy[:, 0, :], xy[:, 1, :]
    w = np.minimum.reduce([x - r, L - x - r, y - r, L - y - r])
    return w.min(axis=1)

def _min_pair_clearance_batch(xy: np.ndarray, r: np.ndarray) -> np.ndarray:
    B, _, N = xy.shape
    out = np.full((B,), np.inf, dtype=np.float64)
    P = np.transpose(xy, (0, 2, 1))
    for b in range(B):
        dx = P[b, :, 0][:, None] - P[b, None, :, 0]
        dy = P[b, :, 1][:, None] - P[b, None, :, 1]
        D  = np.sqrt(dx*dx + dy*dy)
        S  = r[b][:, None] + r[b][None, :]
        np.fill_diagonal(D, np.inf)
        out[b] = np.min(D - S)
    return out

def validate_circle_samples(samples: np.ndarray, tol: float = 1e-9, return_details: bool = True, L: float = 1.0):
    assert samples.ndim == 3 and samples.shape[1] == 3, "Expected (M,3,N)"
    M, _, N = samples.shape
    xy = samples[:, :2, :]
    r  = samples[:, 2, :]

    sum_r = r.sum(axis=1)
    min_wall = _min_wall_clearance_batch(xy, r, L=L)
    min_pair = _min_pair_clearance_batch(xy, r)
    min_clear = np.minimum(min_wall, min_pair)
    feasible = (min_clear >= -tol)

    out = {
        "num_samples": M,
        "num_circles": N,
        "sum_r": sum_r,
        "min_wall_clear": min_wall,
        "min_pair_clear": min_pair,
        "min_clear": min_clear,
        "feasible": feasible
    }
    if return_details:
        out["violations_any"] = (~feasible)
    return out

import csv

def save_validation_csv(metrics, csv_out_path, prefix="fm_circle_generated"):
    """
    Save validation metrics to a stamped CSV file inside the directory
    indicated by `csv_out_path` (if it's a directory) or the directory
    part of `csv_out_path` (if it's a file path).
    """
    # Determine sample count (required)
    if "num_samples" in metrics and isinstance(metrics["num_samples"], (int, np.integer)):
        M = int(metrics["num_samples"])
    else:
        M = None
        for v in metrics.values():
            try:
                arr = np.asarray(v)
                if arr.ndim == 1:
                    M = int(arr.shape[0])
                    break
            except Exception:
                pass
        if M is None:
            raise ValueError("Could not determine num_samples from metrics.")

    # Decide output directory
    is_csv_path = str(csv_out_path).lower().endswith(".csv")
    out_dir = os.path.dirname(csv_out_path) if is_csv_path else csv_out_path
    os.makedirs(out_dir, exist_ok=True)

    # Build stamped filename
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"{prefix}_{M}_{stamp}.csv"
    out_path = os.path.join(out_dir, filename)

    # Identify per-sample columns (1-D arrays of length M)
    per_sample_cols = []
    for k, v in metrics.items():
        try:
            arr = np.asarray(v)
            if arr.ndim == 1 and arr.shape[0] == M:
                per_sample_cols.append(k)
        except Exception:
            continue

    preferred = ["index", "feasible", "sum_r", "min_wall_clear", "min_pair_clear", "min_clear"]
    ordered_cols = [c for c in preferred if c in per_sample_cols]
    ordered_cols += [c for c in per_sample_cols if c not in ordered_cols]

    synth_index = None
    if "index" not in ordered_cols:
        synth_index = np.arange(M, dtype=int)
        ordered_cols.insert(0, "index")

    with open(out_path, "w", newline="") as f:
        scalars = {k: v for k, v in metrics.items()
                   if not (hasattr(v, "__len__") and np.asarray(v).ndim == 1 and np.asarray(v).shape[0] == M)}
        for k, v in sorted(scalars.items()):
            f.write(f"# {k}: {v}\n")

        writer = csv.DictWriter(f, fieldnames=ordered_cols)
        writer.writeheader()

        arrays = {k: np.asarray(metrics[k]) for k in per_sample_cols}
        if synth_index is not None:
            arrays["index"] = synth_index

        for i in range(M):
            row = {col: arrays[col][i].item() if isinstance(arrays[col][i], np.generic) else arrays[col][i]
                   for col in ordered_cols}
            writer.writerow(row)

    return out_path

# =============================
# Plotters
# =============================
def _detect_infeasible_indices(xy: np.ndarray, r: np.ndarray, tol: float = 1e-9, L: float = 1.0):
    N = r.size
    bad = set()
    w1 = xy[0, :] - r < -tol
    w2 = L - xy[0, :] - r < -tol
    w3 = xy[1, :] - r < -tol
    w4 = L - xy[1, :] - r < -tol
    for i in np.where(w1 | w2 | w3 | w4)[0]:
        bad.add(int(i))
    dx = xy[0, :][:, None] - xy[0, :][None, :]
    dy = xy[1, :][:, None] - xy[1, :][None, :]
    D  = np.sqrt(dx*dx + dy*dy)
    S  = r[:, None] + r[None, :]
    viol = (D - S) < -tol
    np.fill_diagonal(viol, False)
    ii, jj = np.where(viol)
    for i, j in zip(ii, jj):
        bad.add(int(i)); bad.add(int(j))
    return bad

def plot_circle_sample(sample_3xN: np.ndarray, ax: plt.Axes = None,
                       title: str = None, mark_infeasible: bool = True, tol: float = 1e-9, lw: float = 1.5, L: float = 1.0):
    assert sample_3xN.shape[0] == 3
    xy = sample_3xN[:2, :]
    r  = sample_3xN[2, :]
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))
    ax.plot([0,L,L,0,0], [0,0,L,L,0], linewidth=1.5)
    bad = _detect_infeasible_indices(xy, r, tol, L=L) if mark_infeasible else set()
    for i in range(xy.shape[1]):
        c = Circle((xy[0, i], xy[1, i]), r[i],
                   fill=False,
                   linewidth=lw,
                   edgecolor=("red" if i in bad else "black"))
        ax.add_patch(c)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, L); ax.set_ylim(0, L)
    if title is None:
        title = f"N={xy.shape[1]}, sum_r={np.sum(r):.4f}, feas={len(bad)==0}"
    ax.set_title(title)
    return ax

def plot_first_k_samples(samples: np.ndarray, k: int, out_dir: str,
                         filename_prefix: str = "gen_circle", mark_infeasible: bool = True, tol: float = 1e-9, L: float = 1.0):
    os.makedirs(out_dir, exist_ok=True)
    M = samples.shape[0]
    k = min(k, M)
    for s in range(k):
        xy = samples[s, :2, :]
        r  = samples[s, 2, :]
        sum_r = float(np.sum(r))
        feas = len(_detect_infeasible_indices(xy, r, tol, L=L)) == 0
        title = f"Sample {s} | N={xy.shape[1]} | sum_r={sum_r:.4f} | feas={feas}"
        fig, ax = plt.subplots(figsize=(5,5))
        plot_circle_sample(samples[s], ax=ax, title=title, mark_infeasible=mark_infeasible, tol=tol, L=L)
        fn = f"{filename_prefix}_N{xy.shape[1]}_sumr{sum_r:.4f}_idx{s}.png"
        fig.savefig(os.path.join(out_dir, fn), dpi=150, bbox_inches="tight")
        plt.close(fig)

# =============================
# Config helper
# =============================
def _get_cfg(sec, key, fallback):
    if HAS_CFG:
        try:
            if isinstance(fallback, int):    return cfg.getint(sec, key, fallback=fallback)
            if isinstance(fallback, float):  return cfg.getfloat(sec, key, fallback=fallback)
            if isinstance(fallback, bool):   return cfg.getboolean(sec, key, fallback=fallback)
            val = cfg.get(sec, key, fallback=fallback)
            return val
        except Exception:
            return fallback
    return fallback

# =============================
# Main
# =============================
if __name__ == "__main__":
    # Section name in your INI (feel free to rename)
    SEC = "flow_matching_sumradii"

    # ---------------- I/O + training args ----------------
    dataset_path   = _get_cfg(SEC, "dataset_path", "./outputs_circle_packing/circle_srp_generated.pt")
    save_model_dir = _get_cfg(SEC, "save_model_path", "./fm_circle_models")
    save_gen_dir   = _get_cfg(SEC, "save_generated_path", "./fm_circle_generated")
    plot_out_dir   = _get_cfg(SEC, "plot_out_dir", "./fm_circle_plots")
    csv_out_path   = _get_cfg(SEC, "csv_out_path", "./fm_circle_metrics")

    dim            = 3
    batch_size     = _get_cfg(SEC, "batch_size", 64)
    learning_rate  = _get_cfg(SEC, "learning_rate", 2e-4)
    eta_min        = _get_cfg(SEC, "eta_min", 2e-5)  # not used by schedulefree, kept for completeness
    num_epochs     = _get_cfg(SEC, "num_epochs", 200)

    mse_strength   = _get_cfg(SEC, "mse_strength", 1.0)
    pen_strength   = _get_cfg(SEC, "distance_penalty_strength", 0.5)

    num_new        = _get_cfg(SEC, "sample_new_points", 1000)
    batch_new      = _get_cfg(SEC, "sample_new_points_batch_size", 50)
    N_points       = _get_cfg(SEC, "num_circles", 50)
    mode           = _get_cfg(SEC, "mode", "train_and_sample")  # train_and_sample | sampling_only | train_only

    # Box
    box_len        = _get_cfg(SEC, "box_len", 1.0)

    # ---------------- Data ----------------
    full_ds = CirclePackingDataset(dataset_path)
    max_samples = len(full_ds)
    test_fraction= _get_cfg(SEC, "test_fraction", 0.1)
    split_seed   = _get_cfg(SEC, "split_seed", 1234)
    gen = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(max_samples, generator=gen)
    test_size = max(1, int(round(max_samples * test_fraction)))
    test_idx = perm[:test_size].tolist()
    train_idx = perm[test_size:].tolist()
    train_ds = Subset(full_ds, train_idx)
    test_ds  = Subset(full_ds, test_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_new, shuffle=False)
    print(f"Train size: {len(train_ds)} | Test size: {len(test_ds)}")

    # ---------------- Model ----------------
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    st_kwargs = {
        'dim_hidden': _get_cfg(SEC, 'st_dim_hidden', 512),
        'num_heads':  _get_cfg(SEC, 'st_num_heads', 8),
        'depth':      _get_cfg(SEC, 'st_depth', 6),
        'attn_dropout': _get_cfg(SEC, 'st_attn_dropout', 0.1),
        'ff_dropout':   _get_cfg(SEC, 'st_ff_dropout', 0.1),
        'dim_time':     _get_cfg(SEC, 'st_dim_time', 128),
        'time_fourier_dim': _get_cfg(SEC, 'time_fourier_dim', 64),
        'time_hidden':      _get_cfg(SEC, 'st_time_hidden', _get_cfg(SEC, 'time_hidden', 256)),
        'time_fourier_sigma': _get_cfg(SEC, 'time_fourier_sigma', 1.0),
        'cond_dim_in': 4,
        'cond_hidden': _get_cfg(SEC, 'cond_hidden', 128),
    }

    if mode in ["sample_only", "sampling_only"]:
        ckpt_path = _get_cfg(SEC, "load_model_path", None)
        assert ckpt_path and os.path.isfile(ckpt_path), "Provide a valid load_model_path for sampling_only mode."
        checkpoint = torch.load(ckpt_path, map_location=dev)
        model = _build_model_from_ckpt(checkpoint, dev, st_kwargs)
        try:
            model.load_state_dict(checkpoint["model"], strict=True)
            print(f"[FM] Loaded model (strict=True) from {ckpt_path}")
        except Exception as e:
            print(f"[FM] Strict load failed ({e}); trying strict=False (only safe if shapes match).")
            model.load_state_dict(checkpoint["model"], strict=False)
            print(f"[FM] Loaded model (strict=False) from {ckpt_path}")
    else:
        model = FlowSetTransformer(3, **st_kwargs).to(dev)

    opt = schedulefree.RAdamScheduleFree(model.parameters(), lr=learning_rate)

    # ---------------- Train ----------------
    if mode in ["train_and_sample", "train_only"]:
        model, hist = train_flow_model(
            model, opt, train_loader, num_epochs,
            mse_strength, pen_strength,
            dev, st_kwargs, save_model_dir
        )

    # ---------------- Sample ----------------
    if mode in ["train_only"]:
        print("Training only mode selected; skipping sampling.")
        raise SystemExit(0)

    # GAS knobs from config
    gas_steps   = _get_cfg(SEC, "gas_steps", 40)
    ode_method_raw = _get_cfg(SEC, "ode_method", "midpoint")
    ode_method   = _sanitize_ode_method(ode_method_raw)
    ode_step_cap = _get_cfg(SEC, "ode_step_cap", 0.05)

    proj_iters   = _get_cfg(SEC, "proj_iters", 6)
    alpha_proj   = _get_cfg(SEC, "alpha_proj", 0.25)
    contact_q    = _get_cfg(SEC, "contact_q", 1.0)

    wall_weight  = _get_cfg(SEC, "wall_weight", 1.0)
    wall_margin  = _get_cfg(SEC, "wall_margin", 0.05)

    prox_iters   = _get_cfg(SEC, "prox_iters", 8)
    prox_step    = _get_cfg(SEC, "prox_step", 0.1)
    prox_lambda  = _get_cfg(SEC, "prox_lambda", 1.5)

    final_passes = _get_cfg(SEC, "final_passes", 4)
    tol_finish   = _get_cfg(SEC, "tol_finish", 1e-8)

    r_min_init   = _get_cfg(SEC, "r_min_init", 1e-4)
    r_max_init_frac = _get_cfg(SEC, "r_max_init_frac", 0.25)

    de_novo_minsep_bump = _get_cfg(SEC, "de_novo_minsep_bump", 0.0)
    cond_minsep_index   = _get_cfg(SEC, "cond_minsep_index", 3)

    samples = sample_flow_model(
        model, opt, num_new, batch_new, N_points, dev,
        clip_min=0.0, clip_max=float(box_len), dim=dim,
        cond_loader=test_loader,
        L=float(box_len),
        gas_steps=int(gas_steps),
        ode_method=str(ode_method),
        ode_step_cap=float(ode_step_cap),
        proj_iters=int(proj_iters),
        alpha_proj=float(alpha_proj),
        contact_q=float(contact_q),
        wall_weight=float(wall_weight),
        wall_margin=float(wall_margin),
        prox_iters=int(prox_iters),
        prox_step=float(prox_step),
        prox_lambda=float(prox_lambda),
        final_passes=int(final_passes),
        tol_finish=float(tol_finish),
        r_min_init=float(r_min_init),
        r_max_init_frac=float(r_max_init_frac),
        de_novo_minsep_bump=float(de_novo_minsep_bump),
        cond_minsep_index=int(cond_minsep_index)
    )

    os.makedirs(save_gen_dir, exist_ok=True)
    out_path = os.path.join(save_gen_dir, f"flow_circles_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
    torch.save(torch.from_numpy(samples), out_path)
    print(f"Saved {samples.shape[0]} samples to {out_path}")

    # ---------------- Validate + CSV ----------------
    os.makedirs(csv_out_path if not csv_out_path.lower().endswith(".csv") else os.path.dirname(csv_out_path), exist_ok=True)
    metrics = validate_circle_samples(samples, tol=1e-9, L=float(box_len))
    save_validation_csv(metrics, csv_out_path, prefix="fm_circle_generated")
    print(f"Validation CSV written under: {csv_out_path}")
    print(f"Feasible: {metrics['feasible'].sum()}/{metrics['num_samples']} | "
          f"best sum_r={metrics['sum_r'].max():.6f}")

    # ---------------- Plots ----------------
    plot_first_k_samples(samples, k=min(20, samples.shape[0]), out_dir=plot_out_dir,
                         filename_prefix="circle_packing_gen", mark_infeasible=True, tol=1e-9, L=float(box_len))
    print(f"Saved plots to {plot_out_dir}")