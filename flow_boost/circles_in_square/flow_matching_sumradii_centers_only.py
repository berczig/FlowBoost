# -*- coding: utf-8 -*-
"""
Flow-Matching for circle-centres only (x,y) in unit square.
Input format: (M,3,N) with rows [x, y, r] (r may be present but ignored by the model).
Output: (M,3,N) with r = 0 for each circle.
Includes training + sampling.

Pipeline additions (minimal):
- _set_cfg helper (safe no-op if cfg missing)
- train_flow_model_centers returns checkpoint path
- main(state=...) entrypoint sets:
    state.model_path   (checkpoint)
    state.samples_path (generated samples .pt)
and supports cfg mode:
    training_and_sampling / retrain_and_sampling / sampling_only / train_only
"""

import os, math, time, re
import numpy as np
from datetime import datetime
from typing import Dict, Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import tqdm

# optional config library
try:
    from flow_boost import cfg
    HAS_CFG = True
    print("Loaded configuration file", getattr(cfg, "cfg_path", "config.cfg") + "!")
except Exception:
    HAS_CFG = False

# Flow-Matching imports (adjust to your setup)
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver

FM_PATH = AffineProbPath(scheduler=CondOTScheduler())

# ---------------- Model (centres only) ----------------
class FlowSetTransformerCenters(nn.Module):
    class TimeEmbedFourier(nn.Module):
        def __init__(self, out_dim: int, fourier_dim: int, hidden: int, sigma: float = 1.0, include_poly: bool = True):
            super().__init__()
            fourier_dim = max(16, fourier_dim)
            self.register_buffer("freqs", torch.randn(fourier_dim) * sigma, persistent=False)
            self.include_poly = include_poly
            in_dim = 2 * fourier_dim
            if include_poly:
                in_dim += 4
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, out_dim),
            )

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            # expect t shape (B,) or (B,1)
            if t.ndim == 2:
                assert t.shape[1] == 1
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

    def __init__(self, d: int = 2, **st_kwargs):
        super().__init__()
        self.d = d
        model_dim = int(st_kwargs.get("dim_hidden", 512))
        heads = int(st_kwargs.get("num_heads", 8))
        depth = int(st_kwargs.get("depth", 6))
        attn_do = float(st_kwargs.get("attn_dropout", 0.1))
        ff_do = float(st_kwargs.get("ff_dropout", 0.1))
        dim_time = int(st_kwargs.get("dim_time", max(64, 4 * d)))
        time_F = int(st_kwargs.get("time_fourier_dim", max(16, dim_time // 2)))
        time_hid = int(st_kwargs.get("time_hidden", 2 * dim_time))
        sigma = float(st_kwargs.get("time_fourier_sigma", 1.0))

        cond_dim_in = int(st_kwargs.get("cond_dim_in", 4))
        cond_hidden = int(st_kwargs.get("cond_hidden", max(64, dim_time)))
        self.uses_cond = cond_dim_in > 0

        self.time_emb = self.TimeEmbedFourier(
            out_dim=dim_time,
            fourier_dim=time_F,
            hidden=time_hid,
            sigma=sigma,
            include_poly=True,
        )

        if self.uses_cond:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim_in, cond_hidden),
                nn.SiLU(),
                nn.Linear(cond_hidden, cond_hidden),
            )
            dim_cond = cond_hidden
        else:
            self.cond_mlp = None
            dim_cond = 0

        self.token_in = nn.Linear(d + dim_time + dim_cond, model_dim)
        self.film_in_dim = dim_time + dim_cond
        self.timecond_to_film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.film_in_dim, 2 * model_dim),
        )

        from x_transformers import Encoder

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
            attn_flash=True,
        )

        self.token_out = nn.Linear(model_dim, d)
        nn.init.zeros_(self.token_out.weight)
        nn.init.zeros_(self.token_out.bias)

    def forward(self, t: torch.Tensor, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        B, d_, N = x.shape
        assert d_ == self.d
        device = x.device

        # Normalize t to shape (B,)
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, dtype=x.dtype, device=device)
        else:
            t = t.to(device=device, dtype=x.dtype)
        if t.ndim == 0:
            t = t.repeat(B)
        elif t.ndim == 1 and t.shape[0] != B:
            t = t.reshape(-1)[0].repeat(B)
        elif t.ndim == 2:
            if t.shape == (B, 1):
                t = t[:, 0]
            elif t.numel() == 1:
                t = t.reshape(-1)[0].repeat(B)
            else:
                t = t[:, 0]

        t_embed = self.time_emb(t)

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
        beta = beta[:, None, :]
        h = h * (1 + gamma) + beta

        h = self.encoder(h)
        out = self.token_out(h).permute(0, 2, 1).contiguous()
        return out


# ---------------- Dataset wrapper ----------------
class CircleCentersDataset(Dataset):
    def __init__(self, path, scale_N=128):
        data = torch.load(path)  # (M,3,N)
        assert data.ndim == 3 and data.size(1) == 3, f"expected (M,3,N); got {tuple(data.shape)}"
        self.data = data.contiguous().float()
        self.M, self.d, self.N = self.data.shape

        xy = self.data[:, :2, :]  # (M,2,N)
        r = self.data[:, 2, :]    # (M,N)
        sum_r = r.sum(dim=1)

        wall_ub = torch.minimum(
            torch.minimum(xy[:, 0, :], 1 - xy[:, 0, :]),
            torch.minimum(xy[:, 1, :], 1 - xy[:, 1, :]),
        )
        min_wall = (wall_ub - r).amin(dim=1)

        P = xy.permute(0, 2, 1).contiguous()
        D = torch.cdist(P, P)
        S = r[:, :, None] + r[:, None, :]
        eye = torch.eye(self.N, device=D.device, dtype=torch.bool)[None]
        D = D.masked_fill(eye, float("inf"))
        min_pair = (D - S).amin(dim=-1).amin(dim=-1)

        N_scaled = torch.full((self.M,), float(self.N) / float(scale_N), dtype=self.data.dtype)
        self.cond = torch.stack([N_scaled, sum_r, min_pair, min_wall], dim=1)
        print(f"Loaded {path}, shape {tuple(self.data.shape)} | cond {tuple(self.cond.shape)}")

    def __len__(self):
        return self.M

    def __getitem__(self, idx):
        xyr = self.data[idx]          # (3,N)
        xy = xyr[:2, :]               # (2,N)
        r = xyr[2, :]                 # (N,) – present but not used by model
        cond = self.cond[idx]
        return (xy, r), cond


# ---------------- Helper functions ----------------
def _get_cfg(sec, key, fallback):
    if HAS_CFG:
        try:
            if isinstance(fallback, int):
                return cfg.getint(sec, key, fallback=fallback)
            if isinstance(fallback, float):
                return cfg.getfloat(sec, key, fallback=fallback)
            if isinstance(fallback, bool):
                return cfg.getboolean(sec, key, fallback=fallback)
            return cfg.get(sec, key, fallback=fallback)
        except Exception:
            return fallback
    return fallback

def _set_cfg(sec, key, value):
    """
    Minimal setter used by pipeline scripts.
    Safe no-op if cfg is unavailable.
    """
    if HAS_CFG:
        try:
            if not cfg.has_section(sec):
                cfg.add_section(sec)
            cfg.set(sec, key, str(value))
        except Exception:
            pass


def _ode_solve_centers(model, xy_init, tau, tau_next, cond, L=1.0):
    """
    Solve ODE on centres only.
    xy_init shape: (B,2,N)
    Return next centres shape: (B,2,N)
    """
    solver = ODESolver(velocity_model=lambda x, t: model(t, x, cond=cond))
    T = torch.tensor([1.0 - tau, 1.0 - tau_next], device=xy_init.device, dtype=xy_init.dtype)
    dt = abs(T[1] - T[0])
    step_size = min(0.05, max(1e-3, float(dt)))
    xy_end = solver.sample(
        time_grid=T,
        x_init=xy_init,
        method="midpoint",
        step_size=step_size,
        return_intermediates=False,
        enable_grad=False,
    )
    return xy_end

def _project_centers_simple(xy, L=1.0):
    """Simple bound clamp of centres."""
    return torch.clamp(xy, min=0.0, max=L)

def _prox_relaxed_centers_only(u0, u_proj, tau_prime, cond):
    """Proximal relaxation update for centres only."""
    return (1.0 - tau_prime) * u0 + tau_prime * u_proj

def _final_polish_centers_only(xy, L=1.0, passes=4):
    """Final polish on centres: clamp etc."""
    return torch.clamp(xy, min=0.0, max=L)


# ---------------- Training function ----------------
def train_flow_model_centers(
    model,
    optimizer,
    loader,
    num_epochs,
    mse_strength,
    geom_strength,
    device,
    params,
    save_path,
):
    """
    Returns: (model, history_np, checkpoint_path)
    """
    model.train().to(device)
    mse = nn.MSELoss()
    history = []

    for epoch in range(num_epochs):
        epoch_losses = []
        ratio = 0.5 * (1 - math.cos(math.pi * min(1.0, epoch / (0.5 * num_epochs))))

        for (xy0, r0), cond in loader:
            xy0 = xy0.to(device)
            cond = cond.to(device)
            B, d_, N = xy0.shape  # d_ is 2 here

            # random target centres
            xy1 = torch.rand(B, d_, N, device=device)

            # Sample t_in shape (B,)
            t_in = torch.rand(B, device=device)

            # Path sample
            path_sample = FM_PATH.sample(t=t_in, x_0=xy0, x_1=xy1)
            x_t = path_sample.x_t     # (B,2,N)
            dx_t = path_sample.dx_t   # (B,2,N)

            u_pred = model(t_in, x_t, cond=cond)
            loss_fm = mse(u_pred, dx_t)

            # geometric penalties on centres
            P = x_t.permute(0, 2, 1).contiguous()
            D = torch.cdist(P, P).clamp_min(1e-6)
            eye = torch.eye(N, device=device, dtype=torch.bool)[None]
            D = D.masked_fill(eye, float("inf"))
            min_pair = D.amin(dim=-1).amin(dim=-1)

            x = x_t[:, 0, :]
            y = x_t[:, 1, :]
            wall_ub = torch.minimum(torch.minimum(x, 1 - x), torch.minimum(y, 1 - y))
            min_wall = wall_ub.amin(dim=1)

            pen_pairs = (F.relu(0.01 - min_pair)).mean()
            pen_walls = (F.relu(0.01 - min_wall)).mean()

            loss = loss_fm * mse_strength + ratio * geom_strength * (pen_pairs + pen_walls)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append([loss_fm.item(), (pen_pairs + pen_walls).item(), loss.item()])

        avg = np.mean(epoch_losses, axis=0)
        history.append(avg)
        print(f"Epoch {epoch+1}/{num_epochs} | FM={avg[0]:.4e} Geom={avg[1]:.4e} Tot={avg[2]:.4e}")

    history = np.array(history)
    os.makedirs(save_path, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(save_path, f"flow_centers_only_{ts}.pth")

    torch.save(
        {
            "model": model.state_dict(),
            "params": {
                "st_kwargs": params,
                "dim": int(getattr(model, "d", 2)),
            },
            "history": history,
        },
        out_file,
    )

    # plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(history[:, 0], label="FM loss")
    plt.plot(history[:, 1], label="Geom penalty")
    plt.plot(history[:, 2], label="Total loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, f"losses_{ts}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return model, history, out_file


# ---------------- Sampling function ----------------
@torch.no_grad()
def sample_flow_model_centers_only(
    model,
    optimizer,
    num_samples,
    batch_size,
    num_points,
    device,
    L=1.0,
    gas_steps=40,
    ode_method="midpoint",
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
):
    model.eval()
    samples = []
    remaining = int(num_samples)

    while remaining > 0:
        bs = min(int(batch_size), remaining)
        xy0 = torch.rand(bs, 2, num_points, device=device) * L
        r0 = torch.zeros(bs, num_points, device=device)
        u0 = torch.cat([xy0, r0.unsqueeze(1)], dim=1)  # (B,3,N)
        u = u0.clone()

        for k in range(max(1, int(gas_steps))):
            tau = k / float(gas_steps)
            tau_next = (k + 1) / float(gas_steps)

            xy1 = _ode_solve_centers(model, u[:, :2, :].clone(), tau, tau_next, None, L=L)
            xy_proj = _project_centers_simple(xy1, L=L)
            u_proj = torch.cat([xy_proj, torch.zeros(bs, 1, num_points, device=device)], dim=1)

            u = _prox_relaxed_centers_only(u0, u_proj, tau_next, None)

        xy_final = _final_polish_centers_only(u[:, :2, :], L=L)
        out = torch.zeros(bs, 3, num_points, device=device)
        out[:, :2, :] = xy_final
        # Third row stays zero (radii = 0)

        samples.append(out.cpu().numpy())
        remaining -= bs

    return np.concatenate(samples, axis=0)


# ---------------- Validator & Plotters ----------------
def _min_wall_clearance_batch(xy: np.ndarray, r: np.ndarray, L: float = 1.0) -> np.ndarray:
    x = xy[:, 0, :]
    y = xy[:, 1, :]
    w = np.minimum.reduce([x - r, L - x - r, y - r, L - y - r])
    return w.min(axis=1)

def _min_pair_clearance_batch(xy: np.ndarray, r: np.ndarray) -> np.ndarray:
    B, _, N = xy.shape
    out = np.full((B,), np.inf, dtype=np.float64)
    P = xy.transpose((0, 2, 1))
    for b in range(B):
        dx = P[b, :, 0][:, None] - P[b, None, :, 0]
        dy = P[b, :, 1][:, None] - P[b, None, :, 1]
        D = np.sqrt(dx * dx + dy * dy)
        S = r[b][:, None] + r[b][None, :]
        np.fill_diagonal(D, np.inf)
        out[b] = np.min(D - S)
    return out

def validate_circle_samples(samples: np.ndarray, tol: float = 1e-9, L: float = 1.0):
    assert samples.ndim == 3 and samples.shape[1] == 3
    M, _, N = samples.shape
    xy = samples[:, :2, :]
    r = samples[:, 2, :]
    sum_r = r.sum(axis=1)
    min_wall = _min_wall_clearance_batch(xy, r, L=L)
    min_pair = _min_pair_clearance_batch(xy, r)
    min_clear = np.minimum(min_wall, min_pair)
    feasible = (min_clear >= -tol)
    return {
        "num_samples": M,
        "num_circles": N,
        "sum_r": sum_r,
        "min_wall_clear": min_wall,
        "min_pair_clear": min_pair,
        "min_clear": min_clear,
        "feasible": feasible,
    }

def plot_circle_sample(
    sample: np.ndarray,
    ax: plt.Axes = None,
    title: str = None,
    mark_infeasible: bool = True,
    tol: float = 1e-9,
    L: float = 1.0,
):
    assert sample.shape[0] == 3
    xy = sample[:2, :]
    r = sample[2, :]
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0], linewidth=1.5)
    for i in range(xy.shape[1]):
        c = Circle((xy[0, i], xy[1, i]), r[i], fill=False, linewidth=1.2, edgecolor="black")
        ax.add_patch(c)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    if title is None:
        title = f"N={xy.shape[1]}, sum_r={float(r.sum()):.4f}"
    ax.set_title(title)
    return ax

def plot_first_k_samples(
    samples: np.ndarray,
    k: int,
    out_dir: str,
    filename_prefix: str = "gen_centers_only",
    mark_infeasible: bool = True,
    tol: float = 1e-9,
    L: float = 1.0,
):
    os.makedirs(out_dir, exist_ok=True)
    M = samples.shape[0]
    k = min(k, M)
    for s in range(k):
        fig, ax = plt.subplots(figsize=(5, 5))
        plot_circle_sample(
            samples[s],
            ax=ax,
            title=f"Sample {s} | N={samples.shape[2]}",
            mark_infeasible=mark_infeasible,
            tol=tol,
            L=L,
        )
        fn = f"{filename_prefix}_{samples.shape[2]}_{s}.png"
        fig.savefig(os.path.join(out_dir, fn), dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------- FM model compatibility (load 3-ch sumradii ckpt) ----------------
def _load_sumradii_model_exact(ckpt: dict, device):
    """
    Load the exact model class used by flow_matching_sumradii.py and return
    a wrapper that accepts (B,2,N) inputs and outputs (B,2,N) (dx,dy).
    """
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if "token_out.weight" not in sd:
        raise RuntimeError("Checkpoint missing 'token_out.weight'; not a compatible FlowSetTransformer.")
    d_out = int(sd["token_out.weight"].shape[0])
    if d_out != 3:
        raise RuntimeError(f"Expected a 3-channel sumradii checkpoint; got d_out={d_out}.")

    try:
        from flow_boost.circles_in_square.flow_matching_sumradii import FlowSetTransformer as SumradiiTransformer
    except Exception as e:
        raise RuntimeError(
            "Failed to import FlowSetTransformer from flow_matching_sumradii.py. "
            "Ensure that file defines the class and is importable."
        ) from e

    if isinstance(ckpt, dict) and "params" in ckpt and isinstance(ckpt["params"], dict):
        params = ckpt["params"]
        d_ckpt = int(params.get("dim", 3))
        st_kwargs = params.get("st_kwargs", {})
    else:
        _, model_dim = sd["token_out.weight"].shape
        st_kwargs = {"dim_hidden": model_dim}
        d_ckpt = 3

    base = SumradiiTransformer(d_ckpt, **st_kwargs).to(device)
    base.load_state_dict(sd, strict=True)

    class XYOnlyAdapter(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model
            self.d = 2
            self._warned = False

        def forward(self, t, x, cond=None):
            B, d_in, N = x.shape

            # Normalize t to shape (B,)
            if not torch.is_tensor(t):
                t = torch.as_tensor(t, dtype=x.dtype, device=x.device)
            else:
                t = t.to(device=x.device, dtype=x.dtype)
            if t.ndim == 0:
                t = t.repeat(B)
            elif t.ndim == 1 and t.shape[0] != B:
                t = t.reshape(-1)[0].repeat(B)
            elif t.ndim == 2:
                if t.shape == (B, 1):
                    t = t[:, 0]
                elif t.numel() == 1:
                    t = t.reshape(-1)[0].repeat(B)
                else:
                    t = t[:, 0]

            d_expected = int(getattr(self.base, "d", d_in))
            if d_in != d_expected:
                if d_in < d_expected:
                    pad = torch.zeros(B, d_expected - d_in, N, device=x.device, dtype=x.dtype)
                    x_in = torch.cat([x, pad], dim=1)
                else:
                    x_in = x[:, :d_expected, :]
                if not self._warned:
                    print(f"[CKPT] Adapting input channels {d_in} -> {d_expected} for loaded model (sumradii).")
                    self._warned = True
            else:
                x_in = x

            v = self.base(t, x_in, cond=cond)  # (B,3,N)
            return v[:, :2, :]                 # keep dx,dy only

    print("[CKPT] Loaded original sumradii model and wrapped to centers-only (dx,dy).")
    return XYOnlyAdapter(base)


def _infer_st_kwargs_from_state_dict(sd: dict):
    """
    Infer SetTransformer kwargs and output dim from a saved state_dict.
    """
    if "token_out.weight" in sd:
        d_out, model_dim = sd["token_out.weight"].shape
        d_inferred = int(d_out)
    else:
        model_dim, _ = sd["token_in.weight"].shape
        d_inferred = 3

    # depth
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

    # heads: infer from to_q weight rows (inner_dim)
    qk_key = None
    for cand in ("encoder.layers.0.1.to_q.weight", "encoder.layers.0.attn.to_q.weight"):
        if cand in sd:
            qk_key = cand
            break

    if qk_key is not None:
        inner_dim, _ = sd[qk_key].shape
        if inner_dim % 64 == 0:
            num_heads = max(1, inner_dim // 64)
        else:
            for h in (12, 8, 6, 4, 2, 1):
                if model_dim % h == 0:
                    num_heads = h
                    break
            else:
                num_heads = 8
    else:
        for h in (8, 6, 4, 2, 1):
            if model_dim % h == 0:
                num_heads = h
                break
        else:
            num_heads = 8

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


def _build_model_from_ckpt_sumradii_compat(FlowSetTransformerCls, checkpoint, device, fallback_kwargs, target_d=2):
    """
    Prefer loading the exact sumradii class if the checkpoint has d_out=3.
    For 2-channel (centers-only) checkpoints, use saved params if present;
    otherwise infer from the state_dict.
    """
    sd = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if "token_out.weight" not in sd:
        raise RuntimeError("Checkpoint missing token_out.weight")
    d_out = int(sd["token_out.weight"].shape[0])

    if d_out == 3:
        return _load_sumradii_model_exact(checkpoint, device)

    # 2-channel -> build FlowSetTransformerCenters
    if isinstance(checkpoint, dict) and "params" in checkpoint and isinstance(checkpoint["params"], dict):
        params = checkpoint["params"]
        d_ckpt = int(params.get("dim", 2))
        st_kw = dict(params.get("st_kwargs", fallback_kwargs or {}))
        base = FlowSetTransformerCls(d_ckpt, **st_kw).to(device)
        try:
            base.load_state_dict(sd, strict=True)
            print(
                f"[CKPT] Loaded centers-only checkpoint via saved params "
                f"(depth={st_kw.get('depth','?')}, heads={st_kw.get('num_heads','?')}, dim={st_kw.get('dim_hidden','?')})."
            )
            return base
        except Exception as e:
            print(f"[CKPT] Strict load with saved params failed: {e}. Falling back to inference...")

    st_kw_inf, d_ckpt_inf = _infer_st_kwargs_from_state_dict(sd)
    base = FlowSetTransformerCls(int(d_ckpt_inf), **st_kw_inf).to(device)
    try:
        base.load_state_dict(sd, strict=True)
        print(
            f"[CKPT] Loaded centers-only checkpoint (inferred) "
            f"depth={st_kw_inf['depth']} heads={st_kw_inf['num_heads']} dim={st_kw_inf['dim_hidden']}."
        )
        return base
    except Exception as e:
        print(f"[CKPT] Strict load (inferred) failed: {e}. Using strict=False as last resort.")
        missing = base.load_state_dict(sd, strict=False)
        print(f"[CKPT] Loaded with strict=False. Missing: {len(missing.missing_keys)} keys, unexpected: {len(missing.unexpected_keys)} keys.")
        return base


# ---------------- Pipeline-friendly main(state=...) ----------------
def main(state=None):
    """
    Pipeline-friendly entrypoint.

    Section: flow_matching_sumradii_centers_only

    Supports cfg mode:
      - training_and_sampling
      - retrain_and_sampling
      - sampling_only
      - train_only

    Reads:
      - dataset_path (required for training modes)
      - resume_model_path (pipeline) or load_model_path (legacy) for retrain/sampling
    Writes (if state provided):
      - state.model_path
      - state.samples_path
    """
    SEC = "flow_matching_sumradii_centers_only"

    mode = str(_get_cfg(SEC, "mode", "training_and_sampling")).strip().lower()
    if mode in ("train_and_sample", "train_and_sampling", "training_and_sampling"):
        mode = "training_and_sampling"
    elif mode in ("retrain_and_sampling", "retrain_and_sample", "retrain"):
        mode = "retrain_and_sampling"
    elif mode in ("sampling_only", "sample_only"):
        mode = "sampling_only"
    elif mode in ("train_only",):
        mode = "train_only"

    dataset_path = _get_cfg(SEC, "dataset_path", "")
    save_model_path = _get_cfg(SEC, "save_model_path", "./fm_centers_only_models")
    save_gen_path = _get_cfg(SEC, "save_generated_path", "./fm_centers_only_generated")
    plot_out_dir = _get_cfg(SEC, "plot_out_dir", "./fm_centers_only_plots")
    csv_out_path = _get_cfg(SEC, "csv_out_path", "./fm_centers_only_metrics")  # unused but kept
    load_model_path = _get_cfg(SEC, "load_model_path", "")
    resume_model_path = _get_cfg(SEC, "resume_model_path", "")
    ckpt_path = resume_model_path or load_model_path

    batch_size = _get_cfg(SEC, "batch_size", 64)
    learning_rate = _get_cfg(SEC, "learning_rate", 2e-4)
    num_epochs = _get_cfg(SEC, "num_epochs", 200)

    mse_strength = _get_cfg(SEC, "mse_strength", 1.0)
    geom_strength = _get_cfg(SEC, "geom_strength", 0.5)

    num_new = _get_cfg(SEC, "sample_new_points", 1000)
    batch_new = _get_cfg(SEC, "sample_new_points_batch_size", 50)
    N_points = _get_cfg(SEC, "num_circles", 50)
    box_len = _get_cfg(SEC, "box_len", 1.0)
    train_top_fraction = _get_cfg(SEC, "train_top_fraction", 0.5)

    test_fraction = _get_cfg(SEC, "test_fraction", 0.1)
    split_seed = _get_cfg(SEC, "split_seed", 1234)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    st_kwargs = {
        "dim_hidden": _get_cfg(SEC, "st_dim_hidden", 128),
        "num_heads": _get_cfg(SEC, "st_num_heads", 8),
        "depth": _get_cfg(SEC, "st_depth", 6),
        "attn_dropout": _get_cfg(SEC, "st_attn_dropout", 0.1),
        "ff_dropout": _get_cfg(SEC, "st_ff_dropout", 0.1),
        "dim_time": _get_cfg(SEC, "st_dim_time", 128),
        "time_fourier_dim": _get_cfg(SEC, "time_fourier_dim", 64),
        "time_hidden": _get_cfg(SEC, "time_hidden", 256),
        "time_fourier_sigma": _get_cfg(SEC, "time_fourier_sigma", 1.0),
        "cond_dim_in": 4,
        "cond_hidden": _get_cfg(SEC, "cond_hidden", 128),
    }

    # ---- Build/load model ----
    if mode in ("sampling_only", "retrain_and_sampling") and ckpt_path:
        ckpt_abspath = os.path.abspath(os.path.expanduser(ckpt_path))
        if not os.path.isfile(ckpt_abspath):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_abspath}")
        checkpoint = torch.load(ckpt_abspath, map_location=device)
        model = _build_model_from_ckpt_sumradii_compat(
            FlowSetTransformerCenters, checkpoint, device, st_kwargs, target_d=2
        )
    else:
        model = FlowSetTransformerCenters(2, **st_kwargs).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ---- Training (if requested) ----
    if mode in ("training_and_sampling", "retrain_and_sampling", "train_only"):
        if not dataset_path:
            raise ValueError(f"{SEC}.dataset_path must be set for mode='{mode}'")

        full_ds = CircleCentersDataset(dataset_path)
        max_samples = len(full_ds)

        gen = torch.Generator().manual_seed(split_seed)
        perm = torch.randperm(max_samples, generator=gen)
        test_size = max(1, int(round(max_samples * test_fraction)))
        test_idx = perm[:test_size].tolist()
        train_idx = perm[test_size:].tolist()

        # Filter training indices to top-X% by sum of radii
        if train_top_fraction is not None and train_top_fraction > 0.0:
            K = max(1, int(round(max_samples * min(1.0, train_top_fraction))))
            sum_r_all = full_ds.cond[:, 1]  # (M,)
            top_idx_overall = torch.topk(sum_r_all, k=K, largest=True).indices.tolist()
            top_set = set(top_idx_overall)
            filtered_train_idx = [i for i in train_idx if i in top_set]
            if len(filtered_train_idx) == 0:
                filtered_train_idx = [i for i in top_idx_overall if i not in set(test_idx)] or top_idx_overall
            print(
                f"[Filter] Training restricted to top {K}/{max_samples} ({100.0*min(1.0, train_top_fraction):.1f}%) by sum(r). "
                f"Train size: {len(filtered_train_idx)} (was {len(train_idx)})."
            )
            train_idx = filtered_train_idx

        train_ds = Subset(full_ds, train_idx)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        print(f"Train size: {len(train_ds)} | Test size: {len(test_idx)}")

        model, _, ckpt_out = train_flow_model_centers(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            num_epochs=num_epochs,
            mse_strength=mse_strength,
            geom_strength=geom_strength,
            device=device,
            params=st_kwargs,
            save_path=save_model_path,
        )

        if state is not None:
            state.set_model_path(ckpt_out)

    if mode == "train_only":
        print("Training only mode – skipping sampling.")
        return None

    # If sampling_only and state.model_path is empty, propagate loaded checkpoint
    if mode == "sampling_only" and state is not None and getattr(state, "model_path", "") == "" and ckpt_path:
        state.set_model_path(os.path.abspath(os.path.expanduser(ckpt_path)))

    # ---- Sampling ----
    samples = sample_flow_model_centers_only(
        model=model,
        optimizer=optimizer,
        num_samples=num_new,
        batch_size=batch_new,
        num_points=N_points,
        device=device,
        L=box_len,
        gas_steps=_get_cfg(SEC, "gas_steps", 40),
        ode_method=_get_cfg(SEC, "ode_method", "midpoint"),
        ode_step_cap=_get_cfg(SEC, "ode_step_cap", 0.05),
        proj_iters=_get_cfg(SEC, "proj_iters", 6),
        alpha_proj=_get_cfg(SEC, "alpha_proj", 0.25),
        contact_q=_get_cfg(SEC, "contact_q", 1.0),
        wall_weight=_get_cfg(SEC, "wall_weight", 1.0),
        wall_margin=_get_cfg(SEC, "wall_margin", 0.05),
        prox_iters=_get_cfg(SEC, "prox_iters", 8),
        prox_step=_get_cfg(SEC, "prox_step", 0.1),
        prox_lambda=_get_cfg(SEC, "prox_lambda", 1.5),
        final_passes=_get_cfg(SEC, "final_passes", 4),
        tol_finish=_get_cfg(SEC, "tol_finish", 1e-8),
    )

    os.makedirs(save_gen_path, exist_ok=True)
    out_path = os.path.join(save_gen_path, f"flow_centers_only_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
    torch.save(torch.from_numpy(samples), out_path)
    print(f"Saved {samples.shape[0]} samples (centres only, r=0) to {out_path}")

    if state is not None:
        state.set_samples_path(out_path)

    # Optional: validate + plot
    metrics = validate_circle_samples(samples, tol=1e-9, L=box_len)
    print(f"Feasible: {metrics['feasible'].sum()}/{metrics['num_samples']} | best sum_r={metrics['sum_r'].max():.6f}")

    try:
        plot_first_k_samples(
            samples,
            k=min(20, samples.shape[0]),
            out_dir=plot_out_dir,
            filename_prefix="circle_centers_only_gen",
            mark_infeasible=True,
            tol=1e-9,
            L=box_len,
        )
        print(f"Saved plots to {plot_out_dir}")
    except Exception as e:
        print(f"[Plot] Warning: failed to plot samples: {e}")

    return out_path


# ---------------- Legacy standalone CLI (now calls main) ----------------
if __name__ == "__main__":
    main(state=None)
