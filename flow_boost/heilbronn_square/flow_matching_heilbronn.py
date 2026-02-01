import os
import math
import time
import numpy as np
from datetime import datetime

import torch
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import schedulefree
from x_transformers import Encoder

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from flow_boost.heilbronn_square.sample_generation import plot_top_k_minarea_samples

import configparser

try:
    from flow_boost import cfg
    HAS_CFG = True
except Exception:
    HAS_CFG = False

# ============
# FM scheduler
# ============
FM_PATH = AffineProbPath(scheduler=CondOTScheduler())

# ==========================
# Utilities for Heilbronn
# ==========================
def clamp_unit_square(x):
    # x: (B,2,N)
    return x.clamp(0.0, 1.0)

def sample_uniform_x1_like(x0):
    # x0 only for shape/device
    B, d, N = x0.shape
    assert d == 2
    return torch.rand(B, d, N, device=x0.device, dtype=x0.dtype)

def all_triangle_areas(points):
    """
    points: (B,2,N) in [0,1]
    returns: tensor (B, K) of areas per batch (K = C(N,3))
    """
    B, _, N = points.shape
    device = points.device
    dtype = points.dtype
    idx = []
    for i in range(N - 2):
        for j in range(i + 1, N - 1):
            for k in range(j + 1, N):
                idx.append((i, j, k))
    K = len(idx)
    areas = torch.empty(B, K, device=device, dtype=dtype)
    P = points.permute(0, 2, 1)  # (B,N,2)

    for t, (i, j, k) in enumerate(idx):
        A = P[:, i, :]  # (B,2)
        Bp = P[:, j, :]
        C = P[:, k, :]
        BA = Bp - A
        CA = C - A
        cross = BA[:, 0] * CA[:, 1] - BA[:, 1] * CA[:, 0]
        areas[:, t] = 0.5 * cross.abs()
    return areas  # (B,K)

def softmin_triangle_area(points, tau=1e-3, sample_K=None, rng=None):
    """
    Smooth approximation to min area via soft-min: a_soft = -tau * logsumexp(-A/tau)
    Optionally subsample K triangles for efficiency.
    Returns: (B,)
    """
    B, _, N = points.shape
    if sample_K is None:
        A = all_triangle_areas(points)  # (B, K)
    else:
        if rng is None: rng = np.random
        idx = []
        for _ in range(sample_K):
            i, j, k = sorted(rng.choice(N, 3, replace=False).tolist())
            idx.append((i, j, k))
        P = points.permute(0, 2, 1)  # (B,N,2)
        A_list = []
        for (i, j, k) in idx:
            A = P[:, i, :]
            Bp = P[:, j, :]
            C = P[:, k, :]
            BA = Bp - A
            CA = C - A
            cross = BA[:, 0] * CA[:, 1] - BA[:, 1] * CA[:, 0]
            A_list.append(0.5 * cross.abs())
        A = torch.stack(A_list, dim=1)  # (B,sample_K)

    m = torch.amax(-A, dim=1, keepdim=True)
    a_soft = -tau * (m + torch.log(torch.clamp(torch.sum(torch.exp(-A / tau - m), dim=1, keepdim=True), min=1e-20)))
    return a_soft.squeeze(1)  # (B,)

# =======================
# Transformer velocity
# =======================
class FlowSetTransformer(nn.Module):
    class TimeEmbedFourier(nn.Module):
        def __init__(self, out_dim=64, fourier_dim=32, hidden=128, sigma=1.0, include_poly=True):
            super().__init__()
            self.include_poly = include_poly
            self.register_buffer("freqs", torch.randn(max(16, fourier_dim)) * sigma, persistent=False)
            in_dim = 2 * max(16, fourier_dim)
            if include_poly:
                in_dim += 4
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, out_dim),
            )
        def forward(self, t):
            if t.ndim == 2: t = t[:, 0]
            angles = (t[:, None] * self.freqs[None, :]) * (2.0 * math.pi)
            sinf, cosf = torch.sin(angles), torch.cos(angles)
            feats = [sinf, cosf]
            if self.include_poly:
                t1 = t
                t2 = t1 * t1
                t3 = t2 * t1
                tlog = torch.log1p(t1.clamp_min(0))
                feats += [t1[:, None], t2[:, None], t3[:, None], tlog[:, None]]
            h = torch.cat(feats, dim=-1)
            return self.net(h)

    def __init__(self, d=2, **st_kwargs):
        super().__init__()
        self.d = d
        model_dim = int(st_kwargs.get("dim_hidden", 512))
        heads = int(st_kwargs.get("num_heads", 8))
        depth = int(st_kwargs.get("depth", 6))
        attn_do = float(st_kwargs.get("attn_dropout", 0.1))
        ff_do   = float(st_kwargs.get("ff_dropout", 0.1))
        dim_time= int(st_kwargs.get("dim_time", max(64, 4 * d)))
        time_F  = int(st_kwargs.get("time_fourier_dim", max(16, dim_time // 2)))
        time_h  = int(st_kwargs.get("time_hidden", 2 * dim_time))
        sigma   = float(st_kwargs.get("time_fourier_sigma", 1.0))
        cond_in = int(st_kwargs.get("cond_dim_in", 2))  # [N/scale_N, target_min_area]
        self.uses_cond = cond_in > 0

        self.time_emb = self.TimeEmbedFourier(dim_time, time_F, time_h, sigma)

        if self.uses_cond:
            cond_hidden = int(st_kwargs.get("cond_hidden", max(64, dim_time)))
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_in, cond_hidden), nn.SiLU(), nn.Linear(cond_hidden, cond_hidden)
            )
            dim_cond = cond_hidden
        else:
            self.cond_mlp = None
            dim_cond = 0

        self.token_in = nn.Linear(d + dim_time + dim_cond, model_dim)
        self.film_in_dim = dim_time + dim_cond
        self.timecond_to_film = nn.Sequential(nn.SiLU(), nn.Linear(self.film_in_dim, 2 * model_dim))

        self.encoder = Encoder(
            dim=model_dim, depth=depth, heads=heads,
            layer_dropout=0.1, attn_dropout=attn_do, ff_dropout=ff_do,
            use_rmsnorm=True, ff_glu=True, ff_no_bias=True, attn_flash=True
        )
        self.token_out = nn.Linear(model_dim, d)
        nn.init.zeros_(self.token_out.weight); nn.init.zeros_(self.token_out.bias)

    def forward(self, t, x, cond=None):
        B, d, N = x.shape
        assert d == self.d == 2
        if t.ndim == 2: t1d = t[:, 0]
        else: t1d = t
        t_embed = self.time_emb(t1d)

        if self.uses_cond:
            if cond is None:
                cond = torch.zeros(B, self.cond_mlp[0].in_features, device=x.device, dtype=x.dtype)
            c_embed = self.cond_mlp(cond)
            tc = torch.cat([t_embed, c_embed], dim=-1)
            tc_rep = tc[:, None, :].expand(B, N, -1)
            tokens = x.permute(0, 2, 1)
            h = torch.cat([tokens, tc_rep], dim=-1)
            film_in = tc
        else:
            t_rep = t_embed[:, None, :].expand(B, N, -1)
            tokens = x.permute(0, 2, 1)
            h = torch.cat([tokens, t_rep], dim=-1)
            film_in = t_embed

        h = self.token_in(h)
        gamma, beta = self.timecond_to_film(film_in).chunk(2, dim=-1)
        gamma = gamma[:, None, :]; beta = beta[:, None, :]
        h = h * (1 + gamma) + beta

        h = self.encoder(h)
        out = self.token_out(h).permute(0, 2, 1).contiguous()
        return out

# ===================================
# Dataset (points + per-sample conds)
# ===================================
class HeilbronnPointsetDataset(Dataset):
    def __init__(self, path, tol=1e-12, scale_N=128):
        # Prefer safe tensor-only loading when supported by the installed PyTorch.
        try:
            data = torch.load(path, weights_only=True)  # (M, 2, N)
        except TypeError:
            data = torch.load(path)  # (M, 2, N)
        assert data.ndim == 3 and data.shape[1] == 2
        self.data = data.contiguous()
        self.M, self.d, self.N = self.data.shape

        # exact min triangle area per sample
        mins = []
        with torch.no_grad():
            for s in range(self.M):
                A = all_triangle_areas(self.data[s:s+1].to(torch.float64)).squeeze(0)
                mins.append(float(A.min().item()))
        self.min_area = torch.tensor(mins, dtype=self.data.dtype)

        # cond = [N/scale_N, min_area]
        N_scaled = torch.full((self.M,), float(self.N)/float(scale_N), dtype=self.data.dtype)
        self.cond = torch.stack([N_scaled, self.min_area], dim=1)

        print(f"Loaded {path}, shape {tuple(self.data.shape)}; precomputed min areas.")

    def __len__(self): return self.M
    def __getitem__(self, i): return self.data[i], self.cond[i]

# ===================================
# Train loop
# ===================================
def sample_t(B, device, small_t_weight=0.5, gamma=2.0):
    s = torch.rand(B, device=device)
    t_small = s**gamma
    use_small = (torch.rand(B, device=device) < small_t_weight).float()
    return (use_small * t_small + (1 - use_small) * s)

def train_flow_model(
    model, optimizer, loader, num_epochs,
    mse_strength, heil_penalty_strength, device, params, save_dir
):
    model.train().to(device)
    optimizer.train()
    mse = nn.MSELoss()
    history = []

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        ep_losses = []
        ratio = 0.5 * (1 - math.cos(math.pi * min(1.0, epoch / max(1, int(0.5 * num_epochs)))))

        for x_0, cond in loader:
            x_0 = x_0.to(device)       # (B,2,N) in [0,1]
            cond = cond.to(device)     # (B,2)   [N_scaled, min_area]

            B = x_0.size(0)
            x_1 = sample_uniform_x1_like(x_0)

            t_in = sample_t(B, device=device)
            path_sample = FM_PATH.sample(t=t_in, x_0=x_0, x_1=x_1)
            x_t, dx_t = clamp_unit_square(path_sample.x_t), path_sample.dx_t

            u_pred = model(t_in, x_t, cond=cond)

            # Flow-matching loss
            loss_fm = mse(u_pred, dx_t)

            # Auxiliary: encourage high min-triangle area near projected x0
            eps_t = 1e-3
            with torch.no_grad():
                sch0 = FM_PATH.scheduler(t_in)
                sch1 = FM_PATH.scheduler((t_in + eps_t).clamp_max(1.0))
            alpha_dot = ((sch1.alpha_t - sch0.alpha_t) / eps_t).view(-1, 1, 1).to(x_t.dtype)
            sigma_dot = ((sch1.sigma_t - sch0.sigma_t) / eps_t).view(-1, 1, 1).to(x_t.dtype)
            alpha_dot_safe = alpha_dot.sign() * alpha_dot.abs().clamp_min(1e-6)
            x0_proj = (u_pred - sigma_dot * x_1) / alpha_dot_safe
            x0_proj = clamp_unit_square(x0_proj)

            target = cond[:, 1]  # (B,)
            a_soft = softmin_triangle_area(x0_proj, tau=1e-3, sample_K=None)
            deficit = (target - a_soft).clamp_min(0.0)
            loss_heil = F.softplus(80.0 * deficit).mean() / 80.0

            loss = mse_strength * loss_fm + heil_penalty_strength * ratio * loss_heil

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_losses.append([float(loss_fm.item()), float(loss_heil.item()), float(loss.item())])

        avg = np.mean(ep_losses, axis=0)
        history.append(avg)
        print(f"Epoch {epoch+1}/{num_epochs} | FM={avg[0]:.5f} HeilPen={avg[1]:.5f} Tot={avg[2]:.5f}")

    hist = np.array(history, dtype=np.float32)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"heilbronn_fm_loss={hist[-1,2]:.6f}_{ts}.pth"
    path = os.path.join(save_dir, name)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "opt": optimizer.state_dict(),
            "epochs": num_epochs,
            "params": params,
            "history": hist,
        },
        path
    )
    print(f"Saved model to: {path}")
    return model, hist, path

# ================================
# GAS-style sampler (+ polishing)
# ================================
def sample_flow_model(model, optimizer, num_samples, batch_size, num_points, device,
                      cond_loader=None, polish_steps=20, polish_lr=0.02, tau=1e-3,
                      # GAS knobs
                      n_steps=40,
                      ode_method="midpoint",
                      ode_step_cap=0.05,
                      proj_steps=6,
                      proj_lr=0.05,
                      prox_steps=6,
                      prox_lr=0.1,
                      prox_lambda=1.0,
                      de_novo_minarea_bump=0.0,
                      # soft-min annealing
                      tau_start=None,
                      tau_end=None,
                      # optional hard-min refinement
                      hardmin_topk=0,
                      hardmin_steps=0,
                      hardmin_lr_frac=0.5):
    model.eval().to(device)
    if optimizer is not None:
        try:
            optimizer.eval()
        except Exception:
            pass

    # Terminal projection: increase soft-min area via gradient ascent
    def _terminal_projection_softmin(u, steps=6, lr=0.05, tau_soft=1e-3):
        x = u.clone()
        for _ in range(max(1, steps)):
            x.requires_grad_(True)
            a_soft = softmin_triangle_area(x, tau=tau_soft, sample_K=None).mean()
            loss = -a_soft
            (grad,) = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)
            with torch.no_grad():
                x = x - lr * grad
                x = clamp_unit_square(x)
        return x.detach()

    # Proximal relaxation: min 0.5||x - x_hat||^2 - lam * a_soft(x)
    def _prox_relaxed_softmin(u, u0, u_proj, tau_prime, steps=6, lr=0.1, lam=1.0, tau_soft=1e-3):
        u_hat = (1.0 - tau_prime) * u0 + tau_prime * u_proj
        x = u.clone()
        for _ in range(max(1, steps)):
            x.requires_grad_(True)
            a_soft = softmin_triangle_area(x, tau=tau_soft, sample_K=None).mean()
            quad = 0.5 * (x - u_hat).pow(2).mean()
            loss = quad - lam * a_soft
            (grad,) = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)
            with torch.no_grad():
                x = x - lr * grad
                x = clamp_unit_square(x)
        return x.detach()

    samples = []
    remaining = int(num_samples)
    cond_iter = iter(cond_loader) if cond_loader is not None else None

    class _FMVF:
        def __init__(self, mdl, c): self.mdl, self.c = mdl, c
        def __call__(self, x, t, **_):
            tb = torch.full((x.size(0),), float(t), device=x.device, dtype=x.dtype)
            return self.mdl(tb, x, cond=self.c) if self.c is not None else self.mdl(tb, x)

    while remaining > 0:
        # Print progress using tqdm bar
        with tqdm(total=remaining, desc="[Sampling]", unit="sample") as pbar:
            pbar.update(0)

        # Conditioning batch (+ optional de-novo bump on min-area target)
        if cond_iter is not None:
            try:
                _, cond = next(cond_iter)
            except StopIteration:
                cond_iter = iter(cond_loader)
                _, cond = next(cond_iter)
            cond = cond.to(device)
            bs = min(batch_size, remaining, cond.size(0))
            cond = cond[:bs]
            if de_novo_minarea_bump > 0.0:
                cond = cond.clone()
                cond[:, 1] = (cond[:, 1] + float(de_novo_minarea_bump)).clamp(max=1.0)
        else:
            bs = min(batch_size, remaining)
            cond = None

        # Initialize from x1 ~ U([0,1]^2)
        u0 = torch.rand(bs, 2, num_points, device=device)
        u = u0.clone()

        solver = ODESolver(velocity_model=_FMVF(model, cond))

        # GAS loop over tau in [0,1]
        for k in range(max(1, n_steps)):
            tau_k = k / max(1, n_steps)
            tau_n = (k + 1) / max(1, n_steps)

            # Soft-min temperature (anneal if provided)
            if (tau_start is not None) and (tau_end is not None):
                # geometric schedule
                tau_k_use = float(tau_start) * ((float(tau_end) / float(tau_start)) ** (k / max(1, n_steps - 1)))
            else:
                tau_k_use = tau

            # 1) ODE integrate a small step (t = 1 - tau)
            t0, t1 = 1.0 - tau_k, 1.0 - tau_n
            T = torch.tensor([t0, t1], device=u.device, dtype=u.dtype)
            step_size = min(ode_step_cap, max(1e-3, abs(t1 - t0)))
            u = solver.sample(time_grid=T, x_init=u, method=ode_method,
                              step_size=step_size, return_intermediates=False, enable_grad=False)
            u = clamp_unit_square(u)

            # 2) Terminal projection: increase soft-min area
            u_proj = _terminal_projection_softmin(u, steps=proj_steps, lr=proj_lr, tau_soft=tau_k_use)

            # 3) Proximal relaxation: balance projection with source-target blend
            u = _prox_relaxed_softmin(u, u0, u_proj, tau_prime=tau_n,
                                      steps=prox_steps, lr=prox_lr, lam=prox_lambda, tau_soft=tau_k_use)

        # Optional light polishing (soft-min)
        tau_polish = tau_end if (tau_start is not None and tau_end is not None) else tau
        if polish_steps > 0:
            u = u.clone().detach().requires_grad_(True)
            opt = torch.optim.SGD([u], lr=polish_lr)
            for _ in range(polish_steps):
                opt.zero_grad()
                a_soft = softmin_triangle_area(u, tau=tau_polish, sample_K=None)
                loss = -a_soft.mean()
                loss.backward()
                with torch.no_grad():
                    u[:] = clamp_unit_square(u)
                opt.step()
            u = u.detach()

        # Optional hard-min top-k refinement (targets true min more directly)
        if hardmin_topk > 0 and hardmin_steps > 0:
            u = u.clone().detach().requires_grad_(True)
            hm_lr = max(1e-5, float(polish_lr) * float(hardmin_lr_frac))
            opt_hm = torch.optim.SGD([u], lr=hm_lr)
            for _ in range(hardmin_steps):
                opt_hm.zero_grad()
                A = all_triangle_areas(u)  # (B, K)
                k = min(int(hardmin_topk), A.size(1))
                vals, _ = torch.topk(A, k, dim=1, largest=False)  # k smallest per batch
                loss_hard = -vals.mean()  # maximize worst areas
                loss_hard.backward()
                with torch.no_grad():
                    u[:] = clamp_unit_square(u)
                opt_hm.step()
            u = u.detach()

        samples.append(u.cpu().numpy())
        remaining -= bs

    return np.concatenate(samples, axis=0)

# ======================
# I/O helpers
# ======================
def load_model_if_exists(model, opt, path, device):
    if path and os.path.isfile(path):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        try:
            opt.load_state_dict(ckpt["opt"])
        except Exception:
            pass
        print(f"Loaded model from {path}")
    return model, opt

# ======================
# Main controlled by INI
# ======================
def main(state=None):
    sec = "heilbronn_flow"
    mode = cfg.get(sec, "mode", fallback="training_and_sampling").strip()

    # Data / IO
    dataset_path = cfg.get(sec, "dataset_path")
    assert dataset_path, "dataset_path required (torch tensor file of shape (M,2,N))"

    save_model_dir = cfg.get(sec, "save_model_dir", fallback="./outputs_heilbronn_models")
    os.makedirs(save_model_dir, exist_ok=True)

    save_generated_dir = cfg.get(sec, "save_generated_dir", fallback="./outputs_heilbronn_samples")
    os.makedirs(save_generated_dir, exist_ok=True)

    # Model + train params
    d = 2
    bs = cfg.getint(sec, "batch_size", fallback=64)
    lr = cfg.getfloat(sec, "learning_rate", fallback=1e-4)
    epochs = cfg.getint(sec, "num_epochs", fallback=200)
    mse_s = cfg.getfloat(sec, "mse_strength", fallback=1.0)
    heil_s= cfg.getfloat(sec, "heilbronn_penalty_strength", fallback=0.2)
    test_fraction = cfg.getfloat(sec, "test_fraction", fallback=0.1)
    split_seed = cfg.getint(sec, "split_seed", fallback=1234)
    
    # Sampling params
    num_new = cfg.getint(sec, "num_generated_samples", fallback=1000)
    batch_n = cfg.getint(sec, "generation_batch_size", fallback=50)
    points_N = cfg.getint(sec, "num_points", fallback=None)  # if None, use N from dataset
    polish_steps = cfg.getint(sec, "polish_steps", fallback=20)
    polish_lr = cfg.getfloat(sec, "polish_lr", fallback=0.02)
    tau_softmin = cfg.getfloat(sec, "tau_softmin", fallback=1e-3)
    plot_k = cfg.getint(sec, "plot_top_k_samples", fallback=5)
    plot_dir = cfg.get(sec, "plot_save_dir", fallback="./outputs_heilbronn_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # NEW: GAS sampling knobs
    gas_steps = cfg.getint(sec, "gas_steps", fallback=40)
    ode_method = cfg.get(sec, "ode_method", fallback="midpoint")
    ode_step_cap = cfg.getfloat(sec, "ode_step_cap", fallback=0.05)
    proj_steps = cfg.getint(sec, "proj_steps", fallback=6)
    proj_lr = cfg.getfloat(sec, "proj_lr", fallback=0.05)
    prox_steps = cfg.getint(sec, "prox_steps", fallback=6)
    prox_lr = cfg.getfloat(sec, "prox_lr", fallback=0.1)
    prox_lambda = cfg.getfloat(sec, "prox_lambda", fallback=1.0)
    de_novo_minarea_bump = cfg.getfloat(sec, "de_novo_minarea_bump", fallback=0.0)
    # soft-min annealing
    tau_softmin_start = cfg.getfloat(sec, "tau_softmin_start", fallback=tau_softmin)
    tau_softmin_end   = cfg.getfloat(sec, "tau_softmin_end",   fallback=tau_softmin)
    # hard-min top-k refinement
    hardmin_topk    = cfg.getint(sec, "hardmin_topk",    fallback=0)
    hardmin_steps   = cfg.getint(sec, "hardmin_steps",   fallback=0)
    hardmin_lr_frac = cfg.getfloat(sec, "hardmin_lr_frac", fallback=0.5)

    # Architecture
    st_kwargs = {
        'dim_hidden':     cfg.getint(sec, "st_dim_hidden", fallback=512),
        'num_heads':      cfg.getint(sec, "st_num_heads", fallback=8),
        'depth':          cfg.getint(sec, "st_depth", fallback=6),
        'attn_dropout':   cfg.getfloat(sec, "st_attn_dropout", fallback=0.1),
        'ff_dropout':     cfg.getfloat(sec, "st_ff_dropout", fallback=0.1),
        'dim_time':       cfg.getint(sec, "st_dim_time", fallback=128),
        'time_fourier_dim': cfg.getint(sec, "time_fourier_dim", fallback=64),
        'time_hidden':    cfg.getint(sec, "time_hidden", fallback=256),
        'time_fourier_sigma': cfg.getfloat(sec, "time_fourier_sigma", fallback=1.0),
        'cond_dim_in':    cfg.getint(sec, "cond_dim_in", fallback=2),
        'cond_hidden':    cfg.getint(sec, "cond_hidden", fallback=128),
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    full_ds = HeilbronnPointsetDataset(dataset_path)
    if points_N is None:
        points_N = full_ds.N
    assert points_N == full_ds.N, f"Config num_points={points_N} but dataset N={full_ds.N}"

    gen = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(len(full_ds), generator=gen)
    test_size = max(1, int(round(len(full_ds) * test_fraction)))
    test_idx = perm[:test_size].tolist()
    train_idx = perm[test_size:].tolist()

    # Keep only the top-X% (by min triangle area) within the training split
    train_top_fraction = cfg.getfloat(sec, "train_top_fraction", fallback=0.5)
    try:
        frac = float(train_top_fraction)
    except Exception:
        frac = 0.5
    if frac < 1.0 and len(train_idx) > 0:
        k = max(1, int(math.ceil(len(train_idx) * frac)))
        if k < len(train_idx):
            # min_area is precomputed in the dataset
            min_area_all = full_ds.min_area  # tensor shape (M,)
            idx_tensor = torch.tensor(train_idx, dtype=torch.long)
            min_area_train = min_area_all[idx_tensor]
            _, top_pos = torch.topk(min_area_train, k=k, largest=True)
            filtered_train_idx = idx_tensor[top_pos].tolist()
            print(f"[Heilbronn] Filtering training to top {100.0*frac:.1f}% by min-area: {len(filtered_train_idx)} of {len(train_idx)}")
            train_idx = filtered_train_idx
        else:
            print(f"[Heilbronn] train_top_fraction keeps all {len(train_idx)} training samples.")

    train_ds = Subset(full_ds, train_idx)
    test_ds  = Subset(full_ds,  test_idx)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_n, shuffle=False)

    print(f"[Heilbronn] Train size: {len(train_ds)} | Test size: {len(test_ds)} | N={points_N}")

    model = FlowSetTransformer(d, **st_kwargs).to(device)
    opt = schedulefree.RAdamScheduleFree(model.parameters(), lr=lr)

    resume_path = cfg.get(sec, "resume_model_path", fallback="").strip()

    if mode == "training_and_sampling":
        model, hist, model_path = train_flow_model(
            model, opt, train_loader, epochs,
            mse_s, heil_s, device, st_kwargs, save_model_dir
        )
        samples = sample_flow_model(
            model, opt, num_new, batch_n, points_N, device,
            cond_loader=test_loader, polish_steps=polish_steps, polish_lr=polish_lr, tau=tau_softmin,
            n_steps=gas_steps, ode_method=ode_method, ode_step_cap=ode_step_cap,
            proj_steps=proj_steps, proj_lr=proj_lr,
            prox_steps=prox_steps, prox_lr=prox_lr, prox_lambda=prox_lambda,
            de_novo_minarea_bump=de_novo_minarea_bump,
            tau_start=tau_softmin_start, tau_end=tau_softmin_end,
            hardmin_topk=hardmin_topk, hardmin_steps=hardmin_steps, hardmin_lr_frac=hardmin_lr_frac
        )
    elif mode == "sampling_only":
        assert resume_path and os.path.isfile(resume_path), "resume_model_path must point to a saved model"
        model, opt = load_model_if_exists(model, opt, resume_path, device)
        samples = sample_flow_model(
            model, opt, num_new, batch_n, points_N, device,
            cond_loader=test_loader, polish_steps=polish_steps, polish_lr=polish_lr, tau=tau_softmin,
            n_steps=gas_steps, ode_method=ode_method, ode_step_cap=ode_step_cap,
            proj_steps=proj_steps, proj_lr=proj_lr,
            prox_steps=prox_steps, prox_lr=prox_lr, prox_lambda=prox_lambda,
            de_novo_minarea_bump=de_novo_minarea_bump,
            tau_start=tau_softmin_start, tau_end=tau_softmin_end,
            hardmin_topk=hardmin_topk, hardmin_steps=hardmin_steps, hardmin_lr_frac=hardmin_lr_frac
        )
        model_path = resume_path
    elif mode == "retrain_and_sampling":
        assert resume_path and os.path.isfile(resume_path), "resume_model_path must point to a saved model"
        model, opt = load_model_if_exists(model, opt, resume_path, device)
        model, hist, model_path = train_flow_model(
            model, opt, train_loader, epochs,
            mse_s, heil_s, device, st_kwargs, save_model_dir
        )
        samples = sample_flow_model(
            model, opt, num_new, batch_n, points_N, device,
            cond_loader=test_loader, polish_steps=polish_steps, polish_lr=polish_lr, tau=tau_softmin,
            n_steps=gas_steps, ode_method=ode_method, ode_step_cap=ode_step_cap,
            proj_steps=proj_steps, proj_lr=proj_lr,
            prox_steps=prox_steps, prox_lr=prox_lr, prox_lambda=prox_lambda,
            de_novo_minarea_bump=de_novo_minarea_bump,
            tau_start=tau_softmin_start, tau_end=tau_softmin_end,
            hardmin_topk=hardmin_topk, hardmin_steps=hardmin_steps, hardmin_lr_frac=hardmin_lr_frac
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(save_generated_dir, f"heilbronn_gen_{num_new}x{points_N}_{ts}.pt")
    torch.save(torch.from_numpy(samples), out_path)
    print(f"Saved {num_new} generated pointsets to {out_path}")
    print(f"(Model path: {model_path})")

    # Save plots for the top_k samples (by minimum triangle area)
    if plot_k > 0:
        plot_top_k_minarea_samples(samples, plot_k, plot_dir, filename_prefix="heilbronn_mintriangles")
        print(f"Saved plots:   {plot_dir}")

    if state is not None:
        state.set_model_path(model_path)
        state.set_samples_path(out_path)

if __name__ == "__main__":
    main()