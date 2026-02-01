# flow_matching_tammes.py
# Flow Matching for Tammes: generate N points on S^{d-1} with manifold constraints.

import os
import math
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from datetime import datetime
from tqdm import tqdm

# Optional: comment out matplotlib if running headless
import matplotlib.pyplot as plt

# External deps (as in your original script)
import schedulefree
from x_transformers import ContinuousTransformerWrapper, Encoder

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.utils import ModelWrapper
from torch.distributions import Independent, Normal

try:
    from flow_boost import cfg
    HAS_CFG = True
except Exception:
    HAS_CFG = False

# -------------------------------
# Config helpers
# -------------------------------
def _get_cfg(section, key, fallback):
    if not HAS_CFG:
        return fallback
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

# -------------------------------
# Manifold utilities: S^{d-1}
# -------------------------------
def _normalize_sphere(x, eps=1e-12):
    # x: (B,d,N) -> normalize each point [d] per token N
    nrm = x.norm(dim=1, keepdim=True).clamp_min(eps)  # (B,1,N)
    return x / nrm

def _project_tangent(x, v):
    # project v onto tangent space at x (both (B,d,N))
    # v_t = v - <x,v> x   (since ||x|| = 1 ideally)
    inner = (x * v).sum(dim=1, keepdim=True)          # (B,1,N)
    return v - inner * x

def _uniform_on_sphere(B, d, N, device, dtype):
    g = torch.randn(B, d, N, device=device, dtype=dtype)
    return _normalize_sphere(g)

# -------------------------------
# Riemannian penalty on S^{d-1}
# -------------------------------
_TRI_CACHE = {}
def angular_hinge_penalty(x, t_cap=None, margin=0.0, beta=40.0, p=2, q=0.10, eps=1e-12):
    """
    x: (B,d,N) points on S^{d-1}. Penalize large pairwise dot products (= small angles).
    If t_cap is None, treat any positive dot (cos < 90°) above 'margin' as violation baseline.
    Use smooth hinge (softplus) on (dot - t_eff). Top-k across pairs for robustness.
    """
    B, d, N = x.shape
    X = x.permute(0, 2, 1).contiguous()            # (B,N,d)
    dots = (X @ X.transpose(1, 2)).clamp(-1+eps, 1-eps)  # (B,N,N)

    key = (x.device, N)
    tri = _TRI_CACHE.get(key)
    if tri is None:
        tri = torch.triu_indices(N, N, offset=1, device=x.device)
        _TRI_CACHE[key] = tri

    # extract upper-triangular pairs
    pair_dots = dots[:, tri[0], tri[1]]            # (B, Pairs)

    if t_cap is None:
        threshold = torch.zeros_like(pair_dots) + margin
    else:
        threshold = torch.as_tensor(t_cap, device=x.device, dtype=x.dtype)
        if threshold.ndim == 0:
            threshold = threshold.view(1).expand(B)   # (B,)
        threshold = threshold[:, None] - margin       # (B,1) -> broadcast to (B,Pairs)

    gap = pair_dots - threshold
    v = F.softplus(beta * gap) / beta               # smooth hinge
    if p != 1:
        v = v.pow(p)

    Pairs = v.size(1)
    k = max(1, int(q * Pairs))
    topk, _ = torch.topk(v, k=k, dim=1, largest=True, sorted=False)
    return topk.mean()

# -------------------------------
# Flow path (Euclidean path; we'll project to the sphere)
# -------------------------------
FM_PATH = AffineProbPath(scheduler=CondOTScheduler())  # linear in Euclidean ambient; we'll enforce manifold

# -------------------------------
# Model
# -------------------------------
class FlowSetTransformer(nn.Module):
    class TimeEmbedFourier(nn.Module):
        def __init__(self, out_dim, fourier_dim, hidden, sigma=1.0, include_poly=True):
            super().__init__()
            fourier_dim = max(16, fourier_dim)
            self.register_buffer("freqs", torch.randn(fourier_dim) * sigma, persistent=False)
            self.include_poly = include_poly
            in_dim = 2 * fourier_dim + (4 if include_poly else 0)
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, out_dim),
            )
        def forward(self, t: torch.Tensor) -> torch.Tensor:
            if t.ndim == 2:
                t = t[:, 0]
            t = t.contiguous()
            angles = (t[:, None] * self.freqs[None, :]) * (2.0 * math.pi)
            sin_feat = torch.sin(angles); cos_feat = torch.cos(angles)
            feats = [sin_feat, cos_feat]
            if self.include_poly:
                t1 = t; t2 = t1 * t1; t3 = t2 * t1
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
        ff_mult = float(st_kwargs.get("ff_mult", 4.0))
        attn_do = float(st_kwargs.get("attn_dropout", 0.1))
        ff_do = float(st_kwargs.get("ff_dropout", 0.1))
        dim_time = int(st_kwargs.get("dim_time", max(64, 4 * d)))
        time_F = int(st_kwargs.get("time_fourier_dim", max(16, dim_time // 2)))
        time_hid = int(st_kwargs.get("time_hidden", 2 * dim_time))
        sigma = float(st_kwargs.get("time_fourier_sigma", 1.0))
        cond_dim_in = int(st_kwargs.get("cond_dim_in", 0))
        cond_hidden = int(st_kwargs.get("cond_hidden", max(64, dim_time)))
        self.uses_cond = cond_dim_in > 0

        self.time_emb = self.TimeEmbedFourier(
            out_dim=dim_time, fourier_dim=time_F, hidden=time_hid, sigma=sigma, include_poly=True
        )

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
        """
        t:    (B,) or (B,1)
        x:    (B, d, N) on (or near) S^{d-1}
        cond: (B, C) or None
        -> tangent velocity (B, d, N)
        """
        B, d, N = x.shape
        assert d == self.d
        device = x.device
        t = t.to(device=device, dtype=x.dtype)
        t_1d = t[:, 0] if t.ndim == 2 else t
        t_embed = self.time_emb(t_1d)  # (B, dim_time)

        if self.uses_cond:
            if cond is None:
                cond = torch.zeros(B, self.cond_mlp[0].in_features, device=device, dtype=x.dtype)
            else:
                cond = cond.to(device=device, dtype=x.dtype)
            c_embed = self.cond_mlp(cond)
        else:
            c_embed = None

        tokens = x.permute(0, 2, 1).contiguous()            # (B, N, d)
        if c_embed is not None:
            tc = torch.cat([t_embed, c_embed], dim=-1)
            tc_rep = tc[:, None, :].expand(B, N, -1)
            h = torch.cat([tokens, tc_rep], dim=-1)
            film_in = tc
        else:
            t_rep = t_embed[:, None, :].expand(B, N, -1)
            h = torch.cat([tokens, t_rep], dim=-1)
            film_in = t_embed

        h = self.token_in(h)                                 # (B, N, model_dim)
        gamma_beta = self.timecond_to_film(film_in)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma[:, None, :]
        beta  = beta[:, None, :]
        h = h * (1 + gamma) + beta

        h = self.encoder(h)                                  # (B, N, model_dim)
        out = self.token_out(h).permute(0, 2, 1).contiguous()# (B, d, N)

        # --- Enforce tangent vector field on S^{d-1} ---
        x_n = _normalize_sphere(x)                           # safe-guard
        out_tan = _project_tangent(x_n, out)
        return out_tan

# -------------------------------
# Dataset for Tammes: (M, d, N) on S^{d-1}
# -------------------------------
class SpherePointsDataset(Dataset):
    def __init__(self, path, chunk=256):
        data = torch.load(path)  # (M, d, N)
        assert data.ndim == 3
        self.data = data.contiguous().float()
        self.M, self.d, self.N = self.data.shape

        # Optional conditioning: min separation angle (cos = max dot)
        # Compute per-sample cosine of min angle (i.e., max pairwise dot)
        cs = []
        for s in range(0, self.M, chunk):
            e = min(self.M, s + chunk)
            xb = self.data[s:e]                    # (B,d,N)
            X = xb.permute(0, 2, 1)                # (B,N,d)
            X = X / (X.norm(dim=2, keepdim=True).clamp_min(1e-12))
            dots = torch.einsum("bid,bjd->bij", X, X).clamp(-1+1e-12, 1-1e-12)  # (B,N,N)
            eye = torch.eye(self.N, device=xb.device, dtype=torch.bool)[None]
            m = dots.masked_fill(eye, -1.0).amax(dim=(1,2))  # max off-diagonal dot
            cs.append(m.cpu())
        cos_min_angle = torch.cat(cs, dim=0)                 # (M,)

        # cond vector: [N/scale_N, cos_min_angle]
        scale_N = 128.0
        self.cond = torch.stack([
            torch.full((self.M,), float(self.N)/scale_N),
            cos_min_angle
        ], dim=1).float()

        print(f"Loaded {path}, shape {tuple(self.data.shape)} | cond shape {tuple(self.cond.shape)}")

    def __len__(self):
        return self.M

    def __getitem__(self, idx):
        x = self.data[idx]
        x = x / (x.norm(dim=0, keepdim=True).clamp_min(1e-12))  # ensure on sphere
        return x, self.cond[idx]

# -------------------------------
# Training
# -------------------------------
def sample_t(B, device, small_t_weight=0.5, gamma=2.0):
    s = torch.rand(B, device=device)
    t_small = s**gamma
    use_small = (torch.rand(B, device=device) < small_t_weight).float()
    return use_small * t_small + (1 - use_small) * s

def _sample_x1_on_sphere_per_batch(x0, jitter=0.0):
    # x0 only used for shape/device
    B, d, N = x0.shape
    u = _uniform_on_sphere(B, d, N, x0.device, x0.dtype)
    if jitter > 0:
        u = _normalize_sphere(u + jitter * torch.randn_like(u))
    return u

def train_flow_model_tammes(
    model, optimizer, loader, num_epochs,
    mse_strength, angle_pen_strength,
    device, params, save_dir, angle_margin=0.0
):
    model.train().to(device)
    optimizer.train()
    mse = nn.MSELoss()
    history = []

    for epoch in tqdm(range(num_epochs), desc="Training"):
        ep_losses = []
        # cosine ramp for the penalty: gentle at start
        ratio = 0.5 * (1 - np.cos(np.pi * min(1.0, epoch / (0.5 * num_epochs))))
        for x_0, cond in loader:
            x_0 = x_0.to(device)     # (B,d,N) on sphere
            cond = cond.to(device)
            B = x_0.size(0)

            # Source x1: uniform on sphere (per sample)
            x_1 = _sample_x1_on_sphere_per_batch(x_0, jitter=0.02)

            # Sample timesteps and FM path in ambient space
            t_in = sample_t(B, device=device, small_t_weight=0.5, gamma=2.0)
            path_sample = FM_PATH.sample(t=t_in, x_0=x_0, x_1=x_1)
            x_t = path_sample.x_t
            dx_t = path_sample.dx_t

            # Project to manifold: feed normalized x_t; target velocity projected to tangent
            x_t_s = _normalize_sphere(x_t)
            dx_t_tan = _project_tangent(x_t_s, dx_t)

            # Predict tangent velocity
            u_pred = model(t_in.squeeze(-1) if t_in.ndim == 2 else t_in, x_t_s, cond=cond)

            # Flow-matching loss (tangent)
            loss_fm = mse(u_pred, dx_t_tan)

            # Angular separation penalty on projected x0 estimate:
            eps_t = 1e-3
            sch0 = FM_PATH.scheduler(t_in)
            sch1 = FM_PATH.scheduler((t_in + eps_t).clamp_max(1.0))
            alpha_dot = ((sch1.alpha_t - sch0.alpha_t) / eps_t).view(-1, 1, 1).to(x_t.dtype)
            sigma_dot = ((sch1.sigma_t - sch0.sigma_t) / eps_t).view(-1, 1, 1).to(x_t.dtype)
            alpha_dot_safe = alpha_dot.sign() * alpha_dot.abs().clamp_min(1e-6)
            x0_proj = (u_pred - sigma_dot * x_1) / alpha_dot_safe
            x0_proj = _normalize_sphere(x0_proj)

            pen = angular_hinge_penalty(
                x0_proj, t_cap=None,
                margin=angle_margin, beta=80.0, p=2, q=0.20
            )

            loss = mse_strength * loss_fm + angle_pen_strength * ratio * pen
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ep_losses.append([loss_fm.item(), pen.item(), loss.item()])

        avg = np.mean(ep_losses, axis=0)
        history.append(avg)
        if (epoch+1) % max(1, num_epochs // 10) == 0:
            print(f"[{epoch+1}/{num_epochs}] FM={avg[0]:.4f} Pen={avg[1]:.4f} Tot={avg[2]:.4f}")

    history = np.array(history)
    os.makedirs(save_dir, exist_ok=True)
    loss = history[-1, 2]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"tammes_flow_loss={loss:.4f}_{ts}.pth"
    path = os.path.join(save_dir, name)
    torch.save({
        "model_state": model.state_dict(),
        "opt_state": optimizer.state_dict(),
        "epoch": num_epochs,
        "params": params
    }, path)

    plt.figure(figsize=(12, 6))
    plt.plot(history[:, 0], label="FM MSE")
    plt.plot(history[:, 1], label="Angular Penalty")
    plt.plot(history[:, 2], label="Total")
    plt.yscale('log'); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True, which='both', ls=':')
    plt.savefig(os.path.join(save_dir, f"tammes_flow_loss_{ts}.png"))
    plt.close()
    return model, history

# -------------------------------
# Sampling (manifold-constrained)
# -------------------------------
@torch.no_grad()
def sample_flow_model_tammes(
    model, optimizer, num_samples, batch_size, num_points,
    device, dim, cond_loader=None
):
    """
    Sampling with manifold constraint on S^{d-1}:
      - initialize u0 uniform on sphere
      - integrate learned tangent field (t=1->0)
      - re-normalize to sphere after integration
      - light final polishing with angular hinge descent in tangent space
    """
    N_steps = 40
    ode_step_cap = 0.05
    jitter = 0.02
    final_passes = 4

    def _tangent_step(u, t_scalar, cond, step):
        # Euler-like step with tangent projection and retraction
        tb = torch.full((u.size(0),), float(t_scalar), device=u.device, dtype=u.dtype)
        v = model(tb, u, cond=cond) if cond is not None else model(tb, u)
        v = _project_tangent(u, v)
        u_next = _normalize_sphere(u + step * v)
        return u_next

    def _final_polish(u):
        # a few steps of angular hinge descent in tangent space
        for _ in range(final_passes):
            # gradient of penalty wrt x approximated by pushing away along tangent
            # We use simple repulsion: v_i ~ sum_j (dot_ij - tau)_+ * x_j projected to tangent
            B, d, N = u.shape
            X = u.permute(0, 2, 1)                      # (B,N,d)
            dots = (X @ X.transpose(1, 2)).clamp(-1+1e-12, 1-1e-12)   # (B,N,N)
            eye = torch.eye(N, device=u.device, dtype=torch.bool)[None]
            dots = dots.masked_fill(eye, -1.0)
            # take positive part as "too close" (threshold 0)
            w = dots.clamp_min(0.0)                     # (B,N,N)
            push = torch.einsum("bij,bjd->bid", w, X)   # (B,N,d)
            push = push.permute(0, 2, 1).contiguous()   # (B,d,N)
            push = _project_tangent(u, push)
            u = _normalize_sphere(u + 0.03 * push)
        return u

    model.eval(); optimizer.eval()
    samples, remaining = [], int(num_samples)
    cond_iter = iter(cond_loader) if cond_loader is not None else None

    while remaining > 0:
        bs = min(batch_size, remaining)
        # optional conditioning
        if cond_iter is not None:
            try:
                x_dummy, cond = next(cond_iter)
            except StopIteration:
                cond_iter = iter(cond_loader)
                x_dummy, cond = next(cond_iter)
            cond = cond.to(device)
            if cond.size(0) > bs:
                cond = cond[:bs]
        else:
            cond = None

        # Initialize on sphere
        u = _uniform_on_sphere(bs, dim, num_points, device, torch.float32)
        u = _normalize_sphere(u + jitter * torch.randn_like(u))

        # integrate from tau=0 -> 1 with small fixed steps; model time t = 1 - tau
        for k in range(N_steps):
            tau = k / N_steps
            tau_next = (k + 1) / N_steps
            t_scalar = 1.0 - tau
            dt = max(1e-3, min(ode_step_cap, (tau_next - tau)))
            u = _tangent_step(u, t_scalar, cond, step=dt)

        u = _final_polish(u)
        samples.append(u.cpu().numpy())
        remaining -= bs

    return np.concatenate(samples, axis=0)

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    sec = _get_cfg("flow_tammes", "_section_name", "flow_tammes")  # unused, just a label
    # Basic settings (similar keys to your cube script where sensible)
    d = _get_cfg("flow_tammes", "dimension", 3)
    bs = _get_cfg("flow_tammes", "batch_size", 64)
    path = _get_cfg("flow_tammes", "dataset_path", "tammes_dataset.pt")
    lr = _get_cfg("flow_tammes", "learning_rate", 5e-3)
    epochs = _get_cfg("flow_tammes", "num_epochs", 1000)
    mse_s  = _get_cfg("flow_tammes", "mse_strength", 1.0)
    pen_s  = _get_cfg("flow_tammes", "angular_penalty_strength", 3.0)
    angle_margin = _get_cfg("flow_tammes", "angle_margin", 0.0)

    # Sampling
    num_new = _get_cfg("flow_tammes", "sample_new_points", 100)
    batch_n = _get_cfg("flow_tammes", "sample_new_points_batch_size", 10)
    pts     = _get_cfg("flow_tammes", "num_points", 89)

    # I/O
    save_m = _get_cfg("flow_tammes", "save_model_path", "output/saved_models/")
    save_g = _get_cfg("flow_tammes", "save_generated_path", "output/generated_sets/")

    # Model hyperparams (reusing your overrides)
    st_kwargs = {
        'dim_hidden': _get_cfg("flow_tammes", "st_dim_hidden", 512),
        'num_heads':  _get_cfg("flow_tammes", "st_num_heads", 8),
        'num_isab':   _get_cfg("flow_tammes", "st_num_isab", 6),
        'cond_dim_in': 2,   # [N/scale, cos_min_angle]
        'dim_out': d
    }

    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    full_ds = SpherePointsDataset(path)
    assert full_ds.d == d, f"Dataset dim {full_ds.d} != configured {d}"
    assert full_ds.N == pts, f"Dataset N {full_ds.N} != configured {pts}"

    max_samples = min(len(full_ds), 25000)
    split_seed   = _get_cfg("flow_tammes", "split_seed", 1234)
    test_fraction= _get_cfg("flow_tammes", "test_fraction", 0.1)
    gen = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(max_samples, generator=gen)
    test_size  = max(1, int(round(max_samples * test_fraction)))
    test_idx   = perm[:test_size].tolist()
    train_idx  = perm[test_size:].tolist()
    train_ds = Subset(full_ds, train_idx)
    test_ds  = Subset(full_ds, test_idx)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_n, shuffle=False)

    print(f"Train size: {len(train_ds)} | Test size: {len(test_ds)}")

    model = FlowSetTransformer(d, **st_kwargs).to(dev)
    opt = schedulefree.RAdamScheduleFree(model.parameters(), lr=lr)

    model, hist = train_flow_model_tammes(
        model, opt, train_loader, epochs,
        mse_s, pen_s,
        dev, st_kwargs, save_m, angle_margin=angle_margin
    )

    samples = sample_flow_model_tammes(
        model, opt, num_new, batch_n, pts,
        dev, d, cond_loader=test_loader
    )

    os.makedirs(save_g, exist_ok=True)
    out_path = os.path.join(save_g, f"tammes_flow_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
    torch.save(torch.from_numpy(samples), out_path)
    print(f"Saved {num_new} samples to {out_path}")
