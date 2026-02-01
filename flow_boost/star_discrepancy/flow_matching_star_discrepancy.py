import os
import math
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
# Utilities for Star Discrepancy
# ==========================
def clamp_unit_square(x):
    # x: (B,2,N)
    return x.clamp(0.0, 1.0)

def sample_uniform_x1_like(x0):
    B, d, N = x0.shape
    assert d == 2
    return torch.rand(B, d, N, device=x0.device, dtype=x0.dtype)

def sample_latin_hypercube_x1_like(x0):
    """
    Geometrically efficient initialization in [0,1]^2:
    Latin Hypercube Sampling per batch (jittered stratified).
    Shape: (B,2,N).
    """
    B, d, N = x0.shape
    assert d == 2
    device = x0.device
    dtype = x0.dtype
    # bins [0..N-1], independently permuted for x and y per batch
    bins = torch.arange(N, device=device)
    x = torch.empty(B, N, device=device, dtype=dtype)
    y = torch.empty(B, N, device=device, dtype=dtype)
    # jitter in each bin
    jx = torch.rand(B, N, device=device, dtype=dtype)
    jy = torch.rand(B, N, device=device, dtype=dtype)
    for b in range(B):
        px = bins[torch.randperm(N, device=device)]
        py = bins[torch.randperm(N, device=device)]
        x[b] = (px.to(dtype) + jx[b]) / float(N)
        y[b] = (py.to(dtype) + jy[b]) / float(N)
    out = torch.stack([x, y], dim=1)  # (B,2,N)
    return out

def make_uniform_grid_torch(Gx, Gy, device, dtype):
    Ax = torch.linspace(1.0 / float(Gx), 1.0, int(Gx), device=device, dtype=dtype)
    Ay = torch.linspace(1.0 / float(Gy), 1.0, int(Gy), device=device, dtype=dtype)
    return Ax, Ay

def _sigmoid_torch(z):
    return torch.sigmoid(z)

def star_surrogate_torch(points, Ax, Ay, beta_softmax=200.0, tau_sigmoid=0.01, eps_abs=1e-12):
    """
    points: (B,2,N) in [0,1]
    Ax: (U,), Ay: (V,)
    returns: (B,) smooth star discrepancy surrogate:
        (1/beta) log sum_{u,v} exp(beta * |C(u,v) - a*b|)
    where C(u,v) = (1/N) sum_i sigmoid((a-x_i)/tau)*sigmoid((b-y_i)/tau).
    """
    B, d, N = points.shape
    assert d == 2
    x = points[:, 0, :]  # (B,N)
    y = points[:, 1, :]  # (B,N)

    # gates: Sx (B,U,N), Sy (B,V,N)
    Sx = _sigmoid_torch((Ax.view(1, -1, 1) - x.view(B, 1, N)) / tau_sigmoid)
    Sy = _sigmoid_torch((Ay.view(1, -1, 1) - y.view(B, 1, N)) / tau_sigmoid)

    # C (B,U,V) via einsum over N
    # C[u,v] = (1/N) sum_i Sx[u,i]*Sy[v,i]
    C = torch.einsum("bun,bvn->buv", Sx, Sy) / float(N)

    # a*b grid (U,V) broadcast
    AB = (Ax.view(-1, 1) * Ay.view(1, -1)).to(points.dtype)  # (U,V)
    Delta = C - AB.view(1, *AB.shape)  # (B,U,V)
    Dabs = torch.sqrt(Delta * Delta + float(eps_abs))  # smooth abs

    # log-sum-exp
    Xlog = beta_softmax * Dabs
    # (B,)
    return (torch.logsumexp(Xlog.flatten(1), dim=1) / beta_softmax)

def star_abs_grid_torch(points, Ax, Ay, tau_sigmoid=0.01, eps_abs=1e-12):
    """
    Returns Dabs grid: (B,U,V) of smooth |C(u,v)-a*b|.
    Useful for top-k refinement.
    """
    B, d, N = points.shape
    assert d == 2
    x = points[:, 0, :]
    y = points[:, 1, :]
    Sx = _sigmoid_torch((Ax.view(1, -1, 1) - x.view(B, 1, N)) / tau_sigmoid)
    Sy = _sigmoid_torch((Ay.view(1, -1, 1) - y.view(B, 1, N)) / tau_sigmoid)
    C = torch.einsum("bun,bvn->buv", Sx, Sy) / float(N)
    AB = (Ax.view(-1, 1) * Ay.view(1, -1)).to(points.dtype)
    Delta = C - AB.view(1, *AB.shape)
    return torch.sqrt(Delta * Delta + float(eps_abs))

# =======================
# Exact star discrepancy (numpy; used for conditioning + plotting only)
# =======================
def exact_star_discrepancy_2d_numpy(pts_np):
    """
    Exact 2D star discrepancy on the critical grid.
    Checks open [0,a)×[0,b) and closed [0,a]×[0,b].
    pts_np: (N,2) in [0,1]
    """
    pts = np.asarray(pts_np, dtype=np.float64)
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

    return max(D_minus_max, D_plus_max), {"open_max": D_minus_max, "closed_max": D_plus_max}

def plot_top_k_lowdiscrepancy_samples(samples_np, k, out_dir, filename_prefix="star_disc"):
    """
    samples_np: (M,2,N) OR (M,N,2) numpy array
    Plots the best K by exact star discrepancy.
    """
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    arr = samples_np
    if arr.ndim != 3:
        raise ValueError(f"Expected (M,2,N) or (M,N,2), got {arr.shape}")
    if arr.shape[1] == 2:
        M, _, N = arr.shape
        get_pts = lambda s: arr[s].T
    elif arr.shape[-1] == 2:
        M, N, _ = arr.shape
        get_pts = lambda s: arr[s]
    else:
        raise ValueError(f"Bad shape {arr.shape}")

    Ds = np.zeros(M, dtype=np.float64)
    for s in range(M):
        pts = get_pts(s).astype(np.float64)
        Ds[s], _ = exact_star_discrepancy_2d_numpy(pts)

    order = np.argsort(Ds)  # smaller better
    for rank in range(min(k, M)):
        s = int(order[rank])
        pts = get_pts(s).astype(np.float64)
        D, info = exact_star_discrepancy_2d_numpy(pts)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="black")
        ax.scatter(pts[:, 0], pts[:, 1], s=12)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"Rank {rank+1} (sample {s}) N={pts.shape[0]}  D*={D:.8f}")
        out_path = os.path.join(
            out_dir,
            f"{filename_prefix}_n={pts.shape[0]}_rank={rank+1}_D={D:.8f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

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
        cond_in = int(st_kwargs.get("cond_dim_in", 2))  # [N/scale_N, target_star_discrepancy]
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
class StarDiscrepancyPointsetDataset(Dataset):
    def __init__(self, path, scale_N=128):
        data = torch.load(path)  # (M, 2, N) expected
        assert data.ndim == 3 and data.shape[1] == 2
        self.data = data.contiguous()
        self.M, self.d, self.N = self.data.shape

        # exact star discrepancy per sample (numpy, critical grid)
        ds = []
        with torch.no_grad():
            for s in tqdm(range(self.M), desc="[Dataset] exact star discrepancy"):
                pts = self.data[s].permute(1, 0).cpu().numpy().astype(np.float64)  # (N,2)
                D, _ = exact_star_discrepancy_2d_numpy(pts)
                ds.append(float(D))
        self.star_disc = torch.tensor(ds, dtype=self.data.dtype)

        # cond = [N/scale_N, star_disc]
        N_scaled = torch.full((self.M,), float(self.N) / float(scale_N), dtype=self.data.dtype)
        self.cond = torch.stack([N_scaled, self.star_disc], dim=1)

        print(f"Loaded {path}, shape {tuple(self.data.shape)}; precomputed exact star discrepancy.")

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
    mse_strength, star_penalty_strength, device, params, save_dir,
    # star discrepancy surrogate knobs
    grid_x=64, grid_y=64, beta_softmax=200.0, tau_sigmoid=0.01, eps_abs=1e-12,
    penalty_kappa=80.0,
):
    model.train().to(device)
    optimizer.train()
    mse = nn.MSELoss()
    history = []

    os.makedirs(save_dir, exist_ok=True)

    # fixed uniform grid for differentiable surrogate in training
    Ax, Ay = make_uniform_grid_torch(grid_x, grid_y, device=device, dtype=torch.float32)

    for epoch in range(num_epochs):
        ep_losses = []
        ratio = 0.5 * (1 - math.cos(math.pi * min(1.0, epoch / max(1, int(0.5 * num_epochs)))))

        for x_0, cond in loader:
            x_0 = x_0.to(device)       # (B,2,N) in [0,1]
            cond = cond.to(device)     # (B,2)   [N_scaled, target_star_disc]

            B = x_0.size(0)

            # source distribution x_1: use LHS (more geometric than pure uniform)
            x_1 = sample_latin_hypercube_x1_like(x_0)

            t_in = sample_t(B, device=device)
            path_sample = FM_PATH.sample(t=t_in, x_0=x_0, x_1=x_1)
            x_t, dx_t = clamp_unit_square(path_sample.x_t), path_sample.dx_t

            u_pred = model(t_in, x_t, cond=cond)

            # Flow-matching loss
            loss_fm = mse(u_pred, dx_t)

            # Auxiliary: encourage low star discrepancy near projected x0
            eps_t = 1e-3
            with torch.no_grad():
                sch0 = FM_PATH.scheduler(t_in)
                sch1 = FM_PATH.scheduler((t_in + eps_t).clamp_max(1.0))
            alpha_dot = ((sch1.alpha_t - sch0.alpha_t) / eps_t).view(-1, 1, 1).to(x_t.dtype)
            sigma_dot = ((sch1.sigma_t - sch0.sigma_t) / eps_t).view(-1, 1, 1).to(x_t.dtype)
            alpha_dot_safe = alpha_dot.sign() * alpha_dot.abs().clamp_min(1e-6)

            # x0_proj ~ inferred x0 from u_pred and x1
            x0_proj = (u_pred - sigma_dot * x_1) / alpha_dot_safe
            x0_proj = clamp_unit_square(x0_proj)

            target = cond[:, 1]  # (B,) exact discrepancy of training sample

            # surrogate discrepancy to minimize
            disc_soft = star_surrogate_torch(
                x0_proj.to(torch.float32), Ax, Ay,
                beta_softmax=float(beta_softmax),
                tau_sigmoid=float(tau_sigmoid),
                eps_abs=float(eps_abs),
            ).to(x0_proj.dtype)

            # penalize if disc_soft > target (we want <= target)
            deficit = (disc_soft - target).clamp_min(0.0)
            loss_star = F.softplus(float(penalty_kappa) * deficit).mean() / float(penalty_kappa)

            loss = mse_strength * loss_fm + star_penalty_strength * ratio * loss_star

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_losses.append([float(loss_fm.item()), float(loss_star.item()), float(loss.item())])

        avg = np.mean(ep_losses, axis=0)
        history.append(avg)
        print(f"Epoch {epoch+1}/{num_epochs} | FM={avg[0]:.5f} StarPen={avg[1]:.5f} Tot={avg[2]:.5f}")

    hist = np.array(history, dtype=np.float32)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"star_fm_loss={hist[-1,2]:.6f}_{ts}.pth"
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
                      cond_loader=None,
                      # star surrogate knobs
                      grid_x=64, grid_y=64, beta_softmax=200.0, tau_sigmoid=0.01, eps_abs=1e-12,
                      # polishing knobs
                      polish_steps=20, polish_lr=0.02,
                      # GAS knobs
                      n_steps=40,
                      ode_method="midpoint",
                      ode_step_cap=0.05,
                      proj_steps=6,
                      proj_lr=0.05,
                      prox_steps=6,
                      prox_lr=0.1,
                      prox_lambda=1.0,
                      # optional conditioning bump: ask for slightly smaller discrepancy
                      de_novo_disc_bump=0.0,
                      # optional anneal tau_sigmoid in projection/prox (geometric)
                      tau_sigmoid_start=None,
                      tau_sigmoid_end=None,
                      # optional top-k max-box refinement
                      hardmax_topk=0,
                      hardmax_steps=0,
                      hardmax_lr_frac=0.5):
    model.eval().to(device)
    if optimizer is not None:
        try:
            optimizer.eval()
        except Exception:
            pass

    Ax, Ay = make_uniform_grid_torch(grid_x, grid_y, device=device, dtype=torch.float32)

    # Terminal projection: decrease star surrogate via gradient descent
    def _terminal_projection_star(u, steps=6, lr=0.05, tau_gate=0.01):
        x = u.clone()
        for _ in range(max(1, steps)):
            x.requires_grad_(True)
            disc = star_surrogate_torch(
                x.to(torch.float32), Ax, Ay,
                beta_softmax=float(beta_softmax),
                tau_sigmoid=float(tau_gate),
                eps_abs=float(eps_abs),
            ).mean()
            loss = disc  # minimize
            (grad,) = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)
            with torch.no_grad():
                x = x - lr * grad
                x = clamp_unit_square(x)
        return x.detach()

    # Proximal relaxation: min 0.5||x - u_hat||^2 + lam * disc(x)
    def _prox_relaxed_star(u, u0, u_proj, tau_prime, steps=6, lr=0.1, lam=1.0, tau_gate=0.01):
        u_hat = (1.0 - tau_prime) * u0 + tau_prime * u_proj
        x = u.clone()
        for _ in range(max(1, steps)):
            x.requires_grad_(True)
            disc = star_surrogate_torch(
                x.to(torch.float32), Ax, Ay,
                beta_softmax=float(beta_softmax),
                tau_sigmoid=float(tau_gate),
                eps_abs=float(eps_abs),
            ).mean()
            quad = 0.5 * (x - u_hat).pow(2).mean()
            loss = quad + lam * disc
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
        # Conditioning batch
        if cond_iter is not None:
            try:
                _, cond = next(cond_iter)
            except StopIteration:
                cond_iter = iter(cond_loader)
                _, cond = next(cond_iter)
            cond = cond.to(device)
            bs = min(batch_size, remaining, cond.size(0))
            cond = cond[:bs]
            # Ask for smaller discrepancy than training target if desired
            if de_novo_disc_bump != 0.0:
                cond = cond.clone()
                cond[:, 1] = (cond[:, 1] + float(de_novo_disc_bump)).clamp(min=0.0)
        else:
            bs = min(batch_size, remaining)
            cond = None

        # Initialize from geometrically efficient x1: LHS
        u0 = sample_latin_hypercube_x1_like(torch.empty(bs, 2, num_points, device=device, dtype=torch.float32))
        u = u0.clone()

        solver = ODESolver(velocity_model=_FMVF(model, cond))

        for k in range(max(1, n_steps)):
            tau_k = k / max(1, n_steps)
            tau_n = (k + 1) / max(1, n_steps)

            # Anneal gate temperature (optional)
            if (tau_sigmoid_start is not None) and (tau_sigmoid_end is not None):
                tau_gate = float(tau_sigmoid_start) * ((float(tau_sigmoid_end) / float(tau_sigmoid_start)) ** (k / max(1, n_steps - 1)))
            else:
                tau_gate = float(tau_sigmoid)

            # 1) ODE integrate a small step (t = 1 - tau)
            t0, t1 = 1.0 - tau_k, 1.0 - tau_n
            T = torch.tensor([t0, t1], device=u.device, dtype=u.dtype)
            step_size = min(ode_step_cap, max(1e-3, abs(t1 - t0)))
            u = solver.sample(time_grid=T, x_init=u, method=ode_method,
                              step_size=step_size, return_intermediates=False, enable_grad=False)
            u = clamp_unit_square(u)

            # 2) Projection: reduce star surrogate
            u_proj = _terminal_projection_star(u, steps=proj_steps, lr=proj_lr, tau_gate=tau_gate)

            # 3) Prox: balance projection with source-target blend
            u = _prox_relaxed_star(u, u0, u_proj, tau_prime=tau_n,
                                   steps=prox_steps, lr=prox_lr, lam=prox_lambda, tau_gate=tau_gate)

        # Optional light polishing: further reduce surrogate
        if polish_steps > 0:
            u = u.clone().detach().requires_grad_(True)
            opt = torch.optim.SGD([u], lr=polish_lr)
            for _ in range(polish_steps):
                opt.zero_grad()
                disc = star_surrogate_torch(
                    u.to(torch.float32), Ax, Ay,
                    beta_softmax=float(beta_softmax),
                    tau_sigmoid=float(tau_sigmoid),
                    eps_abs=float(eps_abs),
                )
                loss = disc.mean()
                loss.backward()
                with torch.no_grad():
                    u[:] = clamp_unit_square(u)
                opt.step()
            u = u.detach()

        # Optional “hard max” refinement: minimize mean of top-k box errors (closest to sup)
        if hardmax_topk > 0 and hardmax_steps > 0:
            u = u.clone().detach().requires_grad_(True)
            hm_lr = max(1e-5, float(polish_lr) * float(hardmax_lr_frac))
            opt_hm = torch.optim.SGD([u], lr=hm_lr)
            for _ in range(hardmax_steps):
                opt_hm.zero_grad()
                Dabs = star_abs_grid_torch(
                    u.to(torch.float32), Ax, Ay,
                    tau_sigmoid=float(tau_sigmoid),
                    eps_abs=float(eps_abs),
                )  # (B,U,V)
                flat = Dabs.flatten(1)  # (B, U*V)
                kk = min(int(hardmax_topk), flat.size(1))
                vals, _ = torch.topk(flat, kk, dim=1, largest=True)  # largest discrepancies
                loss_hard = vals.mean()
                loss_hard.backward()
                with torch.no_grad():
                    u[:] = clamp_unit_square(u)
                opt_hm.step()
            u = u.detach()

        samples.append(u.detach().cpu().numpy())
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
# Main controlled by INI (3 modes preserved)
# ======================
def main(state=None):
    # Prefer "star_flow" but keep backward compatibility if you still have "heilbronn_flow"
    sec = "star_flow"
    if HAS_CFG and not cfg.has_section(sec):
        sec = "heilbronn_flow"

    mode = cfg.get(sec, "mode", fallback="training_and_sampling").strip()

    # Data / IO
    dataset_path = cfg.get(sec, "dataset_path", fallback="")
    assert dataset_path, "dataset_path required (torch tensor file of shape (M,2,N))"

    save_model_dir = cfg.get(sec, "save_model_dir", fallback="./outputs_star_models")
    os.makedirs(save_model_dir, exist_ok=True)

    save_generated_dir = cfg.get(sec, "save_generated_dir", fallback="./outputs_star_samples")
    os.makedirs(save_generated_dir, exist_ok=True)

    # Model + train params
    d = 2
    bs = cfg.getint(sec, "batch_size", fallback=64)
    lr = cfg.getfloat(sec, "learning_rate", fallback=1e-4)
    epochs = cfg.getint(sec, "num_epochs", fallback=200)
    mse_s = cfg.getfloat(sec, "mse_strength", fallback=1.0)
    star_s = cfg.getfloat(sec, "star_penalty_strength", fallback=0.2)
    penalty_kappa = cfg.getfloat(sec, "penalty_kappa", fallback=80.0)
    test_fraction = cfg.getfloat(sec, "test_fraction", fallback=0.1)
    split_seed = cfg.getint(sec, "split_seed", fallback=1234)

    # Star discrepancy surrogate knobs (train + sampling)
    grid_x = cfg.getint(sec, "grid_x", fallback=64)
    grid_y = cfg.getint(sec, "grid_y", fallback=64)
    beta_softmax = cfg.getfloat(sec, "beta_softmax", fallback=200.0)
    tau_sigmoid = cfg.getfloat(sec, "tau_sigmoid", fallback=0.01)
    eps_abs = cfg.getfloat(sec, "abs_eps", fallback=1e-12)

    # Sampling params
    num_new = cfg.getint(sec, "num_generated_samples", fallback=1000)
    batch_n = cfg.getint(sec, "generation_batch_size", fallback=50)
    points_N = cfg.getint(sec, "num_points", fallback=-1)  # if -1, use N from dataset
    polish_steps = cfg.getint(sec, "polish_steps", fallback=20)
    polish_lr = cfg.getfloat(sec, "polish_lr", fallback=0.02)
    plot_k = cfg.getint(sec, "plot_top_k_samples", fallback=5)
    plot_dir = cfg.get(sec, "plot_save_dir", fallback="./outputs_star_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # GAS sampling knobs
    gas_steps = cfg.getint(sec, "gas_steps", fallback=40)
    ode_method = cfg.get(sec, "ode_method", fallback="midpoint")
    ode_step_cap = cfg.getfloat(sec, "ode_step_cap", fallback=0.05)
    proj_steps = cfg.getint(sec, "proj_steps", fallback=6)
    proj_lr = cfg.getfloat(sec, "proj_lr", fallback=0.05)
    prox_steps = cfg.getint(sec, "prox_steps", fallback=6)
    prox_lr = cfg.getfloat(sec, "prox_lr", fallback=0.1)
    prox_lambda = cfg.getfloat(sec, "prox_lambda", fallback=1.0)
    de_novo_disc_bump = cfg.getfloat(sec, "de_novo_disc_bump", fallback=0.0)

    # Gate annealing in sampling (optional)
    tau_sigmoid_start = cfg.getfloat(sec, "tau_sigmoid_start", fallback=tau_sigmoid)
    tau_sigmoid_end   = cfg.getfloat(sec, "tau_sigmoid_end",   fallback=tau_sigmoid)

    # Optional hard-max refinement
    hardmax_topk    = cfg.getint(sec, "hardmax_topk",    fallback=0)
    hardmax_steps   = cfg.getint(sec, "hardmax_steps",   fallback=0)
    hardmax_lr_frac = cfg.getfloat(sec, "hardmax_lr_frac", fallback=0.5)

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

    full_ds = StarDiscrepancyPointsetDataset(dataset_path)
    if points_N is None or points_N < 0:
        points_N = full_ds.N
    assert points_N == full_ds.N, f"Config num_points={points_N} but dataset N={full_ds.N}"

    gen = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(len(full_ds), generator=gen)
    test_size = max(1, int(round(len(full_ds) * test_fraction)))
    test_idx = perm[:test_size].tolist()
    train_idx = perm[test_size:].tolist()

    # Keep only the top-X% (by *low* discrepancy) within the training split
    train_top_fraction = cfg.getfloat(sec, "train_top_fraction", fallback=0.5)
    frac = float(train_top_fraction)
    if frac < 1.0 and len(train_idx) > 0:
        k = max(1, int(math.ceil(len(train_idx) * frac)))
        if k < len(train_idx):
            disc_all = full_ds.star_disc  # (M,)
            idx_tensor = torch.tensor(train_idx, dtype=torch.long)
            disc_train = disc_all[idx_tensor]
            # smallest discrepancy => best
            vals, pos = torch.topk(-disc_train, k=k, largest=True)  # equivalent to taking k smallest
            filtered_train_idx = idx_tensor[pos].tolist()
            print(f"[Star] Filtering training to top {100.0*frac:.1f}% by LOW discrepancy: {len(filtered_train_idx)} of {len(train_idx)}")
            train_idx = filtered_train_idx
        else:
            print(f"[Star] train_top_fraction keeps all {len(train_idx)} training samples.")

    train_ds = Subset(full_ds, train_idx)
    test_ds  = Subset(full_ds,  test_idx)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_n, shuffle=False)

    print(f"[Star] Train size: {len(train_ds)} | Test size: {len(test_ds)} | N={points_N}")

    model = FlowSetTransformer(d, **st_kwargs).to(device)
    opt = schedulefree.RAdamScheduleFree(model.parameters(), lr=lr)

    resume_path = cfg.get(sec, "resume_model_path", fallback="").strip()

    if mode == "training_and_sampling":
        model, hist, model_path = train_flow_model(
            model, opt, train_loader, epochs,
            mse_s, star_s, device, st_kwargs, save_model_dir,
            grid_x=grid_x, grid_y=grid_y,
            beta_softmax=beta_softmax, tau_sigmoid=tau_sigmoid, eps_abs=eps_abs,
            penalty_kappa=penalty_kappa,
        )
        samples = sample_flow_model(
            model, opt, num_new, batch_n, points_N, device,
            cond_loader=test_loader,
            grid_x=grid_x, grid_y=grid_y,
            beta_softmax=beta_softmax, tau_sigmoid=tau_sigmoid, eps_abs=eps_abs,
            polish_steps=polish_steps, polish_lr=polish_lr,
            n_steps=gas_steps, ode_method=ode_method, ode_step_cap=ode_step_cap,
            proj_steps=proj_steps, proj_lr=proj_lr,
            prox_steps=prox_steps, prox_lr=prox_lr, prox_lambda=prox_lambda,
            de_novo_disc_bump=de_novo_disc_bump,
            tau_sigmoid_start=tau_sigmoid_start, tau_sigmoid_end=tau_sigmoid_end,
            hardmax_topk=hardmax_topk, hardmax_steps=hardmax_steps, hardmax_lr_frac=hardmax_lr_frac,
        )
    elif mode == "sampling_only":
        assert resume_path and os.path.isfile(resume_path), "resume_model_path must point to a saved model"
        model, opt = load_model_if_exists(model, opt, resume_path, device)
        samples = sample_flow_model(
            model, opt, num_new, batch_n, points_N, device,
            cond_loader=test_loader,
            grid_x=grid_x, grid_y=grid_y,
            beta_softmax=beta_softmax, tau_sigmoid=tau_sigmoid, eps_abs=eps_abs,
            polish_steps=polish_steps, polish_lr=polish_lr,
            n_steps=gas_steps, ode_method=ode_method, ode_step_cap=ode_step_cap,
            proj_steps=proj_steps, proj_lr=proj_lr,
            prox_steps=prox_steps, prox_lr=prox_lr, prox_lambda=prox_lambda,
            de_novo_disc_bump=de_novo_disc_bump,
            tau_sigmoid_start=tau_sigmoid_start, tau_sigmoid_end=tau_sigmoid_end,
            hardmax_topk=hardmax_topk, hardmax_steps=hardmax_steps, hardmax_lr_frac=hardmax_lr_frac,
        )
        model_path = resume_path
    elif mode == "retrain_and_sampling":
        assert resume_path and os.path.isfile(resume_path), "resume_model_path must point to a saved model"
        model, opt = load_model_if_exists(model, opt, resume_path, device)
        model, hist, model_path = train_flow_model(
            model, opt, train_loader, epochs,
            mse_s, star_s, device, st_kwargs, save_model_dir,
            grid_x=grid_x, grid_y=grid_y,
            beta_softmax=beta_softmax, tau_sigmoid=tau_sigmoid, eps_abs=eps_abs,
            penalty_kappa=penalty_kappa,
        )
        samples = sample_flow_model(
            model, opt, num_new, batch_n, points_N, device,
            cond_loader=test_loader,
            grid_x=grid_x, grid_y=grid_y,
            beta_softmax=beta_softmax, tau_sigmoid=tau_sigmoid, eps_abs=eps_abs,
            polish_steps=polish_steps, polish_lr=polish_lr,
            n_steps=gas_steps, ode_method=ode_method, ode_step_cap=ode_step_cap,
            proj_steps=proj_steps, proj_lr=proj_lr,
            prox_steps=prox_steps, prox_lr=prox_lr, prox_lambda=prox_lambda,
            de_novo_disc_bump=de_novo_disc_bump,
            tau_sigmoid_start=tau_sigmoid_start, tau_sigmoid_end=tau_sigmoid_end,
            hardmax_topk=hardmax_topk, hardmax_steps=hardmax_steps, hardmax_lr_frac=hardmax_lr_frac,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(save_generated_dir, f"star_gen_{num_new}x{points_N}_{ts}.pt")
    torch.save(torch.from_numpy(samples), out_path)
    print(f"Saved {num_new} generated pointsets to {out_path}")
    print(f"(Model path: {model_path})")

    if plot_k > 0:
        plot_top_k_lowdiscrepancy_samples(samples, plot_k, plot_dir, filename_prefix="star_lowdisc")
        print(f"Saved plots:   {plot_dir}")
    
    if state is not None:
        state.set_model_path(model_path)
        state.set_samples_path(out_path)

    return model_path, out_path

if __name__ == "__main__":
    main()
