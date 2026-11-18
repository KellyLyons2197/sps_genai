"""CIFAR-10 generative models for Assignment 4: a DDPM-style diffusion model and an energy-based model."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Diffusion model
# ---------------------------------------------------------------------------
def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Map integer timesteps (B,) to sinusoidal embeddings (B, dim): [sin(t*w_i), cos(t*w_i)]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    """Two 3x3 convs with GroupNorm/SiLU; the time embedding is added as a per-channel bias."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """Small UNet for 32x32 images: 32 -> 16 -> 8 (bottleneck) -> 16 -> 32. Predicts the noise eps."""

    def __init__(self, ch: int = 64, t_dim: int = 256):
        super().__init__()
        self.ch = ch
        self.time_mlp = nn.Sequential(nn.Linear(ch, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))

        self.inc = nn.Conv2d(3, ch, 3, padding=1)
        self.down1 = ResBlock(ch, ch, t_dim)                              # 32x32
        self.pool1 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)            # -> 16x16
        self.down2 = ResBlock(ch, ch * 2, t_dim)                          # 16x16
        self.pool2 = nn.Conv2d(ch * 2, ch * 2, 3, stride=2, padding=1)    # -> 8x8
        self.mid1 = ResBlock(ch * 2, ch * 4, t_dim)                       # bottleneck 8x8
        self.mid2 = ResBlock(ch * 4, ch * 4, t_dim)
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1)  # -> 16x16
        self.dec2 = ResBlock(ch * 4, ch * 2, t_dim)                       # concat with skip
        self.up1 = nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1)      # -> 32x32
        self.dec1 = ResBlock(ch * 2, ch, t_dim)
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, 3, 3, padding=1))

    def forward(self, x, t):
        temb = self.time_mlp(sinusoidal_embedding(t, self.ch))
        x = self.inc(x)
        s1 = self.down1(x, temb)
        s2 = self.down2(self.pool1(s1), temb)
        h = self.mid2(self.mid1(self.pool2(s2), temb), temb)
        h = self.dec2(torch.cat([self.up2(h), s2], dim=1), temb)
        h = self.dec1(torch.cat([self.up1(h), s1], dim=1), temb)
        return self.out(h)


class Diffusion:
    """DDPM noise schedule, training loss, and samplers (full DDPM or faster DDIM)."""

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02, device="cpu"):
        self.T = T
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise):
        """Forward process: x_t = sqrt(a_bar_t) * x0 + sqrt(1 - a_bar_t) * eps."""
        ab = self.alpha_bars[t][:, None, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def loss(self, model, x0):
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        return F.mse_loss(model(self.q_sample(x0, t, noise), t), noise)

    @torch.no_grad()
    def sample(self, model, n: int = 8, steps: int = 50):
        """steps == T runs ancestral DDPM sampling; fewer steps uses deterministic DDIM (much faster)."""
        model.eval()
        x = torch.randn(n, 3, 32, 32, device=self.device)
        if steps >= self.T:
            for i in reversed(range(self.T)):
                t = torch.full((n,), i, device=self.device, dtype=torch.long)
                eps = model(x, t)
                a, ab, b = self.alphas[i], self.alpha_bars[i], self.betas[i]
                mean = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
                x = mean + b.sqrt() * torch.randn_like(x) if i > 0 else mean
        else:
            ts = torch.linspace(self.T - 1, 0, steps, device=self.device).long()
            for k, i in enumerate(ts):
                t = torch.full((n,), int(i), device=self.device, dtype=torch.long)
                eps = model(x, t)
                ab = self.alpha_bars[i]
                ab_prev = self.alpha_bars[ts[k + 1]] if k + 1 < len(ts) else torch.tensor(1.0, device=self.device)
                x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
                x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return x.clamp(-1, 1)


# ---------------------------------------------------------------------------
# Energy-based model
# ---------------------------------------------------------------------------
class EnergyModel(nn.Module):
    """CNN mapping an image to a scalar energy E(x). Low energy = realistic image."""

    def __init__(self, ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 5, stride=2, padding=2), nn.SiLU(),            # 16x16
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1), nn.SiLU(),       # 8x8
            nn.Conv2d(ch * 2, ch * 4, 3, stride=2, padding=1), nn.SiLU(),   # 4x4
            nn.Conv2d(ch * 4, ch * 4, 3, stride=2, padding=1), nn.SiLU(),   # 2x2
            nn.Flatten(),
            nn.Linear(ch * 4 * 4, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def langevin_sample(model, x, steps: int = 60, step_size: float = 10.0, noise_std: float = 0.005):
    """Langevin dynamics: walk x downhill on E(x) using grad wrt the *input*, plus small noise."""
    was_training = model.training
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    x = x.clone().detach().requires_grad_(True)
    with torch.enable_grad():  # works even if called inside torch.no_grad()
        for _ in range(steps):
            x.data.add_(noise_std * torch.randn_like(x)).clamp_(-1, 1)
            energy = model(x).sum()
            (grad,) = torch.autograd.grad(energy, x)
            x.data.add_(-step_size * grad.clamp(-0.03, 0.03)).clamp_(-1, 1)

    for p in model.parameters():
        p.requires_grad_(True)
    model.train(was_training)
    return x.detach()


class ReplayBuffer:
    """Keeps past Langevin samples so chains continue across batches (95% buffer, 5% fresh noise)."""

    def __init__(self, size: int = 8192, device="cpu"):
        self.device = device
        self.examples = torch.rand(size, 3, 32, 32, device=device) * 2 - 1

    def sample_new(self, model, batch_size: int, steps: int = 60):
        n_fresh = max(1, int(0.05 * batch_size))
        fresh = torch.rand(n_fresh, 3, 32, 32, device=self.device) * 2 - 1
        idx = torch.randint(0, len(self.examples), (batch_size - n_fresh,), device=self.device)
        x = torch.cat([fresh, self.examples[idx]], dim=0)
        x = langevin_sample(model, x, steps=steps)
        self.examples = torch.cat([x, self.examples], dim=0)[: len(self.examples)]
        return x


def energy_loss(model, real, fake, alpha: float = 0.1):
    """Contrastive divergence: push E(real) down, E(fake) up, with an L2 term to keep energies bounded."""
    e_real, e_fake = model(real), model(fake)
    cdiv = e_real.mean() - e_fake.mean()
    reg = alpha * (e_real.pow(2) + e_fake.pow(2)).mean()
    return cdiv + reg, cdiv.item()
