"""Train the CIFAR-10 diffusion or energy model and save weights for the API.

    uv run python train_cifar.py --model diffusion --epochs 30
    uv run python train_cifar.py --model energy --epochs 20
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.cifar_models import Diffusion, EnergyModel, ReplayBuffer, UNet, energy_loss

MODEL_DIR = Path("models")


def get_loader(batch_size: int):
    tfm = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),  # scale to [-1, 1]
    ])
    ds = datasets.CIFAR10("data", train=True, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)


def train_diffusion(epochs, batch_size, lr, device):
    model = UNet().to(device)
    diffusion = Diffusion(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, _ in get_loader(batch_size):
            loss = diffusion.loss(model, x.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch + 1}/{epochs}  loss {total / (50000 // batch_size):.4f}")
        torch.save(model.state_dict(), MODEL_DIR / "diffusion_cifar10.pt")


def train_energy(epochs, batch_size, lr, device):
    model = EnergyModel().to(device)
    buffer = ReplayBuffer(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.0, 0.999))
    for epoch in range(epochs):
        model.train()
        for step, (x, _) in enumerate(get_loader(batch_size)):
            real = (x.to(device) + 0.005 * torch.randn_like(x.to(device))).clamp(-1, 1)
            fake = buffer.sample_new(model, batch_size)
            loss, cdiv = energy_loss(model, real, fake)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 100 == 0:
                print(f"epoch {epoch + 1}  step {step}  loss {loss.item():.4f}  cdiv {cdiv:.4f}")
        torch.save(model.state_dict(), MODEL_DIR / "energy_cifar10.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["diffusion", "energy"], required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    MODEL_DIR.mkdir(exist_ok=True)
    if args.model == "diffusion":
        train_diffusion(args.epochs, args.batch_size, args.lr or 2e-4, device)
    else:
        train_energy(args.epochs, args.batch_size, args.lr or 1e-4, device)
