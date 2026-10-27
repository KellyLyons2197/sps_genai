"""
Train the Assignment 3 GAN on MNIST and save the generator weights.

Run from the repo root:
    uv run python train_gan.py --epochs 20

Outputs:
    models/gan_generator.pth      <- loaded by the FastAPI endpoint
    models/gan_discriminator.pth
    samples/epoch_XX.png          <- 8x8 grid of generated digits per epoch
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from app.gan_model import LATENT_DIM, Discriminator, Generator, weights_init


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():   # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = get_device()
    print(f"Training on {device}")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    # Normalize to [-1, 1] so real images match the generator's Tanh range
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    G = Generator().to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, LATENT_DIM, device=device)  # same z each epoch to watch progress

    for epoch in range(1, args.epochs + 1):
        for i, (real, _) in enumerate(loader):
            real = real.to(device)
            b = real.size(0)
            real_labels = torch.ones(b, 1, device=device)
            fake_labels = torch.zeros(b, 1, device=device)

            # ---- 1. Train Discriminator: maximize log D(x) + log(1 - D(G(z))) ----
            z = torch.randn(b, LATENT_DIM, device=device)
            fake = G(z)
            loss_D_real = criterion(D(real), real_labels)
            loss_D_fake = criterion(D(fake.detach()), fake_labels)  # detach: don't update G here
            loss_D = loss_D_real + loss_D_fake
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ---- 2. Train Generator: maximize log D(G(z)) (non-saturating loss) ----
            loss_G = criterion(D(fake), real_labels)  # G wants D to call fakes "real"
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            if i % 200 == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Step [{i}/{len(loader)}] "
                      f"Loss_D: {loss_D.item():.4f}  Loss_G: {loss_G.item():.4f}")

        # Save a sample grid each epoch
        G.eval()
        with torch.no_grad():
            samples = G(fixed_noise)
        save_image(samples, f"samples/epoch_{epoch:02d}.png", nrow=8, normalize=True, value_range=(-1, 1))
        G.train()

    torch.save(G.state_dict(), os.path.join(args.out_dir, "gan_generator.pth"))
    torch.save(D.state_dict(), os.path.join(args.out_dir, "gan_discriminator.pth"))
    print(f"Saved weights to {args.out_dir}/")


if __name__ == "__main__":
    main()
