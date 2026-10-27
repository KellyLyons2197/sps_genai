"""
Assignment 3 - GAN architecture for 28x28 MNIST digits.

Generator:     z (B, 100) -> FC -> (B, 128, 7, 7) -> ConvT -> (B, 64, 14, 14) -> ConvT -> (B, 1, 28, 28)
Discriminator: x (B, 1, 28, 28) -> Conv -> (B, 64, 14, 14) -> Conv -> (B, 128, 7, 7) -> Linear -> (B, 1)
"""
import torch
import torch.nn as nn

LATENT_DIM = 100


class Generator(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        # Fully connected layer to 7 x 7 x 128 (reshaped in forward)
        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)

        self.net = nn.Sequential(
            # 128 -> 64, 7x7 -> 14x14   ((7-1)*2 - 2*1 + 4 = 14)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64 -> 1, 14x14 -> 28x28   ((14-1)*2 - 2*1 + 4 = 28)
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # outputs in [-1, 1], matching normalized MNIST
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)                  # (B, 6272)
        x = x.view(-1, 128, 7, 7)       # reshape -> (B, 128, 7, 7)
        return self.net(x)              # (B, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 1 -> 64, 28x28 -> 14x14   ((28 + 2*1 - 4)/2 + 1 = 14)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 -> 128, 14x14 -> 7x7   ((14 + 2*1 - 4)/2 + 1 = 7)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),                       # (B, 128*7*7)
            nn.Linear(128 * 7 * 7, 1),          # single output
            nn.Sigmoid(),                       # real/fake probability in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)              # (B, 1)


def weights_init(m: nn.Module) -> None:
    """DCGAN-style init: N(0, 0.02) for conv weights, N(1, 0.02) for BatchNorm."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


if __name__ == "__main__":
    # Quick shape check: python -m app.gan_model
    G, D = Generator(), Discriminator()
    z = torch.randn(8, LATENT_DIM)
    fake = G(z)
    print("Generator output:", tuple(fake.shape))        # (8, 1, 28, 28)
    print("Discriminator output:", tuple(D(fake).shape)) # (8, 1)
