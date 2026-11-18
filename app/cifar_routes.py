"""FastAPI endpoints serving the CIFAR-10 diffusion and energy models.

Add to app/main.py:
    from app.cifar_routes import router as cifar_router
    app.include_router(cifar_router)
"""
import io
from functools import lru_cache
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from torchvision.utils import make_grid
from PIL import Image

from app.cifar_models import Diffusion, EnergyModel, UNet, langevin_sample

router = APIRouter(tags=["cifar10"])
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load(model, filename):
    path = MODEL_DIR / filename
    if not path.exists():
        raise HTTPException(503, f"Weights not found at {path}. Run train_cifar.py first.")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE).eval()


@lru_cache
def get_unet():
    return _load(UNet(), "diffusion_cifar10.pt")


@lru_cache
def get_energy_model():
    return _load(EnergyModel(), "energy_cifar10.pt")


def to_png(images: torch.Tensor) -> bytes:
    """(N,3,32,32) in [-1,1] -> PNG grid bytes."""
    grid = make_grid((images + 1) / 2, nrow=min(len(images), 8), padding=2)
    arr = (grid.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).resize((arr.shape[1] * 4, arr.shape[0] * 4), Image.NEAREST).save(buf, format="PNG")
    return buf.getvalue()


@router.get("/diffusion/generate", responses={200: {"content": {"image/png": {}}}})
def generate_diffusion(
    n: int = Query(8, ge=1, le=32, description="Number of images"),
    steps: int = Query(50, ge=10, le=1000, description="DDIM steps (1000 = full DDPM)"),
):
    images = Diffusion(device=DEVICE).sample(get_unet(), n=n, steps=steps)
    return Response(to_png(images), media_type="image/png")


@router.get("/energy/generate", responses={200: {"content": {"image/png": {}}}})
def generate_energy(
    n: int = Query(8, ge=1, le=32, description="Number of images"),
    steps: int = Query(256, ge=10, le=1000, description="Langevin steps"),
):
    x0 = torch.rand(n, 3, 32, 32, device=DEVICE) * 2 - 1  # start from uniform noise
    images = langevin_sample(get_energy_model(), x0, steps=steps)
    return Response(to_png(images), media_type="image/png")
