"""
FastAPI router that serves the trained MNIST GAN generator.

Add to your existing app/main.py:

    from app.gan_endpoints import router as gan_router
    app.include_router(gan_router)

Endpoints:
    GET /gan/generate?num_images=16&seed=42   -> PNG grid of generated digits
    GET /gan/generate_json?num_images=4       -> JSON with pixel arrays (values 0-255)
"""
import io
import os
from typing import Optional

import torch
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from torchvision.utils import make_grid, save_image

from app.gan_model import LATENT_DIM, Generator

MODEL_PATH = os.getenv("GAN_MODEL_PATH", "models/gan_generator.pth")

router = APIRouter(prefix="/gan", tags=["GAN"])

_generator: Optional[Generator] = None


def get_generator() -> Generator:
    """Load the generator once, on first request, and reuse it."""
    global _generator
    if _generator is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail=f"GAN weights not found at {MODEL_PATH}. Run train_gan.py first.",
            )
        g = Generator()
        g.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        g.eval()  # important: BatchNorm uses running stats at inference
        _generator = g
    return _generator


def _sample(num_images: int, seed: Optional[int]) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    z = torch.randn(num_images, LATENT_DIM)
    with torch.no_grad():
        imgs = get_generator()(z)       # (N, 1, 28, 28), range [-1, 1]
    return (imgs + 1) / 2               # rescale to [0, 1]


@router.get("/generate")
def generate_digits(
    num_images: int = Query(16, ge=1, le=64, description="How many digits to generate"),
    seed: Optional[int] = Query(None, description="Optional seed for reproducible output"),
):
    imgs = _sample(num_images, seed)
    nrow = min(8, num_images)
    buf = io.BytesIO()
    save_image(make_grid(imgs, nrow=nrow, padding=2), buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@router.get("/generate_json")
def generate_digits_json(
    num_images: int = Query(4, ge=1, le=16),
    seed: Optional[int] = Query(None),
):
    imgs = _sample(num_images, seed)
    pixels = (imgs.squeeze(1) * 255).round().to(torch.uint8).tolist()  # (N, 28, 28)
    return {"num_images": num_images, "shape": [28, 28], "images": pixels}
