# SPS GenAI – 
# ----------------------
# Assignment 1

FastAPI app with a bigram text generator (from Module 2) and spaCy word embeddings.

## Endpoints

| Method | Endpoint | Description | Example request body |
|---|---|---|---|
| GET | `/` | Health check | none |
| POST | `/generate` | Generate text using bigram probabilities | `{"start_word": "bigram", "length": 5}` |
| POST | `/embedding` | Return the spaCy word embedding (300 dimensions) for a word or phrase | `{"text": "king"}` |

`/embedding` returns a 404 error if spaCy has no vector for the input (for example, a made-up word).

## Run with Docker

From the project root:

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

Then open http://127.0.0.1:8000/docs to try the endpoints.

## Run locally with uv

```bash
uv sync
uv run fastapi dev app/main.py
```

Then open http://127.0.0.1:8000/docs.

## Project structure

- `app/main.py` – FastAPI app and endpoint definitions
- `app/bigram_model.py` – `BigramModel` class, adapted from the Module 2 word sampling notebook
- `app/embedding_model.py` – `EmbeddingModel` class that loads spaCy's `en_core_web_md` model
- `Dockerfile` – container setup using uv
- `pyproject.toml` / `uv.lock` – dependencies (FastAPI, spaCy, and the `en_core_web_md` model)

# -------------------------
# Assignment 2: CNN Image Classifier

A CNN (`SimpleCNN`, `app/cnn_model.py`) trained on CIFAR-10 and served through a new `/classify` endpoint.

**Architecture:** 64×64×3 input → Conv2D(16, 3×3, stride 1, padding 1) → ReLU → MaxPool(2×2, stride 2) → Conv2D(32, 3×3, stride 1, padding 1) → ReLU → MaxPool(2×2, stride 2) → Flatten (8192) → FC(100) → ReLU → FC(10)

**Training:** `train_cnn.ipynb`, 10 epochs, Adam (lr 0.0005), CIFAR-10 images resized from 32×32 to 64×64. Test accuracy: **66.23%**. Weights are saved in `app/models/cnn_cifar10.pth`, so no retraining is needed to run the API.

## New endpoint

| Method | Endpoint | Description | Request |
|---|---|---|---|
| POST | `/classify` | Classify an uploaded image into one of the 10 CIFAR-10 classes | multipart form, field `file` = an image (.jpg/.png) |

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## How to test

After `docker build -t sps-genai .` and `docker run -p 8000:80 sps-genai`:

- **Browser:** open http://127.0.0.1:8000/docs → POST /classify → Try it out → choose an image → Execute
- **curl:**

```bash
curl -X POST http://127.0.0.1:8000/classify -F "file=@cat.jpg"
```

Example response:

```json
{"filename": "cat.jpg", "predicted_class": "cat", "confidence": 0.726}
```

## Files added

- `app/cnn_model.py`: `SimpleCNN` architecture
- `app/image_classifier.py`: loads the trained weights and predicts the class of an uploaded image
- `app/models/cnn_cifar10.pth`: trained weights
- `train_cnn.ipynb`: training and evaluation notebook

# ------------------------------------
## Assignment 3: GAN for MNIST Digit Generation

A Generative Adversarial Network (GAN), built in PyTorch, that generates handwritten digits. It was trained on the MNIST dataset and is served through FastAPI.

### Architecture

**Generator** (`app/gan_model.py`)
- Input: noise vector of shape (batch_size, 100)
- Fully connected layer → 7 × 7 × 128, then reshaped
- ConvTranspose2d (128 → 64, kernel 4, stride 2, padding 1) → 14 × 14, BatchNorm2d, ReLU
- ConvTranspose2d (64 → 1, kernel 4, stride 2, padding 1) → 28 × 28, Tanh

**Discriminator** (`app/gan_model.py`)
- Input: image of shape (1, 28, 28)
- Conv2d (1 → 64, kernel 4, stride 2, padding 1) → 14 × 14, LeakyReLU(0.2)
- Conv2d (64 → 128, kernel 4, stride 2, padding 1) → 7 × 7, BatchNorm2d, LeakyReLU(0.2)
- Flatten → Linear → Sigmoid (real/fake probability)

### Files
| File | Purpose |
|---|---|
| `app/gan_model.py` | Generator and Discriminator classes |
| `train_gan.py` | Trains the GAN on MNIST and saves the weights |
| `app/gan_endpoints.py` | FastAPI router with the GAN endpoints |
| `models/gan_generator.pth` | Trained generator weights, loaded by the API |

### Training
```bash
uv run python train_gan.py --epochs 20
```
- Loss: binary cross-entropy (BCELoss); optimizer: Adam (lr = 2e-4, betas = (0.5, 0.999)); batch size 128
- Images are normalized to [-1, 1] to match the generator's Tanh output
- Saves weights to `models/`, plus a sample grid for each epoch in `samples/`

### API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| GET | `/gan/generate?num_images=16&seed=42` | Returns a PNG grid of generated digits (1–64 images) |
| GET | `/gan/generate_json?num_images=4` | Returns the generated digits as 28 × 28 pixel arrays in JSON (1–16 images) |

`seed` is optional. Using the same seed returns the same digits.

### Run with Docker
```bash
docker build -t sps_genai .
docker run -p 8000:80 sps_genai
```
Then open http://localhost:8000/docs and try **GET /gan/generate**.

Or from the terminal:
```bash
curl -o digits.png "http://localhost:8000/gan/generate?num_images=16&seed=42"
```