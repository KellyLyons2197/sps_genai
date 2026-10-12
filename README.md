# SPS GenAI – Assignment 1

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

# Assignment 2

| POST | `/classify` | Classify an uploaded image into one of 10 CIFAR-10 classes using a CNN | multipart form, field `file` = image |