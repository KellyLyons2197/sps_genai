from fastapi import FastAPI, File, HTTPException, UploadFile 
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel  # NEW
from app.image_classifier import ImageClassifier  # NEW

app = FastAPI()
image_classifier = ImageClassifier()  # NEW

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel()  # NEW


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):  # NEW
    text: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embedding")  # NEW
def get_embedding(request: EmbeddingRequest):
    result = embedding_model.get_embedding(request.text)
    if not result["has_vector"]:
        raise HTTPException(status_code=404, detail=f"No vector found for '{request.text}'")
    return result

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        result = image_classifier.predict(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the file as an image")
    return {"filename": file.filename, **result}