import spacy


class EmbeddingModel:
    def __init__(self, model_name="en_core_web_md"):
        self.nlp = spacy.load(model_name)

    def get_embedding(self, text):
        doc = self.nlp(text)
        return {
            "text": text,
            "has_vector": doc.has_vector,
            "dimensions": int(doc.vector.shape[0]),
            "embedding": doc.vector.tolist(),
        }