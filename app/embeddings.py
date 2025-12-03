from typing import List
from sentence_transformers import SentenceTransformer

_model = None


def get_embedding_model() -> SentenceTransformer:
    """
    Lazy-load and cache the embedding model.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def embed_texts(texts: List[str]):
    """
    Encode a list of texts into vectors.
    """
    model = get_embedding_model()
    return model.encode(texts, batch_size=32, show_progress_bar=False)