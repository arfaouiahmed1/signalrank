from functools import lru_cache
from typing import List
import numpy as np

from app.config import get_settings

@lru_cache(maxsize=1)
def get_embed_model():
    from sentence_transformers import SentenceTransformer
    s = get_settings()
    return SentenceTransformer(s.embed_model)

def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embed_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
