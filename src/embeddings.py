from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL


def load_embedding_model(model_name: str | None = None):
    """Load the sentence-transformer used for document and query embeddings."""
    model_name = model_name or EMBEDDING_MODEL
    return SentenceTransformer(model_name)


def embed_texts(model, texts: list[str]) -> np.ndarray:
    """Convert text into normalized float32 vectors for cosine-similarity search."""
    # Normalization makes inner-product search equivalent to cosine similarity.
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)
