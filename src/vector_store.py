from __future__ import annotations

import numpy as np
import faiss


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build an exact inner-product FAISS index over normalized embeddings."""
    if embeddings.size == 0:
        raise ValueError("No embeddings available to build the FAISS index.")
    # FAISS expects float32 vectors; reshape a single embedding to a matrix.
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    # Normalize stored vectors so dot products represent cosine similarity.
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    normalized = matrix / norm
    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)
    return index


def search_faiss(index: faiss.IndexFlatIP, query_embedding: np.ndarray, top_k: int = 3):
    """Return the top-k scores and positions for one query embedding."""
    # Apply the same normalization used when the index was built.
    query_vector = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    normalized_query = query_vector / norm
    scores, indices = index.search(normalized_query, top_k)
    return scores[0], indices[0]
