from __future__ import annotations

import numpy as np

from src.embeddings import embed_texts, load_embedding_model


def retrieve_top_chunks(chunks: list[dict], query: str, index, model, top_k: int = 3):
    """Embed one user query and return its top-k most similar policy chunks."""
    # This is single-query dense retrieval: the original question is embedded once.
    query_embedding = embed_texts(model, [query])[0]
    scores, indices = index.search(np.asarray([query_embedding], dtype=np.float32), top_k)
    result = []
    for rank, idx in enumerate(indices[0], start=1):
        # FAISS can return a negative placeholder when no vector is available.
        if idx < 0:
            continue
        chunk = chunks[int(idx)]
        # Return chunk text plus its provenance for answer generation and UI display.
        result.append(
            {
                "rank": rank,
                "score": float(scores[0][rank - 1]),
                "source": chunk.get("source"),
                "page": chunk.get("page"),
                "pages": chunk.get("pages", [chunk.get("page")]),
                "section_id": chunk.get("section_id"),
                "section_title": chunk.get("section_title"),
                "chunk_id": chunk.get("chunk_id"),
                "text": chunk.get("text", ""),
            }
        )
    return result
