from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_policy_sections(sections: list[dict]) -> list[dict]:
    """Split each policy section into overlapping, metadata-preserving chunks."""
    # Prefer paragraph and sentence boundaries before splitting at words/characters.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[dict] = []
    for section in sections:
        # Empty sections cannot contribute useful retrieval evidence.
        section_text = (section.get("content") or "").strip()
        if not section_text:
            continue
        section_chunks = splitter.split_text(section_text)
        for index, chunk_text in enumerate(section_chunks):
            # Keep the source location with each chunk for retrieval and citations.
            chunk_meta = {
                "source": section["source"],
                "page": section.get("page", 1),
                "pages": section.get("pages", [section.get("page", 1)]),
                "section_id": section["section_id"],
                "section_title": section.get("section_title", ""),
                "chunk_id": f"{section['section_id']}-{index + 1}",
                "text": chunk_text.strip(),
            }
            chunks.append(chunk_meta)
    return chunks
