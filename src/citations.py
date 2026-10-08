from __future__ import annotations


def format_citation(source: str, page: int | str, section_id: str) -> str:
    """Return a consistently formatted policy-document citation."""
    return f"[Source: {source}, Page: {page}, Section: {section_id}]"


def unique_citations(metadata_list: list[dict]) -> list[str]:
    """Build citations once per distinct source, page, and section."""
    seen = set()
    citations: list[str] = []
    for item in metadata_list:
        # A set preserves uniqueness while the list retains display order.
        citation = format_citation(item.get("source"), item.get("page"), item.get("section_id"))
        if citation not in seen:
            seen.add(citation)
            citations.append(citation)
    return citations
