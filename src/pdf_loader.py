from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pymupdf

from src.config import EXPECTED_SECTION_IDS

# Policy section headings use IDs such as AS01 and PL02 followed by their title.
SECTION_PATTERN = re.compile(r"^(AS\d{2}|PL\d{2})\s+(.+)$", re.IGNORECASE)


def extract_page_sections(page_text: str, page_number: int) -> list[dict[str, Any]]:
    """Parse identified policy sections from one extracted PDF page."""
    sections: list[dict[str, Any]] = []
    current_id: str | None = None
    current_title: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        # Save the accumulated lines whenever a section ends or a new one begins.
        nonlocal current_id, current_title, current_lines
        if current_id and current_title:
            sections.append(
                {
                    "section_id": current_id.upper(),
                    "section_title": current_title.strip(),
                    "page": page_number,
                    "text": "\n".join(current_lines).strip(),
                }
            )
        current_id = None
        current_title = None
        current_lines = []

    for line in page_text.splitlines():
        # A heading starts a new section; other text belongs to the current section.
        match = SECTION_PATTERN.match(line.strip())
        if match:
            flush()
            current_id = match.group(1).upper()
            current_title = match.group(2).strip()
            continue
        if current_id:
            current_lines.append(line)
    flush()
    return sections


def extract_pdf_sections(pdf_path: str | Path) -> tuple[int, list[dict[str, Any]], list[str]]:
    """Extract, merge, and validate labelled policy sections from a PDF."""
    doc = pymupdf.open(str(pdf_path))
    page_count = doc.page_count
    section_index: OrderedDict[str, dict[str, Any]] = OrderedDict()

    for page_number, page in enumerate(doc, start=1):
        # A section may continue across pages, so merge content by its section ID.
        page_text = page.get_text("text")
        found_sections = extract_page_sections(page_text, page_number)
        for section in found_sections:
            section_id = section["section_id"]
            if section_id not in section_index:
                section_index[section_id] = {
                    "section_id": section_id,
                    "section_title": section["section_title"],
                    "pages": [page_number],
                    "content": section["text"],
                }
            else:
                section_index[section_id]["pages"].append(page_number)
                section_index[section_id]["content"] += "\n" + section["text"]

    # Convert the internal ordered mapping into the common section-data format.
    sections = [
        {
            "section_id": section_id,
            "section_title": meta["section_title"],
            "pages": meta["pages"],
            "page": meta["pages"][0],
            "content": meta["content"].strip(),
            "source": Path(pdf_path).name,
        }
        for section_id, meta in section_index.items()
    ]

    found_ids = [section["section_id"] for section in sections]
    missing_sections = [sid for sid in EXPECTED_SECTION_IDS if sid not in found_ids]

    return page_count, sections, missing_sections


def validate_loaded_pdfs(pdf_paths: list[str | Path]) -> tuple[dict, list[str]]:
    """Report document-level and combined coverage of required policy sections."""
    validation_report = {}
    combined_found: set[str] = set()
    combined_missing: set[str] = set(EXPECTED_SECTION_IDS)

    for path in pdf_paths:
        # Record per-file validation while accumulating cross-document section coverage.
        file_name = Path(path).name
        page_count, sections, missing = extract_pdf_sections(path)
        found_ids = [section["section_id"] for section in sections]
        combined_found.update(found_ids)
        validation_report[file_name] = {
            "page_count": page_count,
            "sections_found": found_ids,
            "missing_sections": missing,
            "section_count": len(sections),
        }

    combined_missing = set(EXPECTED_SECTION_IDS) - combined_found
    overall_report = {
        "combined_sections_found": sorted(combined_found),
        "combined_missing_sections": sorted(combined_missing),
        "expected_section_count": len(EXPECTED_SECTION_IDS),
        "actual_section_count": len(combined_found),
        "validation_report": validation_report,
    }
    return overall_report, sorted(combined_missing)
