from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.chunker import chunk_policy_sections
from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, EMBEDDING_MODEL, EXPECTED_SECTION_IDS, PROJECT_ROOT, TOP_K
from src.embeddings import embed_texts, load_embedding_model
from src.generator import generate_answer
from src.pdf_loader import extract_pdf_sections, validate_loaded_pdfs
from src.retriever import retrieve_top_chunks
from src.vector_store import build_faiss_index

# Configure Streamlit before any UI elements are created.
st.set_page_config(page_title="Banking RAG Policy Assistant", page_icon="🏦", layout="wide")


def ensure_pdf_files_exist() -> list[Path]:
    """Return the PDF files currently stored in the application's data folder."""
    data_files = sorted(DATA_DIR.glob("*.pdf"))
    return data_files


@st.cache_resource
def get_embedding_model():
    """Load the embedding model once per Streamlit session/resource cache."""
    return load_embedding_model(EMBEDDING_MODEL)


def save_uploaded_pdfs(uploaded_files):
    """Copy user-selected policy PDFs into the local data directory."""
    saved = []
    for uploaded_file in uploaded_files:
        target = DATA_DIR / uploaded_file.name
        target.write_bytes(uploaded_file.read())
        saved.append(target)
    return saved


def build_knowledge_base():
    """Validate PDFs, create chunks and embeddings, then build the FAISS index."""
    files = ensure_pdf_files_exist()
    if len(files) != 2:
        st.error("Please upload exactly two policy PDFs to the data directory before building the knowledge base.")
        return False

    # Validate expected policy-section coverage before retrieval is enabled.
    overall_validation, missing = validate_loaded_pdfs(files)
    validation_report = overall_validation.get("validation_report", {})
    rows = []
    all_sections = []
    for pdf_path in files:
        # Extract labelled sections from each source document.
        page_count, sections, section_missing = extract_pdf_sections(pdf_path)
        rows.extend(sections)
        all_sections.extend(sections)
        if section_missing:
            st.warning(f"{pdf_path.name}: missing sections {section_missing}")
        if len(rows) >= 1 and len(rows) < 16:
            st.warning("Expected 16 section IDs but fewer were detected; verification report is available below.")

    if overall_validation.get("actual_section_count", 0) < 16:
        st.warning(f"Overall section validation: expected 16 IDs, found {overall_validation.get('actual_section_count', 0)}. Missing: {overall_validation.get('combined_missing_sections', [])}")

    # Build the retrieval pipeline: chunks -> normalized vectors -> FAISS index.
    chunks = chunk_policy_sections(all_sections)
    model = get_embedding_model()
    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(model, chunk_texts)
    index = build_faiss_index(embeddings)

    # Keep the knowledge base in session state for subsequent user questions.
    st.session_state["kb_built"] = True
    st.session_state["chunks"] = chunks
    st.session_state["index"] = index
    st.session_state["model"] = model
    st.session_state["validation_report"] = validation_report
    st.session_state["missing_sections"] = missing
    st.session_state["chunk_count"] = len(chunks)
    st.session_state["section_count"] = len(all_sections)
    st.session_state["page_count"] = sum(report["page_count"] for report in validation_report.values())
    st.success("Knowledge base built successfully.")
    return True


# Main application interface: upload, build the knowledge base, then ask questions.
st.title("Banking RAG Policy Assistant")

with st.sidebar:
    # The sidebar owns document ingestion and retrieval-configuration visibility.
    st.subheader("Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload the two policy PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        # Show whether the two required source document names were supplied.
        names = {file.name for file in uploaded_files}
        if "Accounts_and_Service_Policy.pdf" in names:
            st.write("✓ Accounts policy loaded")
        else:
            st.write("✗ Accounts policy missing")
        if "Personal_Loan_Policy.pdf" in names:
            st.write("✓ Personal loan policy loaded")
        else:
            st.write("✗ Personal loan policy missing")
        if len(uploaded_files) == 2:
            if st.button("Load selected PDFs into data folder"):
                save_uploaded_pdfs(uploaded_files)
                st.success("Files copied to the data folder.")
    if st.button("Build Knowledge Base"):
        build_knowledge_base()

    if st.session_state.get("kb_built"):
        # Display the active retrieval settings for transparency and reproducibility.
        st.markdown("### Knowledge base status")
        st.write(f"Sections discovered: {st.session_state.get('section_count', 0)}")
        st.write(f"Chunks created: {st.session_state.get('chunk_count', 0)}")
        st.write(f"Embedding model: {EMBEDDING_MODEL}")
        st.write(f"Chunk size: {CHUNK_SIZE}")
        st.write(f"Chunk overlap: {CHUNK_OVERLAP}")
        st.write(f"Top-k: {TOP_K}")
        st.write("Vector index: FAISS IndexFlatIP")

if not st.session_state.get("kb_built"):
    # Questions cannot be answered before the in-memory vector index exists.
    st.info("Please upload both policy PDFs and build the knowledge base first.")
    st.stop()

question = st.text_area("Question", height=150)
if st.button("ASK"):
    if not question.strip():
        st.warning("Please enter a policy question.")
    else:
        index = st.session_state.get("index")
        chunks = st.session_state.get("chunks")
        model = st.session_state.get("model")
        if index is None or chunks is None or model is None:
            st.warning("Please upload both policy PDFs and build the knowledge base first.")
        else:
            # Embed one question and retrieve the most similar policy chunks.
            retrieved = retrieve_top_chunks(chunks, question, index, model, top_k=TOP_K)
            if not retrieved:
                st.warning("No supporting policy passages were found.")
            else:
                # Safety checks and LLM/fallback generation occur inside this function.
                result = generate_answer(question, retrieved)
                answer = result["answer"]
                st.subheader("Answer")
                st.write(answer)

                st.subheader("Sources")
                unique_sources = []
                seen = set()
                for passage in retrieved:
                    # Deduplicate the compact source list while retaining all passage detail below.
                    key = (passage["source"], passage["page"], passage["section_id"])
                    if key not in seen:
                        seen.add(key)
                        unique_sources.append(passage)
                for item in unique_sources:
                    st.write(f"- {item['source']} | Page {item['page']} | Section {item['section_id']}")

                st.subheader("Retrieved Supporting Passages")
                for passage in retrieved:
                    # Show the exact evidence used for transparent, auditable answers.
                    with st.expander(f"Rank {passage['rank']} | {passage['source']} | Page {passage['page']} | Section {passage['section_id']} | Score {passage['score']:.3f}"):
                        st.write(f"Section title: {passage.get('section_title', '')}")
                        st.write(passage["text"])

validation_report = st.session_state.get("validation_report")
if validation_report:
    # Surface precomputed offline evaluation metrics and PDF-validation details.
    st.subheader("Evaluation")
    st.write("Offline evaluation results")
    scorecard_path = PROJECT_ROOT / "evaluation" / "scorecard.json"
    csv_path = PROJECT_ROOT / "evaluation" / "question_level_results.csv"
    if scorecard_path.exists():
        with open(scorecard_path, "r", encoding="utf-8") as fh:
            scorecard = json.load(fh)
        for split_name in ["development", "final_test", "challenge"]:
            metrics = scorecard.get(split_name, {})
            if not metrics:
                continue
            st.markdown(f"### {split_name.replace('_', ' ').title()}")
            for key, value in metrics.items():
                st.write(f"{key}: {value}")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        st.dataframe(df)

    st.subheader("Validation report")
    for file_name, report in validation_report.items():
        st.write(f"{file_name}: pages={report['page_count']}, sections={report['sections_found']}, missing={report['missing_sections']}")
