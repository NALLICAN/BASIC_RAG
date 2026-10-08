# Banking RAG Policy Assistant

This project implements a production-style banking policy RAG assistant that answers staff questions strictly from the two policy PDFs supplied in the project.

## Overview

The system:

- loads only the two permitted PDFs
- preserves section metadata and page metadata during extraction
- chunks the policy text with section-aware splitting
- embeds chunks with SentenceTransformers
- stores vectors in FAISS locally
- retrieves the top 3 chunks for each question
- sends only the question and retrieved evidence to the model
- refuses to invent missing policy information
- safely rejects OTP / credential / guarantee requests
- provides an offline evaluation screen for development and final-test metrics

## Architecture

The application is organized into modular Python components:

- app.py: Streamlit user interface
- src/pdf_loader.py: PDF text extraction and section validation
- src/chunker.py: section-aware chunking with metadata
- src/embeddings.py: embedding generation
- src/vector_store.py: local FAISS index build and cosine-equivalent search
- src/retriever.py: top-3 retrieval logic
- src/generator.py: answer generation and fallback policy-grounded generation
- src/safety.py: OTP and guarantee safety handling
- src/citations.py: citation formatting
- src/evaluation.py: evaluation helpers and scoring logic

## Directory structure

```text
banking-rag-assistant/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── Accounts_and_Service_Policy.pdf
│   └── Personal_Loan_Policy.pdf
├── notebooks/
│   └── banking_rag_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── pdf_loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── generator.py
│   ├── citations.py
│   ├── safety.py
│   └── evaluation.py
├── evaluation/
│   ├── questions.json
│   ├── results.json
│   ├── scorecard.json
│   ├── question_level_results.csv
│   └── config.json
├── .gitignore
└── .env.example
```

## Prerequisites

- Python 3.10 or newer
- pip
- access to the two bank policy PDFs in the data directory
- optional OpenAI API key for a live model request

## Python version

The project is intended for Python 3.10+.

## Installation

```bash
pip install -r requirements.txt
```

## Environment variables

Create a .env file if needed with values such as:

```bash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_key_here
CHUNK_SIZE=700
CHUNK_OVERLAP=100
TOP_K=3
```

Do not commit secrets. The app never displays API keys in the UI.

## How to run

### Streamlit

```bash
streamlit run app.py
```

### Notebook

```bash
jupyter notebook notebooks/banking_rag_evaluation.ipynb
```

## Evaluation

```bash
python notebooks/banking_rag_evaluation.py
```

The notebook is the reproducible evaluation workflow for:

- section validation
- chunk creation
- embedding generation
- FAISS retrieval
- answer generation
- citation verification
- safe handling checks
- latency tracking
- artifact generation under evaluation/

## Embedding model

Default embedding model:

- sentence-transformers/all-MiniLM-L6-v2
- embedding dimension: 384

The same model is used for document and query embedding.

## LLM model

The LLM is configured via environment variables:

- LLM_PROVIDER
- LLM_MODEL
- OPENAI_API_KEY

The app supports OpenAI-compatible inference when a key is provided. If no API key is configured, the app falls back to a deterministic policy-grounded response that is still constrained to the retrieved passages.

## Chunk size and overlap

- Chunk size: 700 characters
- Chunk overlap: 100 characters
- Top-k retrieval: 3
- Vector store: FAISS IndexFlatIP

## PDF processing approach

The pipeline uses PyMuPDF to read each PDF page, extract page text, and associate section headings with the text that follows them. Section IDs and section titles are retained in metadata.

## Section extraction approach

The loader checks for section headings matching ASxx and PLxx patterns and validates that all expected 16 section IDs are discovered. Missing sections are reported explicitly without silently ignoring them.

## Citation approach

Every material claim is tied back to the retrieved source with the format:

```text
[Source: filename, Page: X, Section: SECTION_ID]
```

The UI also shows the structured metadata for each supporting passage.

## Safety handling

The app refuses to:

- request or process OTPs, passwords, PINs, CVVs, or security secrets
- guarantee loan approval or sanction
- invent policy values not present in the source documents

When a question cannot be answered from the retrieved policy, the app explicitly tells the user that the policy documents do not specify the information and refers the user to authorised bank staff.

## Evaluation methodology

The project keeps a separate evaluation workflow that stores:

- question-level results
- retrieval evidence recall
- faithfulness
- answer correctness
- response relevance
- citation verification
- safe handling
- latency measurements

The evaluation artifacts are saved under evaluation/.

## Results

Offline evaluation artifacts are generated in the evaluation directory. These are used for review and classroom grading.

## Known limitations

- The app uses retrieved policy passages as the only approved evidence source.
- It does not claim to be a production banking decision engine.
- If an external LLM API key is not configured, the system uses deterministic fallback generation rather than a remote model call.
- Ragas is not required for the assignment and is only used opportunistically if available.

## Colab instructions

If run in Colab:

1. Upload the two PDF files into the working directory or mount Google Drive.
2. Install dependencies with pip install -r requirements.txt.
3. Set environment variables for LLM access and run the notebook or Streamlit app.
4. Keep the knowledge base restricted to the two supplied policy PDFs.
