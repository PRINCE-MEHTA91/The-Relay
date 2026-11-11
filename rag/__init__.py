"""Retrieval-Augmented Generation (RAG) components for paper extraction, chunking, search, and citation validation."""

from .extract_papers import extract_text_from_pdf
from .chunk_papers import build_chunks_for_size, chunk_paper
from .generate_embeddings import generate_embeddings
from .search_papers import (
    search,
    hybrid_rerank,
    build_faiss_index,
    compute_lexical_overlap,
)
from .citation_validator import (
    validate_citations,
    parse_citations,
    normalize_title,
    matches_paper_name,
    is_valid_refusal,
)

__all__ = [
    "extract_text_from_pdf",
    "build_chunks_for_size",
    "chunk_paper",
    "generate_embeddings",
    "search",
    "hybrid_rerank",
    "build_faiss_index",
    "compute_lexical_overlap",
    "validate_citations",
    "parse_citations",
    "normalize_title",
    "matches_paper_name",
    "is_valid_refusal",
]

