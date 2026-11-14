"""Document retrieval, PDF text extraction, and dynamic token chunking for autonomous research."""

import re
import logging
from typing import List, Tuple, Dict, Any, Optional
import httpx
import fitz  # PyMuPDF

from agent.connectors.base import AcademicPaper
from rag.chunk_papers import load_tokenizer, chunk_token_sequence

logger = logging.getLogger(__name__)

# Max PDF size to download (25 MB)
MAX_PDF_BYTES = 25 * 1024 * 1024


async def download_paper_pdf(
    pdf_url: str,
    timeout_seconds: float = 12.0,
    max_bytes: int = MAX_PDF_BYTES,
) -> Optional[bytes]:
    """Download open-access PDF bytes with strict size bounds and polite headers."""
    if not pdf_url:
        return None

    headers = {
        "User-Agent": "TheRelay-ResearchAssistant/2.0 (mailto:researcher@the-relay.local)",
        "Accept": "application/pdf, application/octet-stream, */*",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True) as client:
            # Check headers first if server supports HEAD
            try:
                head_resp = await client.head(pdf_url, headers=headers)
                cl = head_resp.headers.get("content-length")
                if cl and int(cl) > max_bytes:
                    logger.warning(f"PDF at {pdf_url} exceeds max size ({cl} bytes); skipping.")
                    return None
            except Exception:
                pass  # Fall through to GET if HEAD unsupported

            # Stream download to enforce max_bytes limit
            async with client.stream("GET", pdf_url, headers=headers) as response:
                if response.status_code != 200:
                    logger.info(f"Failed to download PDF ({response.status_code}) from {pdf_url}")
                    return None

                chunks = []
                total_downloaded = 0
                async for chunk in response.aiter_bytes():
                    total_downloaded += len(chunk)
                    if total_downloaded > max_bytes:
                        logger.warning(f"Aborting download from {pdf_url}: exceeded {max_bytes} bytes.")
                        return None
                    chunks.append(chunk)

                pdf_bytes = b"".join(chunks)
                # Verify PDF magic bytes
                if not pdf_bytes.startswith(b"%PDF"):
                    # Check if HTML paywall or captcha was returned
                    logger.info(f"Downloaded content from {pdf_url} is not a valid PDF file.")
                    return None

                return pdf_bytes

    except Exception as e:
        logger.info(f"PDF download error from {pdf_url}: {e}")
        return None


def extract_pdf_pages(pdf_bytes: bytes, max_pages: int = 50) -> List[Tuple[int, str]]:
    """Extract page-by-page text from in-memory PDF bytes preserving page numbers."""
    pages: List[Tuple[int, str]] = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = min(len(doc), max_pages)

        for page_num in range(page_count):
            page = doc[page_num]
            text = page.get_text("text")

            # Clean control chars while preserving newlines
            cleaned = "".join(ch for ch in text if ch == "\n" or ch == "\t" or (ord(ch) >= 32))
            lines = [line.rstrip() for line in cleaned.splitlines()]
            cleaned_text = "\n".join(lines).strip()

            if cleaned_text:
                pages.append((page_num + 1, cleaned_text))

        doc.close()
    except Exception as e:
        logger.error(f"PyMuPDF extraction failed: {e}")

    return pages


async def process_academic_paper(
    paper: AcademicPaper,
    timeout_seconds: float = 12.0,
) -> Tuple[str, str, List[Tuple[int, str]]]:
    """Retrieve full text from PDF if available; fallback to abstract if unavailable.
    
    Returns:
        (content_type, message, pages)
        content_type is either "full_text" or "abstract_only"
    """
    if paper.pdf_url:
        pdf_bytes = await download_paper_pdf(paper.pdf_url, timeout_seconds=timeout_seconds)
        if pdf_bytes:
            pages = extract_pdf_pages(pdf_bytes)
            if pages:
                return (
                    "full_text",
                    f"Successfully retrieved and parsed full text ({len(pages)} pages).",
                    pages,
                )

    # Fallback to abstract (verify actual substantive abstract text exists)
    abstract_text = paper.abstract.strip() if paper.abstract else ""
    if len(abstract_text) < 50:
        # Crossref metadata alone (or single character/whitespace) is NOT a valid abstract
        metadata_text = f"Title: {paper.title}. Authors: {', '.join(paper.authors)}. (No abstract or full text available)."
        return (
            "metadata_only",
            "Open-access PDF unavailable and no substantive abstract text provided by source (citation metadata only).",
            [(1, metadata_text)],
        )

    return (
        "abstract_only",
        "Open-access PDF unavailable or download failed; using paper abstract.",
        [(1, abstract_text)],
    )


def chunk_academic_paper(
    paper: AcademicPaper,
    content_type: str,
    pages: List[Tuple[int, str]],
    chunk_size: int = 400,
    overlap: int = 40,
    min_tokens: int = 15,
) -> List[Dict[str, Any]]:
    """Tokenize and slide window across paper pages, attaching comprehensive metadata."""
    encode_fn, decode_fn, _ = load_tokenizer()
    chunks: List[Dict[str, Any]] = []

    flat_tokens: List[int] = []
    token_page_map: List[int] = []

    for page_num, page_text in pages:
        p_tokens = encode_fn(page_text)
        flat_tokens.extend(p_tokens)
        token_page_map.extend([page_num] * len(p_tokens))

    if not flat_tokens:
        return []

    spans = chunk_token_sequence(flat_tokens, chunk_size, overlap)
    is_abstract_only = (content_type == "abstract_only")

    for idx, (start, end) in enumerate(spans, start=1):
        chunk_tokens = flat_tokens[start:end]
        if len(chunk_tokens) < min_tokens and len(spans) > 1:
            continue

        chunk_text = decode_fn(chunk_tokens).strip()
        page_votes = token_page_map[start:end]
        page_num = max(set(page_votes), key=page_votes.count) if page_votes else 1

        chunk_id = f"{paper.paper_id}_c{idx:03d}"
        chunks.append({
            "chunk_id": chunk_id,
            "paper_id": paper.paper_id,
            "paper_title": paper.title,
            "authors": paper.authors,
            "year": paper.year,
            "doi": paper.doi,
            "arxiv_id": paper.arxiv_id,
            "url": paper.url,
            "pdf_url": paper.pdf_url,
            "page_number": page_num,
            "content_type": content_type,
            "is_abstract_only": is_abstract_only,
            "token_count": len(chunk_tokens),
            "text": chunk_text,
        })

    return chunks
