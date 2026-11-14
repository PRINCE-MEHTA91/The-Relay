"""Task 10: Citation Validation for RAG answers against retrieved chunk metadata and text."""

import re
from typing import Dict, Any, List, Set, Tuple, Optional

# Supported paper titles for fuzzy matching
KNOWN_PAPERS = [
    "Attention Is All You Need",
    "Moshi",
    "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
    "Dynamic Chunking for RAG",
]

# Regex pattern for citations: [Paper Name, p. X] or [Paper Name, page X] or [Paper Name] or [Paper Name: p. X] or [Paper Name, pp. X-Y]
# Supports Unicode non-breaking hyphens (\u2011), en-dashes (\u2013), em-dashes (\u2014), and any paper title within brackets
CITATION_REGEX = re.compile(
    r"\[([^,\]]+?)(?:[,:]\s*(?:pp?\.|pages?|pgs?\.?)\s*(\d+)(?:[-–—\u2011\u2013\u2014]\d+)?)?\]",
    re.IGNORECASE,
)

# Standard refusal phrases for unsupported questions
REFUSAL_PHRASES = [
    "the available research papers do not contain sufficient information",
    "the available research papers do not contain",
    "available research papers do not contain sufficient information",
    "available research papers do not contain",
    "the answer cannot be found in the provided research papers",
    "the answer cannot be found in the provided",
    "the answer cannot be found",
    "cannot be found in the provided research papers",
    "cannot be found in the provided",
    "cannot be found",
    "not found in the provided research papers",
    "not found in the provided",
    "the provided passages do not contain",
    "does not contain sufficient information",
    "no information provided",
    "no relevant research paper passages",
]

# Common English stop words for content-word overlap checks
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "did", "do", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it",
    "its", "itself", "let's", "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "were", "weren't", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "won't", "would", "you", "your", "yours", "yourself",
}


def normalize_title(title: str) -> str:
    """Normalize paper title for case-insensitive and punctuation-free matching."""
    cleaned = re.sub(r"[^a-z0-9\s]", "", title.lower())
    return " ".join(cleaned.split())


def matches_paper_name(cited_name: str, candidate_name: str) -> bool:
    """Check if cited paper name matches a candidate paper name fuzzily or exactly."""
    norm_cited = normalize_title(cited_name)
    norm_cand = normalize_title(candidate_name)
    if not norm_cited or not norm_cand:
        return False
    return norm_cited == norm_cand or norm_cited in norm_cand or norm_cand in norm_cited


def parse_citations(
    text: str,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Extract all in-text citations [Paper, p. X] or [Passage X] from the text."""
    citations = []
    for match in CITATION_REGEX.finditer(text):
        paper_raw = match.group(1).strip()
        page_raw = match.group(2)
        page_num = int(page_raw) if page_raw else None

        # Check for [Passage X] or [Chunk X] references
        p_lower = paper_raw.lower()
        if p_lower.startswith("passage") or p_lower.startswith("chunk") or paper_raw.isdigit():
            idx_digits = re.findall(r"\d+", paper_raw)
            if idx_digits and retrieved_chunks:
                p_idx = int(idx_digits[0])
                if 1 <= p_idx <= len(retrieved_chunks):
                    chunk = retrieved_chunks[p_idx - 1]
                    resolved_paper = chunk.get("paper_name", paper_raw)
                    resolved_page = page_num if page_num is not None else chunk.get("page_number")
                    if str(resolved_page).isdigit():
                        resolved_page = int(resolved_page)
                    citations.append({
                        "raw_citation": match.group(0),
                        "paper_name": resolved_paper,
                        "page_number": resolved_page,
                    })
                    continue
            continue

        citations.append({
            "raw_citation": match.group(0),
            "paper_name": paper_raw,
            "page_number": page_num,
        })
    return citations


def split_sentences(text: str) -> List[str]:
    """Split text into distinct sentences while ignoring citation brackets."""
    # Remove citations temporarily to avoid splitting on dots in 'p. 3'
    cleaned = re.sub(r"\[[^\]]+\]", "", text)
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def extract_content_words(text: str) -> Set[str]:
    """Extract lowercase non-stopword tokens of length >= 3."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return {w for w in words if w not in STOP_WORDS}


def verify_citation_metadata(
    citations: List[Dict[str, Any]],
    retrieved_chunks: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Check that cited paper names and page numbers exist in retrieved chunks."""
    validated = []
    errors = []

    # Map retrieved paper names to their covered pages and chunk IDs
    retrieved_papers: Dict[str, Dict[str, Any]] = {}
    for chunk in retrieved_chunks:
        p_name = chunk.get("paper_name") or chunk.get("paper_title", "")
        norm_p = normalize_title(p_name)
        if norm_p not in retrieved_papers:
            retrieved_papers[norm_p] = {
                "display_name": p_name,
                "paper_id": chunk.get("paper_id", ""),
                "pages": set(),
                "chunks": [],
            }
        page = chunk.get("page_number")
        if page is not None and str(page).isdigit():
            retrieved_papers[norm_p]["pages"].add(int(page))
        retrieved_papers[norm_p]["chunks"].append(chunk)

    for cit in citations:
        cited_paper = cit["paper_name"]
        cited_page = cit["page_number"]
        raw = cit["raw_citation"]

        # 1. Match paper title against retrieved chunks
        matched_key = None
        for r_norm, r_data in retrieved_papers.items():
            if matches_paper_name(cited_paper, r_data["display_name"]) or (r_data.get("paper_id") and cited_paper.lower() == r_data["paper_id"].lower()):
                matched_key = r_norm
                break

        if not matched_key:
            errors.append(
                f"Nonexistent or unretrieved paper cited: '{cited_paper}' in {raw}. "
                f"Retrieved papers are: {[d['display_name'] for d in retrieved_papers.values()]}"
            )
            continue

        paper_data = retrieved_papers[matched_key]
        available_pages = paper_data["pages"]

        # 2. Match page number if specified
        if cited_page is not None:
            page_matched = False
            if cited_page in available_pages:
                page_matched = True
            elif any(abs(cited_page - p) <= 1 for p in available_pages):
                # Tolerance of +/- 1 page for overlapping chunk windows
                page_matched = True
            else:
                for c in paper_data["chunks"]:
                    txt = c.get("text", "").lower()
                    if f"page {cited_page}" in txt or f"p. {cited_page}" in txt or f"p.{cited_page}" in txt:
                        page_matched = True
                        break

            if not page_matched and available_pages:
                sorted_pages = sorted(list(available_pages))
                errors.append(
                    f"Page mismatch for '{paper_data['display_name']}': cited page {cited_page}, "
                    f"but retrieved chunks only cover page(s): {sorted_pages}."
                )
                continue

        # Validated citation
        matching_chunk = next(
            (c for c in paper_data["chunks"] if c.get("page_number") == cited_page),
            paper_data["chunks"][0] if paper_data["chunks"] else {}
        )

        validated.append({
            "raw_citation": raw,
            "paper_name": paper_data["display_name"],
            "page_number": cited_page if cited_page is not None else matching_chunk.get("page_number", "N/A"),
            "chunk_id": matching_chunk.get("chunk_id", "N/A"),
            "verified": True,
        })

    return validated, errors


# Preamble and framing phrases that do not constitute technical claims
FRAMING_PREFIXES = [
    "based on the provided",
    "based on the retrieved",
    "according to the provided",
    "according to the retrieved",
    "in the provided research",
    "in the provided passages",
    "in the retrieved passages",
    "the retrieved research papers indicate",
    "as stated in the",
    "as described in the",
    "the paper explains",
    "here is the summary",
    "to answer your question",
]


def check_claim_support(
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    min_word_overlap_ratio: float = 0.30,
) -> List[str]:
    """Check if sentences in the answer are supported by retrieved text."""
    # Combine all retrieved text into a single reference text
    context_text = " ".join(c.get("text", "") for c in retrieved_chunks)
    context_words = extract_content_words(context_text)

    if not context_words:
        return ["Retrieved context is empty; cannot support any claims."]

    sentences = split_sentences(answer)
    unsupported = []

    for sentence in sentences:
        s_lower = sentence.lower()
        # Skip refusal / meta statements
        if any(refusal in s_lower for refusal in REFUSAL_PHRASES):
            continue
        if any(s_lower.startswith(prefix) for prefix in FRAMING_PREFIXES):
            continue
        if sentence.strip().endswith("?"):
            continue

        claim_words = extract_content_words(sentence)
        if len(claim_words) < 4:
            # Skip very short conversational or transition phrases
            continue

        overlap = claim_words.intersection(context_words)
        ratio = len(overlap) / len(claim_words) if claim_words else 0.0

        if ratio < min_word_overlap_ratio:
            unsupported.append(
                f"Unsupported claim (overlap {ratio:.1%}): \"{sentence}\""
            )

    return unsupported


def is_valid_refusal(answer: str) -> bool:
    """Check if the answer is a legitimate refusal statement for unsupported queries."""
    lower_ans = answer.lower()
    return any(phrase in lower_ans for phrase in REFUSAL_PHRASES)


def validate_citations(
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    citation_metadata: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Validate citations and claims in the answer against retrieved chunks."""
    # Case 1: Empty answer
    if not answer or not answer.strip():
        return {
            "is_valid": False,
            "citation_errors": ["Generated answer is empty."],
            "unsupported_claims": [],
            "validated_citations": [],
            "validation_status": "EMPTY_ANSWER",
        }

    # Case 2: Refusal for unsupported question
    if is_valid_refusal(answer):
        return {
            "is_valid": True,
            "citation_errors": [],
            "unsupported_claims": [],
            "validated_citations": [],
            "validation_status": "REFUSAL_VERIFIED",
        }

    # Case 3: Parse citations from answer
    citations = parse_citations(answer, retrieved_chunks=retrieved_chunks)
    if not citations and citation_metadata:
        citations = citation_metadata

    # Case 4: Missing citations when substantive claims exist
    if not citations:
        return {
            "is_valid": False,
            "citation_errors": ["Answer makes technical claims but includes no in-text citations."],
            "unsupported_claims": [],
            "validated_citations": [],
            "validation_status": "MISSING_CITATIONS",
        }

    # Case 5: Verify metadata (paper names and page numbers)
    validated_citations, metadata_errors = verify_citation_metadata(citations, retrieved_chunks)

    # Case 6: Verify factual claim support in retrieved passages
    unsupported_claims = check_claim_support(answer, retrieved_chunks)

    all_errors = metadata_errors
    is_valid = (len(all_errors) == 0) and (len(unsupported_claims) == 0) and (len(validated_citations) > 0)

    if is_valid:
        status = "VERIFIED"
    elif metadata_errors and unsupported_claims:
        status = "METADATA_AND_CLAIM_ERRORS"
    elif metadata_errors:
        status = "METADATA_MISMATCH"
    else:
        status = "UNSUPPORTED_CLAIMS"

    return {
        "is_valid": is_valid,
        "citation_errors": all_errors,
        "unsupported_claims": unsupported_claims,
        "validated_citations": validated_citations,
        "validation_status": status,
    }


def run_test_suite():
    """Run all 7 required test cases to verify the validator."""
    print("=" * 66)
    print("  THE RELAY — Task 10: Citation Validator Test Suite")
    print("=" * 66)

    # Mock retrieved chunks representing realistic FAISS search output
    mock_chunks = [
        {
            "chunk_id": 3,
            "paper_name": "Attention Is All You Need",
            "page_number": 3,
            "text": (
                "The Transformer follows an encoder-decoder architecture using stacked self-attention "
                "and point-wise, fully connected layers for both the encoder and decoder. "
                "The encoder maps an input sequence to continuous representations."
            ),
        },
        {
            "chunk_id": 4,
            "paper_name": "Attention Is All You Need",
            "page_number": 4,
            "text": (
                "An attention function can be described as mapping a query and a set of key-value pairs "
                "to an output, where the query, keys, values, and output are all vectors."
            ),
        },
        {
            "chunk_id": 50,
            "paper_name": "Moshi",
            "page_number": 11,
            "text": (
                "Moshi is a speech-text foundation model that enables real-time full-duplex spoken dialogue. "
                "It jointly models user audio and system audio using separate audio codec streams."
            ),
        },
    ]

    test_cases = [
        {
            "name": "Test 1: Correct answer with correct citation",
            "answer": (
                "The Transformer uses an encoder-decoder architecture based on stacked self-attention "
                "and fully connected layers [Attention Is All You Need, p. 3]."
            ),
            "chunks": mock_chunks,
            "expected_valid": True,
            "expected_status": "VERIFIED",
        },
        {
            "name": "Test 2: Correct paper but incorrect page number",
            "answer": (
                "The Transformer architecture maps inputs to continuous representations "
                "[Attention Is All You Need, p. 99]."
            ),
            "chunks": mock_chunks,
            "expected_valid": False,
            "expected_status": "METADATA_MISMATCH",
        },
        {
            "name": "Test 3: Citation to a nonexistent research paper",
            "answer": (
                "The Transformer was improved using deep reasoning algorithms "
                "[DeepSeek-R1 Technical Report, p. 5]."
            ),
            "chunks": mock_chunks,
            "expected_valid": False,
            "expected_status": "METADATA_MISMATCH",
        },
        {
            "name": "Test 4: Answer containing unsupported information",
            "answer": (
                "The Transformer architecture was invented in 1895 by Nikola Tesla to control electric space satellites "
                "[Attention Is All You Need, p. 3]."
            ),
            "chunks": mock_chunks,
            "expected_valid": False,
            "expected_status": "UNSUPPORTED_CLAIMS",
        },
        {
            "name": "Test 5: Answer with no citations",
            "answer": (
                "The Transformer architecture uses self-attention mechanisms to map query and key-value vectors."
            ),
            "chunks": mock_chunks,
            "expected_valid": False,
            "expected_status": "MISSING_CITATIONS",
        },
        {
            "name": "Test 6: Empty generated answer",
            "answer": "",
            "chunks": mock_chunks,
            "expected_valid": False,
            "expected_status": "EMPTY_ANSWER",
        },
        {
            "name": "Test 7: Correctly refused unsupported question",
            "answer": "The answer cannot be found in the provided research papers.",
            "chunks": mock_chunks,
            "expected_valid": True,
            "expected_status": "REFUSAL_VERIFIED",
        },
    ]

    passed = 0
    total = len(test_cases)

    for i, tc in enumerate(test_cases, start=1):
        print(f"\n--- {tc['name']} ---")
        result = validate_citations(tc["answer"], tc["chunks"])

        is_valid_match = (result["is_valid"] == tc["expected_valid"])
        status_match = (result["validation_status"] == tc["expected_status"])
        case_passed = is_valid_match and status_match

        if case_passed:
            passed += 1
            print(f"  Result : PASS ✅")
        else:
            print(f"  Result : FAIL ❌")

        print(f"  is_valid           : {result['is_valid']} (Expected: {tc['expected_valid']})")
        print(f"  validation_status  : {result['validation_status']} (Expected: {tc['expected_status']})")
        if result["citation_errors"]:
            print(f"  citation_errors    : {result['citation_errors']}")
        if result["unsupported_claims"]:
            print(f"  unsupported_claims : {result['unsupported_claims']}")
        if result["validated_citations"]:
            print(f"  validated_citations: {len(result['validated_citations'])} source(s)")

    print(f"\n{'=' * 66}")
    print(f"  TEST SUITE SCORE: {passed}/{total} Passed")
    print(f"{'=' * 66}")


if __name__ == "__main__":
    run_test_suite()

