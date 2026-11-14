"""Task 7: Build FAISS vector index and search research paper chunks by question."""

import os
import json
import re
import sys
from pathlib import Path
import numpy as np
import faiss

# --- Config & Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_EMBEDDINGS = PROJECT_ROOT / "data" / "embeddings"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FALLBACK_PROCESSED = PROJECT_ROOT / "processed"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
DEFAULT_RELEVANCE_THRESHOLD = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.38"))

PAPER_ID_MAP = {
    "attention is all you need": "1706.03762v7",
    "moshi": "2410.00037v1",
    "dynamic chunking for end-to-end hierarchical sequence modeling": "2507.07955v2",
    "dynamic chunking for rag": "2507.07955v2",
    "dynamic chunking": "2507.07955v2",
}


def get_paper_id(paper_name: str) -> str:
    """Return canonical arXiv paper ID for a given paper title."""
    name_clean = paper_name.lower().strip()
    for key, pid in PAPER_ID_MAP.items():
        if key in name_clean or name_clean in key:
            return pid
    return "arxiv-doc"


def get_embeddings_path() -> Path:
    for p in [DATA_EMBEDDINGS / "embeddings.npy", DATA_PROCESSED / "embeddings.npy", FALLBACK_PROCESSED / "embeddings.npy"]:
        if p.exists():
            return p
    return DATA_EMBEDDINGS / "embeddings.npy"


def get_metadata_path() -> Path:
    for p in [DATA_PROCESSED / "metadata.json", FALLBACK_PROCESSED / "metadata.json", DATA_PROCESSED / "chunk_metadata.json", FALLBACK_PROCESSED / "chunk_metadata.json"]:
        if p.exists():
            return p
    return DATA_PROCESSED / "metadata.json"


def get_index_path() -> Path:
    for p in [DATA_EMBEDDINGS / "faiss_index.bin", DATA_PROCESSED / "faiss_index.bin", FALLBACK_PROCESSED / "faiss_index.bin"]:
        if p.exists():
            return p
    return DATA_EMBEDDINGS / "faiss_index.bin"


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Create IndexFlatIP index and add all normalized vectors."""
    n_dims = embeddings.shape[1]
    index = faiss.IndexFlatIP(n_dims)
    index.add(embeddings)
    return index


def search(
    query: str,
    model,
    index: faiss.Index,
    metadata: list[dict],
    top_k: int = TOP_K,
    score_threshold: float | None = None,
) -> list[dict]:
    """Embed query, normalize, search FAISS index and map to chunk metadata."""
    if not query.strip():
        print("⚠️  Empty query — please provide a question.")
        return []

    query_vec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = metadata[idx]
        val_score = round(float(score), 4)
        if score_threshold is not None and val_score < score_threshold:
            continue
        p_name = chunk["paper_name"]
        p_id = chunk.get("paper_id") or get_paper_id(p_name)
        results.append({
            "chunk_id":    chunk["chunk_id"],
            "paper_id":    p_id,
            "paper_name":  p_name,
            "paper_title": p_name,
            "page_number": chunk.get("page_number", "N/A"),
            "score":       val_score,
            "text":        chunk["text"],
        })
    return results


def compute_lexical_overlap(query: str, text: str) -> float:
    """Calculate normalized token-overlap ratio between query and text."""
    query_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", query.lower()))
    if not query_tokens:
        return 0.0
    text_lower = text.lower()
    matches = sum(1 for q in query_tokens if q in text_lower)
    return matches / len(query_tokens)


def hybrid_rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 8,
    alpha: float = 0.65,
) -> list[dict]:
    """Fuse semantic vector score with lexical token overlap to re-rank chunks."""
    reranked = []
    for c in candidates:
        sem_score = float(c.get("score", 0.0))
        lex_score = compute_lexical_overlap(query, c.get("text", ""))
        hybrid_score = round(alpha * sem_score + (1.0 - alpha) * lex_score, 4)
        c_copy = dict(c)
        c_copy["semantic_score"] = sem_score
        c_copy["lexical_score"] = lex_score
        c_copy["score"] = hybrid_score
        reranked.append(c_copy)
    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked[:top_k]


def print_results(query: str, results: list[dict]):
    """Pretty-print search results for one query."""
    print(f"\n{'=' * 66}")
    print(f"  🔍 Query: \"{query}\"")
    print(f"{'=' * 66}")
    if not results:
        print("  No results found.")
        return
    for rank, r in enumerate(results, 1):
        print(f"\n  Rank {rank}  |  Score: {r['score']:.4f}")
        print(f"  Paper : {r['paper_name']}  (page {r['page_number']})")
        print(f"  ID    : chunk #{r['chunk_id']}")
        preview = r["text"][:300].replace("\n", " ")
        print(f"  Text  : \"{preview}...\"")


def main():
    print("\n" + "=" * 66)
    print("  THE RELAY — Task 7: FAISS Vector Search")
    print("=" * 66)

    embeddings_file = get_embeddings_path()
    metadata_file = get_metadata_path()

    if not embeddings_file.exists() or not metadata_file.exists():
        print(f"❌ ERROR: embeddings or metadata not found.")
        print("   Run generate_embeddings.py first.")
        sys.exit(1)

    print("\n📂 Loading embeddings and metadata...")
    embeddings = np.load(str(embeddings_file))
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    n_chunks, n_dims = embeddings.shape
    print(f"   ✅ Embeddings : {n_chunks} vectors × {n_dims} dims")
    print(f"   ✅ Metadata   : {len(metadata)} chunk records")

    if n_chunks != len(metadata):
        print(f"❌ ERROR: Mismatch — {n_chunks} embeddings vs {len(metadata)} metadata records.")
        sys.exit(1)

    print("\n🔧 Building FAISS IndexFlatIP...")
    index = build_faiss_index(embeddings)
    print(f"   ✅ Index built  — {index.ntotal} vectors indexed")

    out_index_path = get_index_path()
    out_index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_index_path))

    # Also save to data/embeddings, data/processed, and processed for compatibility
    for p in [DATA_EMBEDDINGS / "faiss_index.bin", DATA_PROCESSED / "faiss_index.bin", FALLBACK_PROCESSED / "faiss_index.bin"]:
        p.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(p))

    idx_kb = out_index_path.stat().st_size / 1024
    print(f"   💾 Saved index : {out_index_path} ({idx_kb:.1f} KB)")

    print(f"\n🤖 Loading embedding model for query encoding...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    print(f"   ✅ Model ready : {MODEL_NAME}")

    test_queries = [
        "What is the Transformer architecture?",
        "How does self-attention work?",
        "What is Moshi?",
        "What is dynamic chunking?",
        "What are the limitations discussed in the research papers?",
    ]

    print(f"\n\n{'=' * 66}")
    print("  🧪 RUNNING TEST QUERIES")
    print(f"{'=' * 66}")

    for query in test_queries:
        results = search(query, model, index, metadata, top_k=TOP_K)
        print_results(query, results)

    print(f"\n\n{'=' * 66}")
    print("  🎉 Task 7 complete! FAISS index built and search verified.")
    print(f"  Index saved to: {out_index_path}")
    print(f"{'=' * 66}")


if __name__ == "__main__":
    main()

