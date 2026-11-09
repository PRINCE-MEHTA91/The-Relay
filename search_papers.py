"""
Task 7: Build a FAISS vector index and search research paper chunks by question.

Index type : IndexFlatIP (exact inner-product search on normalized vectors = cosine similarity)
Input      : processed/embeddings.npy  |  processed/metadata.json
Output     : processed/faiss_index.bin
"""

import os
import json
import sys
import numpy as np
import faiss

# --- Config ---
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR   = os.path.join(BASE_DIR, "processed")
EMBEDDINGS_FILE = os.path.join(PROCESSED_DIR, "embeddings.npy")
METADATA_FILE   = os.path.join(PROCESSED_DIR, "metadata.json")
INDEX_FILE      = os.path.join(PROCESSED_DIR, "faiss_index.bin")

MODEL_NAME      = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K           = 5     # number of results to return per query


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Create an IndexFlatIP (Inner Product) index and add all embeddings.
    IndexFlatIP = exact brute-force search. With normalized vectors,
    inner product == cosine similarity. No training step needed.
    """
    n_dims = embeddings.shape[1]
    index  = faiss.IndexFlatIP(n_dims)   # IP = Inner Product
    index.add(embeddings)                # add all vectors at once
    return index


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(
    query:     str,
    model,
    index:     faiss.Index,
    metadata:  list[dict],
    top_k:     int = TOP_K,
) -> list[dict]:
    """
    1. Embed the query with the same model used for documents.
    2. Normalize it (must match how document embeddings were normalized).
    3. Search the FAISS index for top_k nearest vectors.
    4. Map result IDs back to chunk metadata.
    """
    if not query.strip():
        print("⚠️  Empty query — please provide a question.")
        return []

    # Embed and normalize query
    query_vec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # FAISS search returns (scores, indices) arrays of shape [1, top_k]
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:          # FAISS returns -1 when fewer than top_k results exist
            continue
        chunk = metadata[idx]
        results.append({
            "chunk_id":    chunk["chunk_id"],
            "paper_name":  chunk["paper_name"],
            "page_number": chunk.get("page_number", "N/A"),
            "score":       round(float(score), 4),
            "text":        chunk["text"],
        })
    return results


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
        # Show first 300 chars of text
        preview = r["text"][:300].replace("\n", " ")
        print(f"  Text  : \"{preview}...\"")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 66)
    print("  THE RELAY — Task 7: FAISS Vector Search")
    print("=" * 66)

    # 1. Check required files
    for path, label in [
        (EMBEDDINGS_FILE, "embeddings.npy"),
        (METADATA_FILE,   "metadata.json"),
    ]:
        if not os.path.exists(path):
            print(f"❌ ERROR: processed/{label} not found.")
            print("   Run generate_embeddings.py first.")
            sys.exit(1)

    # 2. Load embeddings and metadata
    print("\n📂 Loading embeddings and metadata...")
    embeddings = np.load(EMBEDDINGS_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    n_chunks, n_dims = embeddings.shape
    print(f"   ✅ Embeddings : {n_chunks} vectors × {n_dims} dims")
    print(f"   ✅ Metadata   : {len(metadata)} chunk records")

    # Sanity check: embedding rows must align with metadata records
    if n_chunks != len(metadata):
        print(f"❌ ERROR: Mismatch — {n_chunks} embeddings vs {len(metadata)} metadata records.")
        sys.exit(1)

    # 3. Build FAISS index
    print("\n🔧 Building FAISS IndexFlatIP...")
    index = build_faiss_index(embeddings)
    print(f"   ✅ Index built  — {index.ntotal} vectors indexed")
    print(f"   Index type     : IndexFlatIP (exact cosine search)")

    # 4. Save FAISS index
    faiss.write_index(index, INDEX_FILE)
    idx_kb = os.path.getsize(INDEX_FILE) / 1024
    print(f"   💾 Saved index : processed/faiss_index.bin ({idx_kb:.1f} KB)")

    # 5. Load embedding model for query encoding
    print(f"\n🤖 Loading embedding model for query encoding...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    print(f"   ✅ Model ready : {MODEL_NAME}")

    # 6. Test queries
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
    print(f"  Index saved to: processed/faiss_index.bin")
    print(f"{'=' * 66}")


if __name__ == "__main__":
    main()

