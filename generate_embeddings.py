"""Task 6: Generate normalized sentence embeddings for all chunks using Sentence Transformers."""

import os
import json
import sys
import numpy as np

# --- Config ---
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR   = os.path.join(BASE_DIR, "processed")
CHUNKS_FILE     = os.path.join(PROCESSED_DIR, "chunks.json")
EMBEDDINGS_FILE = os.path.join(PROCESSED_DIR, "embeddings.npy")
METADATA_FILE   = os.path.join(PROCESSED_DIR, "metadata.json")

MODEL_NAME      = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE      = 32    # chunks per batch — tune down if you run out of RAM


def load_chunks(path: str) -> list[dict]:
    """Load and validate chunks from JSON. Exit if file missing or empty."""
    if not os.path.exists(path):
        print(f"❌ ERROR: {path} not found. Run chunk_papers.py first.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if not chunks:
        print("❌ ERROR: chunks.json is empty.")
        sys.exit(1)
    return chunks


def load_model(model_name: str):
    """Load the Sentence Transformer model (downloads on first run ~22 MB)."""
    from sentence_transformers import SentenceTransformer
    print(f"  Loading model: {model_name}")
    print("  (First run will download ~22 MB — cached after that)")
    model = SentenceTransformer(model_name)
    print("  ✅ Model loaded.\n")
    return model


def generate_embeddings(model, texts: list[str]) -> np.ndarray:
    """Encode all texts in batches with normalize_embeddings=True for cosine similarity."""
    print(f"  Encoding {len(texts)} chunks in batches of {BATCH_SIZE}...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,  # required for IndexFlatIP cosine search
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)   # FAISS requires float32


def main():
    print("\n" + "=" * 62)
    print("  THE RELAY — Task 6: Generate Text Embeddings")
    print("=" * 62)

    # 1. Load chunks
    print(f"\n📂 Loading chunks from processed/chunks.json ...")
    chunks = load_chunks(CHUNKS_FILE)
    print(f"   ✅ {len(chunks)} chunks loaded.")
    print(f"   Keys : {list(chunks[0].keys())}")

    # 2. Load model
    print(f"\n🤖 Setting up embedding model...")
    model = load_model(MODEL_NAME)

    # 3. Extract text in chunk_id order (preserve alignment with metadata)
    texts = [chunk["text"] for chunk in chunks]

    # 4. Generate embeddings
    print("⚡ Generating embeddings...")
    embeddings = generate_embeddings(model, texts)

    # 5. Validate shape
    n_chunks, n_dims = embeddings.shape
    print(f"\n  ✅ Embeddings generated!")
    print(f"  📊 Total chunks      : {n_chunks}")
    print(f"  📊 Embedding dims    : {n_dims}")
    print(f"  📊 Model             : {MODEL_NAME}")
    print(f"  📊 Normalized        : Yes (cosine-ready)")
    print(f"  📊 Array dtype       : {embeddings.dtype}")
    print(f"  📊 Memory usage      : {embeddings.nbytes / 1024:.1f} KB")

    # 6. Save embeddings as numpy array (.npy — compact, fast to load)
    np.save(EMBEDDINGS_FILE, embeddings)
    emb_kb = os.path.getsize(EMBEDDINGS_FILE) / 1024
    print(f"\n💾 Saved embeddings   : processed/embeddings.npy  ({emb_kb:.1f} KB)")

    # 7. Save metadata (all chunk fields except we keep text for retrieval display)
    metadata = [
        {
            "chunk_id":    c["chunk_id"],
            "paper_name":  c["paper_name"],
            "page_number": c["page_number"],
            "token_count": c["token_count"],
            "text":        c["text"],
        }
        for c in chunks
    ]
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    meta_kb = os.path.getsize(METADATA_FILE) / 1024
    print(f"💾 Saved metadata     : processed/metadata.json   ({meta_kb:.1f} KB)")

    print(f"\n  🎉 Task 6 complete! Run search_papers.py to build FAISS index and search.")


if __name__ == "__main__":
    main()

