"""Task 6: Generate normalized sentence embeddings for all chunks using Sentence Transformers."""

import os
import json
import sys
import numpy as np
from pathlib import Path

# --- Config & Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EMBEDDINGS = PROJECT_ROOT / "data" / "embeddings"
FALLBACK_PROCESSED = PROJECT_ROOT / "processed"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


def get_chunks_file() -> Path:
    p = DATA_PROCESSED / "chunks.json"
    if p.exists():
        return p
    return FALLBACK_PROCESSED / "chunks.json"


def load_chunks(path: str) -> list[dict]:
    """Load and validate chunks from JSON. Exit if file missing or empty."""
    p = Path(path)
    if not p.exists():
        print(f"❌ ERROR: {path} not found. Run chunk_papers.py first.")
        sys.exit(1)
    with open(p, "r", encoding="utf-8") as f:
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
    return embeddings.astype(np.float32)


def main():
    print("\n" + "=" * 66)
    print("  THE RELAY — Task 6: Sentence Transformers Embedding Generation")
    print("=" * 66)

    chunks_file = get_chunks_file()
    print(f"\n📂 Loading chunks from: {chunks_file}")
    chunks = load_chunks(str(chunks_file))
    print(f"   ✅ Loaded {len(chunks)} chunks.")

    texts = [c["text"] for c in chunks]
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

    print("\n🤖 Loading embedding model...")
    model = load_model(MODEL_NAME)

    print("\n⚡ Generating embeddings...")
    embeddings = generate_embeddings(model, texts)
    print(f"\n   ✅ Embeddings generated: shape {embeddings.shape} (chunks × dims)")

    DATA_EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    FALLBACK_PROCESSED.mkdir(parents=True, exist_ok=True)

    for target_dir in [DATA_EMBEDDINGS, DATA_PROCESSED, FALLBACK_PROCESSED]:
        emb_path = target_dir / "embeddings.npy"
        np.save(str(emb_path), embeddings)

    for target_dir in [DATA_PROCESSED, FALLBACK_PROCESSED]:
        meta_path = target_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"   💾 Saved embeddings and metadata.")
    print("\n  🎉 Task 6 complete! Ready for FAISS indexing in Task 7.\n")


if __name__ == "__main__":
    main()

