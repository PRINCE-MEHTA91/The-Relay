"""Chunk extracted research paper text into token-sized segments with metadata."""

import os
import json
import re
import sys

# --- Config ---
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR       = os.path.join(BASE_DIR, "extracted_text")
OUTPUT_DIR      = os.path.join(BASE_DIR, "processed")
OUTPUT_FILE     = os.path.join(OUTPUT_DIR, "chunks.json")

CHUNK_SIZE      = 500   # max tokens per chunk
CHUNK_OVERLAP   = 50    # tokens shared between consecutive chunks
MIN_CHUNK_TOKENS = 20   # discard very short chunks

PAPERS = [
    ("attention_is_all_you_need.txt", "Attention Is All You Need"),
    ("moshi.txt",                     "Moshi"),
    ("dynamic_chunking.txt",          "Dynamic Chunking for RAG"),
]

# Page marker pattern written by extract_papers.py
PAGE_MARKER_RE = re.compile(r"---\s*Page\s+(\d+)\s+of\s+(\d+)\s*---")


def load_tokenizer():
    """Load the BERT tokenizer; fallback to whitespace splitting if unavailable."""
    try:
        from transformers import AutoTokenizer
        print("  Loading BERT tokenizer (bert-base-uncased)...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        print("  ✅ BERT tokenizer loaded.\n")

        def encode(text: str) -> list[int]:
            return tokenizer.encode(text, add_special_tokens=False)

        def decode(token_ids: list[int]) -> str:
            return tokenizer.decode(token_ids, skip_special_tokens=True)

        return encode, decode, "bert-base-uncased"

    except Exception as e:
        print(f"  ⚠️  BERT tokenizer unavailable ({e}).")
        print("  ↳ Falling back to whitespace tokenization (1 word ≈ 1 token).\n")

        def encode(text: str) -> list[int]:
            return list(range(len(text.split())))

        def decode(token_ids: list[int]) -> str:
            return f"[fallback — {len(token_ids)} word tokens]"

        return encode, decode, "whitespace-fallback"


def parse_pages(raw_text: str) -> list[tuple[int, str]]:
    """Split raw text into (page_number, text) tuples using page header markers."""
    segments = PAGE_MARKER_RE.split(raw_text)
    if len(segments) < 4:
        return [(1, raw_text.strip())]

    pages = []
    for i in range(1, len(segments) - 2, 3):
        page_num  = int(segments[i])
        page_text = segments[i + 2].strip()
        if page_text:
            pages.append((page_num, page_text))
    return pages


def chunk_token_sequence(token_ids: list[int], chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    """Slide a window across token IDs with overlap, returning (start, end) index pairs."""
    spans = []
    start = 0
    step  = chunk_size - overlap
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        spans.append((start, end))
        if end == len(token_ids):
            break
        start += step
    return spans


def chunk_paper(
    raw_text: str,
    paper_name: str,
    encode_fn,
    decode_fn,
    chunk_id_start: int,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Tokenize and slide window across paper text, attaching chunk_id and metadata."""
    pages = parse_pages(raw_text)
    all_chunks = []
    chunk_id   = chunk_id_start

    flat_tokens: list[int] = []
    token_page_map: list[int] = []

    for page_num, page_text in pages:
        page_tokens = encode_fn(page_text)
        flat_tokens.extend(page_tokens)
        token_page_map.extend([page_num] * len(page_tokens))

    if not flat_tokens:
        return []

    spans = chunk_token_sequence(flat_tokens, chunk_size, overlap)

    for start, end in spans:
        chunk_tokens = flat_tokens[start:end]
        chunk_text   = decode_fn(chunk_tokens).strip()

        if len(chunk_tokens) < MIN_CHUNK_TOKENS:
            continue

        page_votes = token_page_map[start:end]
        page_num   = max(set(page_votes), key=page_votes.count)

        all_chunks.append({
            "chunk_id":    chunk_id,
            "paper_name":  paper_name,
            "page_number": page_num,
            "token_count": len(chunk_tokens),
            "text":        chunk_text,
        })
        chunk_id += 1

    return all_chunks


def build_chunks_for_size(chunk_size: int = 250, overlap: int = 25) -> list[dict]:
    """Generate chunks for all three papers at specified token size and overlap."""
    encode_fn, decode_fn, _ = load_tokenizer()
    chunks = []
    cid = 1
    for txt_name, paper_title in PAPERS:
        path = os.path.join(INPUT_DIR, txt_name)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        paper_chunks = chunk_paper(raw_text, paper_title, encode_fn, decode_fn, cid, chunk_size, overlap)
        chunks.extend(paper_chunks)
        cid += len(paper_chunks)
    return chunks


def main():
    """Check inputs, chunk all papers, save chunks.json and display summary."""
    print("\n" + "=" * 64)
    print("  THE RELAY — Task 5: Text Chunking")
    print("=" * 64)
    print(f"  chunk_size={CHUNK_SIZE} tokens  |  overlap={CHUNK_OVERLAP} tokens")

    print("\n📋 Checking input files...")
    missing = []
    for txt_name, _ in PAPERS:
        path = os.path.join(INPUT_DIR, txt_name)
        if os.path.exists(path):
            kb = os.path.getsize(path) / 1024
            print(f"   ✅ {txt_name}  ({kb:.1f} KB)")
        else:
            print(f"   ❌ MISSING: {txt_name}  → run extract_papers.py first")
            missing.append(txt_name)
    if missing:
        print("\n⛔ Missing files. Aborting.")
        sys.exit(1)

    print("\n🔤 Setting up tokenizer...")
    encode_fn, decode_fn, tokenizer_name = load_tokenizer()
    print(f"  Tokenizer: {tokenizer_name}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"\n📁 Created folder: processed/")
    else:
        print(f"\n📁 Output folder already exists: processed/")

    all_chunks: list[dict] = []
    paper_stats: list[tuple[str, int]] = []

    for txt_name, paper_name in PAPERS:
        print(f"\n{'=' * 64}")
        print(f"  📄 {paper_name}")
        print(f"{'=' * 64}")

        path = os.path.join(INPUT_DIR, txt_name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            if not raw_text.strip():
                print("  ❌ ERROR: File is empty.")
                paper_stats.append((paper_name, 0))
                continue

            char_count = len(raw_text)
            print(f"  Characters in file : {char_count:,}")

            start_id = len(all_chunks)
            chunks   = chunk_paper(raw_text, paper_name, encode_fn, decode_fn, start_id)
            all_chunks.extend(chunks)

            print(f"  ✅ Chunks created   : {len(chunks)}")
            avg_tokens = sum(c["token_count"] for c in chunks) // len(chunks) if chunks else 0
            print(f"  Avg tokens/chunk   : {avg_tokens}")
            paper_stats.append((paper_name, len(chunks)))

        except Exception as e:
            print(f"  ❌ ERROR: {type(e).__name__}: {e}")
            paper_stats.append((paper_name, 0))

    print(f"\n💾 Saving {len(all_chunks)} total chunks to processed/chunks.json ...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    output_kb = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"   ✅ Saved  ({output_kb:.1f} KB)")

    print(f"\n{'=' * 64}")
    print("  CHUNKING SUMMARY")
    print(f"{'=' * 64}")
    for name, count in paper_stats:
        bar = "█" * min(count // 10, 40)
        print(f"  {name:<40} {count:>4} chunks  {bar}")
    print(f"  {'TOTAL':<40} {len(all_chunks):>4} chunks")
    print(f"{'=' * 64}")

    if all_chunks:
        sample = all_chunks[len(all_chunks) // 2]
        print("\n📌 Sample Chunk (middle of dataset):")
        print(f"  chunk_id   : {sample['chunk_id']}")
        print(f"  paper_name : {sample['paper_name']}")
        print(f"  page_number: {sample['page_number']}")
        print(f"  token_count: {sample['token_count']}")
        print(f"  text preview:")
        preview = sample["text"][:300].replace("\n", " ")
        print(f"  \"{preview}...\"")

    print("\n  🎉 Task 5 complete! Chunks ready for embedding in Task 6.")


if __name__ == "__main__":
    main()
