"""Extract text from research paper PDFs using PyMuPDF and save as .txt files."""

import fitz  # PyMuPDF
import os
import sys
from pathlib import Path

# --- Config & Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = PROJECT_ROOT / "papers"
OUTPUT_DIR = PROJECT_ROOT / "data" / "extracted_text"
FALLBACK_OUTPUT_DIR = PROJECT_ROOT / "extracted_text"
MIN_CHARS_THRESHOLD = 500  # warn if extracted text is suspiciously short

PAPERS = [
    ("attention_is_all_you_need_v7.pdf", "attention_is_all_you_need.txt", "Attention Is All You Need"),
    ("moshi_v1.pdf",                     "moshi.txt",                     "Moshi: A Speech-Speech Foundation Model"),
    ("dynamic_chunking_v2.pdf",          "dynamic_chunking.txt",          "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"),
]


def check_pdfs_exist():
    """Check all PDFs are present before starting. Exit if any are missing."""
    print("\n📋 Checking PDF files...")
    all_found = True
    for pdf_name, _, _ in PAPERS:
        pdf_path = PAPERS_DIR / pdf_name
        if pdf_path.exists():
            size_kb = pdf_path.stat().st_size / 1024
            print(f"   ✅ Found : {pdf_name} ({size_kb:.1f} KB)")
        else:
            print(f"   ❌ Missing: {pdf_name}  → Run papers/download_papers.py first!")
            all_found = False

    if not all_found:
        print("\n⛔ One or more PDFs are missing. Aborting.")
        sys.exit(1)
    print("   All PDFs present.\n")


def create_output_folder():
    """Create data/extracted_text/ folder if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FALLBACK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Verified output folder: {OUTPUT_DIR}\n")


def extract_text_from_pdf(pdf_path: str) -> tuple[str, int]:
    """Open a PDF, extract text from every page in order, return (full_text, page_count)."""
    page_texts = []
    doc = fitz.open(str(pdf_path))
    page_count = len(doc)

    for page_num in range(page_count):
        page = doc[page_num]
        text = page.get_text("text")

        # Strip null bytes and non-printable control characters, but preserve newlines
        cleaned = "".join(ch for ch in text if ch == "\n" or ch == "\t" or (ord(ch) >= 32))
        lines = [line.rstrip() for line in cleaned.splitlines()]
        cleaned = "\n".join(lines).strip()

        # Add explicit page boundary header for downstream chunking
        page_texts.append(f"--- Page {page_num + 1} of {page_count} ---\n{cleaned}")

    doc.close()
    full_text = "\n\n".join(page_texts)
    return full_text, page_count


def process_paper(pdf_name: str, txt_name: str, title: str):
    """Extract text from one PDF, validate length, and write to output file."""
    pdf_path = PAPERS_DIR / pdf_name
    txt_path = OUTPUT_DIR / txt_name
    fallback_txt_path = FALLBACK_OUTPUT_DIR / txt_name

    print(f"📄 Processing: {title}")
    print(f"   Input  : papers/{pdf_name}")

    full_text, page_count = extract_text_from_pdf(str(pdf_path))
    char_count = len(full_text)
    word_count = len(full_text.split())

    print(f"   Pages  : {page_count}")
    print(f"   Words  : {word_count:,}")
    print(f"   Chars  : {char_count:,}")

    if char_count < MIN_CHARS_THRESHOLD:
        print(f"   ⚠️  WARNING: Only {char_count} characters extracted!")
        print("      The PDF might be scanned or image-based.")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    with open(fallback_txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    size_kb = txt_path.stat().st_size / 1024
    print(f"   Output : {txt_path} ({size_kb:.1f} KB)")
    print("   Status : ✅ Done\n")


def print_summary():
    """Print final summary of all extracted files."""
    print("=" * 66)
    print("  EXTRACTION SUMMARY")
    print("=" * 66)
    total_words = 0
    total_chars = 0

    for _, txt_name, title in PAPERS:
        txt_path = OUTPUT_DIR / txt_name
        if txt_path.exists():
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read()
            words = len(content.split())
            chars = len(content)
            total_words += words
            total_chars += chars
            print(f"  ✅ {title}")
            print(f"     File  : {txt_name} | {words:,} words | {chars:,} chars")
        else:
            print(f"  ❌ {title} — output file missing!")

    print("-" * 66)
    print(f"  Total across all papers: {total_words:,} words | {total_chars:,} chars")
    print("=" * 66)


def main():
    print("\n" + "=" * 66)
    print("  THE RELAY — Task 4: Research Paper Text Extraction")
    print("=" * 66)

    check_pdfs_exist()
    create_output_folder()

    for pdf_name, txt_name, title in PAPERS:
        process_paper(pdf_name, txt_name, title)

    print_summary()
    print("\n🎉 Task 4 complete! All text extracted successfully.\n")


if __name__ == "__main__":
    main()

