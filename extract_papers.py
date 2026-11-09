""" Extract text from research paper PDFs using PyMuPDF and save as .txt files."""

import fitz  # PyMuPDF
import os
import sys

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAPERS_DIR = os.path.join(BASE_DIR, "papers")
OUTPUT_DIR = os.path.join(BASE_DIR, "extracted_text")
MIN_CHARS_THRESHOLD = 500  # warn if extracted text is suspiciously short

PAPERS = [
    ("attention_is_all_you_need_v7.pdf", "attention_is_all_you_need.txt", "Attention Is All You Need"),
    ("moshi_v1.pdf",                     "moshi.txt",                     "Moshi: A Speech-Speech Foundation Model"),
    ("dynamic_chunking_v2.pdf",          "dynamic_chunking.txt",          "Dynamic Chunking for RAG"),
]


def check_pdfs_exist():
    """Check all PDFs are present before starting. Exit if any are missing."""
    print("\n📋 Checking PDF files...")
    all_found = True
    for pdf_name, _, _ in PAPERS:
        pdf_path = os.path.join(PAPERS_DIR, pdf_name)
        if os.path.exists(pdf_path):
            size_kb = os.path.getsize(pdf_path) / 1024
            print(f"   ✅ Found : {pdf_name} ({size_kb:.1f} KB)")
        else:
            print(f"   ❌ Missing: {pdf_name}  → Run papers/download_papers.py first!")
            all_found = False

    if not all_found:
        print("\n⛔ One or more PDFs are missing. Aborting.")
        sys.exit(1)
    print("   All PDFs present.\n")


def create_output_folder():
    """Create extracted_text/ folder if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print("📁 Created folder: extracted_text/\n")
    else:
        print("📁 Output folder already exists: extracted_text/\n")


def extract_text_from_pdf(pdf_path: str) -> tuple[str, int]:
    """Open a PDF, extract text from every page in order, return (full_text, page_count)."""
    page_texts = []
    with fitz.open(pdf_path) as doc:
        page_count = len(doc)
        for i in range(page_count):
            header = f"\n--- Page {i + 1} of {page_count} ---\n"
            page_texts.append(header + doc[i].get_text("text"))
    return "\n".join(page_texts), page_count


def save_text(text: str, output_path: str):
    """Save text to a UTF-8 encoded .txt file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def process_paper(pdf_name: str, txt_name: str, title: str) -> bool:
    """Extract text from one PDF and save it. Returns True on success."""
    print(f"{'=' * 62}")
    print(f"  📄 {title}")
    print(f"{'=' * 62}")
    print(f"  Input  : papers/{pdf_name}")
    print(f"  Output : extracted_text/{txt_name}")

    pdf_path = os.path.join(PAPERS_DIR, pdf_name)
    output_path = os.path.join(OUTPUT_DIR, txt_name)

    try:
        print("  Extracting text...")
        full_text, page_count = extract_text_from_pdf(pdf_path)
        char_count = len(full_text)

        if char_count == 0:
            print("  ❌ ERROR: No text extracted! PDF may be image-only (scanned).")
            return False

        if char_count < MIN_CHARS_THRESHOLD:
            print(f"  ⚠️  WARNING: Only {char_count} characters — PDF might be image-based or corrupted.")

        save_text(full_text, output_path)
        size_kb = os.path.getsize(output_path) / 1024

        print(f"  ✅ Extraction complete!")
        print(f"  📊 Pages extracted   : {page_count}")
        print(f"  📊 Total characters  : {char_count:,}")
        print(f"  📊 Output file size  : {size_kb:.1f} KB\n")
        return True

    except fitz.FileNotFoundError:
        print(f"  ❌ ERROR: PDF not found at {pdf_path}")
    except fitz.FileDataError as e:
        print(f"  ❌ ERROR: PDF corrupted or unreadable: {e}")
    except PermissionError:
        print(f"  ❌ ERROR: Permission denied reading {pdf_path}")
    except Exception as e:
        print(f"  ❌ ERROR: {type(e).__name__}: {e}")
    return False


def main():
    print("\n" + "=" * 62)
    print("  THE RELAY — Task 4: PDF Text Extraction")
    print("=" * 62)

    check_pdfs_exist()
    create_output_folder()

    results = []
    for pdf_name, txt_name, title in PAPERS:
        success = process_paper(pdf_name, txt_name, title)
        results.append((txt_name, success))

    # Summary table
    print(f"\n{'=' * 62}")
    print("  EXTRACTION SUMMARY")
    print(f"{'=' * 62}")
    all_ok = True
    for txt_name, success in results:
        output_path = os.path.join(OUTPUT_DIR, txt_name)
        status = "✅ SUCCESS" if success else "❌ FAILED "
        if success and os.path.exists(output_path):
            size_kb = os.path.getsize(output_path) / 1024
            print(f"  {status} | {txt_name:<35} | {size_kb:>7.1f} KB")
        else:
            print(f"  {status} | {txt_name}")
            all_ok = False
    print(f"{'=' * 62}")

    if all_ok:
        print("\n  🎉 All 3 papers extracted successfully!")
        print("  Text files saved in: extracted_text/")
    else:
        failed = [n for n, ok in results if not ok]
        print(f"\n  ⚠️  {len(failed)} extraction(s) failed. See errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
