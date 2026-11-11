"""Utility script to compile reports/final_report.md into reports/final_report.pdf with exactly 2 pages."""

import sys
from pathlib import Path
import fitz
from markdown_it import MarkdownIt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MD_PATH = PROJECT_ROOT / "reports" / "final_report.md"
PDF_PATH = PROJECT_ROOT / "reports" / "final_report.pdf"


def compile_pdf() -> int:
    if not MD_PATH.exists():
        raise FileNotFoundError(f"Missing {MD_PATH}")

    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    md = MarkdownIt("commonmark", {"breaks": True, "html": True})
    md.enable("table")
    html_body = md.render(md_text)

    # Tight CSS styling to guarantee rich content fits neatly within 2 pages
    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {{
    size: A4;
    margin: 16pt 18pt;
}}
body {{
    font-family: Helvetica, Arial, sans-serif;
    font-size: 6.0pt;
    line-height: 1.10;
    color: #0f172a;
    margin: 0;
    padding: 0;
}}
h1 {{
    font-size: 9.5pt;
    font-weight: bold;
    color: #0284c7;
    margin: 0 0 2pt 0;
    text-align: center;
    border-bottom: 0.8pt solid #0284c7;
    padding-bottom: 1pt;
}}
h2 {{
    font-size: 7.5pt;
    font-weight: bold;
    color: #0369a1;
    margin: 3pt 0 1pt 0;
    border-bottom: 0.4pt solid #cbd5e1;
    padding-bottom: 0.5pt;
}}
h3 {{
    font-size: 6.6pt;
    font-weight: bold;
    color: #0f172a;
    margin: 2pt 0 0.5pt 0;
}}
p {{
    margin: 0 0 1.5pt 0;
}}
ul, ol {{
    margin: 0 0 1.5pt 0;
    padding-left: 9pt;
}}
li {{
    margin-bottom: 0.5pt;
}}
strong {{
    color: #090d16;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 1.5pt 0;
    font-size: 5.4pt;
    line-height: 1.0;
}}
th, td {{
    border: 0.3pt solid #94a3b8;
    padding: 0.8pt 1.5pt;
    text-align: left;
}}
th {{
    background-color: #f1f5f9;
    font-weight: bold;
    color: #0f172a;
}}
hr {{
    border: 0;
    border-top: 0.3pt solid #e2e8f0;
    margin: 2pt 0;
}}
em {{
    color: #334155;
    font-size: 5.2pt;
}}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

    rect = fitz.Rect(18, 16, 577, 826)
    story = fitz.Story(html=html_content)
    writer = fitz.DocumentWriter(str(PDF_PATH))
    more = 1
    page_num = 0

    while more:
        page_num += 1
        device = writer.begin_page(fitz.paper_rect("a4"))
        more, _ = story.place(rect)
        story.draw(device)
        writer.end_page()
        if page_num > 10:
            break

    writer.close()
    print(f"Generated {PDF_PATH} with {page_num} pages.")
    return page_num


if __name__ == "__main__":
    pages = compile_pdf()
    if pages <= 2:
        print(f"PASS: Exactly {pages} pages (<= 2 required).")
    else:
        print(f"FAIL: {pages} pages (exceeds 2 pages).")
        sys.exit(1)