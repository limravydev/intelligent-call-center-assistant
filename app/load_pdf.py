import os
import glob
from pathlib import Path
import pdfplumber

def _is_english_line(line: str) -> bool:
    """Heuristic: line contains A–Z letters."""
    for c in line:
        if "a" <= c.lower() <= "z":
            return True
    return False

def _has_chinese(line: str) -> bool:
    """Detect if a line contains Chinese characters."""
    return any("\u4e00" <= c <= "\u9fff" for c in line)

def extract_english_block(page_text: str) -> str:
    """
    Extract only the English section from one PDF page.
    Stops collecting if Chinese characters (bottom block) are detected.
    """
    lines = [l.strip() for l in page_text.splitlines()]
    english_lines = []
    started = False

    for line in lines:
        if not line:
            continue

        if _has_chinese(line):
            # Stop if we hit the Chinese block (usually at the bottom)
            if started:
                break
            else:
                continue

        # Start collecting if we see English characters
        if _is_english_line(line):
            started = True

        if started:
            english_lines.append(line)

    return "\n".join(english_lines).strip()

def _load_single_pdf(pdf_path: Path, default_category: str):
    """Helper to load a single PDF file."""
    print(f"[PDF] Reading {pdf_path.name}")
    docs = []
    
    # Simple subcategory from filename
    # e.g. "6. Terms and Conditions.pdf" -> "Terms and Conditions"
    base_name = pdf_path.stem # filename without extension
    
    # Remove leading numbers like "6. "
    name = base_name
    if ". " in name:
        parts = name.split(". ", 1)
        if len(parts) > 1:
            name = parts[1]
    
    # Remove prefix "Terms and Conditions for " if present
    prefix = "Terms and Conditions for "
    if name.startswith(prefix):
        name = name[len(prefix):]
        
    subcategory = name.strip() or base_name

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                raw_text = page.extract_text() or ""
                english_text = extract_english_block(raw_text)

                if not english_text:
                    continue

                docs.append({
                    "id": f"{pdf_path.name}_page_{i+1}",
                    "text": english_text,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": i + 1,
                        "category": default_category,
                        "subcategory": subcategory,
                    },
                })
    except Exception as e:
        print(f"[PDF ERROR] Failed to read {pdf_path.name}: {e}")
    
    return docs

def load_all_pdfs(path_or_dir: str, default_category: str = "Policy"):
    """
    Smart Loader:
    - If path_or_dir is a FILE -> loads just that PDF.
    - If path_or_dir is a DIR  -> loads all .pdf files inside.
    """
    p = Path(path_or_dir)
    all_docs = []

    if p.is_file():
        if p.suffix.lower() == ".pdf":
            all_docs.extend(_load_single_pdf(p, default_category))
    
    elif p.is_dir():
        # Glob all .pdf files (case insensitive if possible, but usually lowercase)
        pdf_files = sorted(p.glob("*.pdf"))
        if not pdf_files:
            print(f"[PDF] No PDFs found in {p}")
        
        for pdf_file in pdf_files:
            all_docs.extend(_load_single_pdf(pdf_file, default_category))
            
        print(f"[PDF] Loaded {len(all_docs)} chunks from {len(pdf_files)} PDF files.")
    
    else:
        print(f"[PDF] Warning: Path '{path_or_dir}' does not exist.")

    return all_docs