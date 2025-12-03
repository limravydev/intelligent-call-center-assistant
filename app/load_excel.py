import os
from pathlib import Path
import pandas as pd


def _row_to_text(row: pd.Series) -> str:
    """
    Turn a KB row into a single text block.
    Works with question/answer, notes, steps columns.
    """
    parts = []

    for col, val in row.items():
        if pd.isna(val):
            continue
        text = str(val).strip()
        if not text:
            continue

        col_lower = str(col).strip().lower()

        if "question" in col_lower:
            parts.append(f"Customer question: {text}")
        elif "answer" in col_lower:
            parts.append(f"Customer answer: {text}")
        elif "internal" in col_lower:
            parts.append(f"Internal notes: {text}")
        elif "step" in col_lower:
            parts.append(f"Steps: {text}")
        else:
            parts.append(f"{col}: {text}")

    return "\n".join(parts)


def _load_single_excel(path: Path):
    print(f"[Excel] Loading {path.name}")
    try:
        # Check extensions just in case
        if path.suffix.lower() not in [".xlsx", ".xls"]:
            print(f"[Excel] Skipping non-excel file: {path.name}")
            return []

        df = pd.read_excel(path)
        docs = []

        for idx, row in df.iterrows():
            full_text = _row_to_text(row)
            if not full_text:
                continue

            docs.append({
                "text": full_text,
                "metadata": {
                    "source": path.name,
                    "row_index": int(idx),
                    "category": "General Knowledge" 
                },
            })
        return docs
    except Exception as e:
        print(f"[Excel ERROR] Could not load {path.name}: {e}")
        return []


def load_excel_as_docs(path_or_dir: str):
    """
    Smart Loader:
    - If path_or_dir is a FILE -> loads that Excel file.
    - If path_or_dir is a DIR  -> loads ALL .xlsx AND .xls files inside.
    """
    p = Path(path_or_dir)
    all_docs = []

    if p.is_file():
        return _load_single_excel(p)

    if p.is_dir():
        # Grab both .xlsx and .xls
        # Note: glob isn't guaranteed order, so we sort them
        files = sorted([f for f in p.glob("*") if f.suffix.lower() in [".xlsx", ".xls"]])
        
        if not files:
            print(f"[Excel] No Excel files found in {p}")

        for f in files:
            all_docs.extend(_load_single_excel(f))
            
        print(f"[Excel] TOTAL docs from {p}: {len(all_docs)}")
        return all_docs

    print(f"[Excel] Warning: Path '{path_or_dir}' does not exist.")
    return []