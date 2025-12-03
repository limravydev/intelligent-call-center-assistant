# debug_excel.py
from app.load_excel import load_excel_as_docs

EXCEL_PATH = "data/excel/call_center_rag.xlsx"

docs = load_excel_as_docs(EXCEL_PATH)
print(f"Total Excel docs: {len(docs)}\n")

for i, d in enumerate(docs[:5]):  # show first 5
    print(f"--- Excel doc {i} ---")
    print("id:", d["id"])
    print("metadata:", d["metadata"])
    print("text preview:")
    print(d["text"][:500])
    print()