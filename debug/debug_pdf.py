# debug_pdf.py
from app.load_pdf import load_all_pdfs

PDF_DIR = "data/pdf"

docs = load_all_pdfs(PDF_DIR, default_category="Policy")
print(f"Total PDF chunks: {len(docs)}\n")

for i, d in enumerate(docs[:5]):  # show first 5
    print(f"--- PDF doc {i} ---")
    print("id:", d["id"])
    print("metadata:", d["metadata"])
    print("text preview:")
    print(d["text"][:500])
    print()