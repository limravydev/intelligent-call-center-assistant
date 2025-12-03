import os
import chromadb

from app.embeddings import get_embedding_model, embed_texts
from app.load_excel import load_excel_as_docs
from app.load_pdf import load_all_pdfs  # uses your existing load_pdf.py


# Paths / constants
EXCEL_DIR = "data/excel"          # folder containing all your KB .xlsx files
PDF_DIR = "data/pdf"              # folder with policy PDFs
DB_DIR = "./chroma_excel_rag"
COLLECTION_NAME = "excel_kb"


def get_collection():
    """
    Get or create the Chroma collection.
    """
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    return collection


def build_or_load_index(
    excel_path: str = EXCEL_DIR,
    pdf_dir: str = PDF_DIR,
    rebuild: bool = False,
):
    """
    Build the index from Excel + PDF if needed.

    - If collection exists and rebuild=False  -> reuse it.
    - If rebuild=True or collection is empty -> clear and re-index.
    """
    collection = get_collection()
    embed_model = get_embedding_model()
    existing_count = collection.count()

    if existing_count > 0 and not rebuild:
        print(f"[RAG] Using existing Chroma collection with {existing_count} documents.")
        return collection, embed_model

    print("[RAG] Building new Chroma index from Excel/PDF...")

    # clear old data if any
    if existing_count > 0:
        existing = collection.get()
        if existing and "ids" in existing:
            if existing["ids"]:
                collection.delete(ids=existing["ids"])

    # ---- 1) Load Excel KB docs ----
    docs_excel = load_excel_as_docs(excel_path)  # we’ll update load_excel.py to accept a folder
    print(f"[RAG] Loaded {len(docs_excel)} docs from Excel.")

    # ---- 2) Load PDF docs (optional but recommended) ----
    docs_pdf = []
    if os.path.isdir(pdf_dir):
        try:
            docs_pdf = load_all_pdfs(pdf_dir)
            print(f"[RAG] Loaded {len(docs_pdf)} docs from PDFs.")
        except Exception as e:
            print(f"[RAG] Warning: could not load PDFs from {pdf_dir}: {e}")

    docs = docs_excel + docs_pdf
    if not docs:
        print("[RAG] WARNING: no documents loaded from Excel/PDF.")
        return collection, embed_model

    # ---- 3) Prepare fields for Chroma ----
    texts = [d["text"] for d in docs]
    metadatas = [d.get("metadata", {}) for d in docs]

    # Generate simple sequential IDs instead of assuming docs already have 'id'
    ids = [f"doc-{i}" for i in range(len(texts))]

    # compute embeddings
    embeddings = embed_texts(texts)

    # ---- 4) Add to Chroma ----
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"[RAG] Indexed {len(texts)} documents.")
    return collection, embed_model


def retrieve_context(question: str, collection, embed_model, k: int = 5):
    """
    Retrieve the top-k most relevant chunks for a question.

    Returns:
      - context_docs: list of (text, metadata, distance)
      - best_distance: float or None
    """
    # Embed the question
    q_emb = embed_model.encode([question])[0].tolist()

    # Ask Chroma for documents, metadata, and distances
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    context_docs = []
    best_distance = None

    if dists:
        best_distance = min(dists)

    for doc, meta, dist in zip(docs, metas, dists):
        context_docs.append((doc, meta, dist))

    return context_docs, best_distance