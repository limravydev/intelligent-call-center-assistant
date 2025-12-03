# debug_chroma.py
from app.rag import build_or_load_index, get_collection
from app.embeddings import get_embedding_model

# Rebuild to be sure DB is fresh
collection, embed_model = build_or_load_index(
    excel_path="data/excel/call_center_rag.xlsx",
    pdf_dir="data/pdf",
    rebuild=True,
)

print("Total documents in collection:", collection.count())

# Peek at some docs
peek = collection.peek(5)  # first 5 entries

print("\n=== Peek documents ===")
for i in range(len(peek["ids"])):
    print(f"--- Doc {i} ---")
    print("id:", peek["ids"][i])
    print("metadata:", peek["metadatas"][i])
    print("text preview:", peek["documents"][i][:300])
    print()

# Optional: run a manual query to see what gets retrieved
question = "What information is required on a withdrawal slip?"
q_emb = embed_model.encode([question])[0].tolist()
results = collection.query(
    query_embeddings=[q_emb],
    n_results=3,
    include=["documents", "metadatas", "distances"],
)

print("\n=== Query results ===")
for i in range(len(results["documents"][0])):
    print(f"Result {i}")
    print("distance:", results["distances"][0][i])
    print("metadata:", results["metadatas"][0][i])
    print("text preview:", results["documents"][0][i][:300])
    print()