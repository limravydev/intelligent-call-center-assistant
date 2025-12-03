# 📘 Intelligent Call Center Assistant

An end-to-end Retrieval-Augmented Generation (RAG) copilot built for bank call-center agents. It ingests Excel policies and PDF playbooks, indexes them with SentenceTransformers + ChromaDB, and uses Gemini to draft grounded responses with clear agent action steps.

---

## ✨ Key Capabilities

- **Multi-source ingestion:** Cleans Excel sheets and PDF manuals, chunks intelligently, and keeps metadata for traceability.
- **Vector search:** MiniLM embeddings persisted in ChromaDB for fast, top-*k* semantic retrieval.
- **Structured answers:** Gemini prompt produces three sections—Customer Answer, Internal Notes, Agent Steps—plus smalltalk fallback and Khmer detection.
- **Agent-ready UI:** Modern Streamlit interface with context viewer and conversation history.
- **Quality guardrails:** Automated evaluation for accuracy, completeness, hallucination rate, and procedural clarity.

---

## 🧱 Architecture at a Glance

1. **Ingestion:** `load_excel.py` and `load_pdf.py` normalize source files and create text chunks with metadata.
2. **Embedding:** `embeddings.py` loads MiniLM from SentenceTransformers to convert text → vectors.
3. **Storage:** `rag.py` builds/loads a persistent ChromaDB collection under `chroma_excel_rag/`.
4. **Retrieval:** `chatbot.py` queries the store for the most relevant chunks.
5. **Synthesis:** Gemini receives the prompt template + retrieved evidence and returns a structured response.
6. **Experience:** `ui.py` (Streamlit) streams the chat, shows citations, and surfaces follow-up actions.

---

## 📁 Repository Layout

```
app/
 ├── chatbot.py         # RAG orchestration + Gemini prompt engineering
 ├── embeddings.py      # MiniLM loader & caching helpers
 ├── load_excel.py      # Excel ingestion & cleaning
 ├── load_pdf.py        # PDF parsing and chunking
 ├── rag.py             # Index build + retrieval utilities
 └── ui.py              # Streamlit front end
chroma_excel_rag/       # Persistent ChromaDB store
data/
 ├── excel/             # Drop Excel knowledge base files here
 └── pdf/               # Drop PDF policy manuals here
evaluation/
 ├── evaluate_rag_auto.py
 ├── evaluate_rag.py
 └── plot_eval.py
debug/                  # Optional scripts for inspecting Excel/PDF/Chroma
requirements.txt
README.md
```

---

## 🚀 Getting Started

### 1. Clone and install

```bash
git clone <your_repo_link>
cd call-center-rag
python -m venv venv
source venv/bin/activate  # macOS / Linux
# venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

### 2. Configure secrets

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Keys are available from https://ai.google.dev.

### 3. Prepare the knowledge base

1. Place Excel policy files in `data/excel/`.
2. Place PDF manuals in `data/pdf/`.
3. (Optional) use the scripts in `debug/` to validate parsing before embedding.

### 4. Build or refresh embeddings

```bash
python -c "from app.rag import build_or_load_index; build_or_load_index(rebuild=True)"
```

This step cleans the sources, chunks text, generates embeddings, and persists them to `chroma_excel_rag/`.

### 5. Launch the Streamlit assistant

```bash
streamlit run app/ui.py
```

Navigate to http://localhost:8501 to start chatting. Each response shows grounding snippets, internal guidance, and action steps.

---

## 🧪 Evaluation & QA

Run the automated benchmark to score accuracy, completeness, hallucination rate, and step clarity:

```bash
python evaluation/evaluate_rag_auto.py
```

Artifacts include:

- `eval_results.csv` – per-question metrics
- `evaluation_summary.png` – visual summary
- Optional manual evaluation via `evaluation/evaluate_rag.py`

---

## 🧰 Debugging Utilities

- `debug/debug_excel.py` – Inspect cleaned Excel rows and chunks.
- `debug/debug_pdf.py` – Preview PDF parsing and chunk generation.
- `debug/debug_chroma.py` – Peek into stored documents, embeddings, and metadata.

Use these scripts to verify data quality before running a full rebuild.

---

## 🧠 Technology Stack

- **LLM:** Gemini API (Google AI Studio)
- **Vector store:** ChromaDB persistent client
- **Embeddings:** MiniLM SentenceTransformer
- **App framework:** Streamlit
- **Data wrangling:** pandas, PyPDF

---

## 🤝 Contributing

1. Fork the repo and create a feature branch.
2. Keep documentation and evaluation results up to date with any behavioral changes.
3. Open a pull request describing the motivation, approach, and verification steps.

---

## 📄 License

Academic use only unless otherwise specified.

---

## 🙌 Credits

Developed for **MSAI 550 – Generative AI** using ChromaDB, SentenceTransformers, Gemini API, and Streamlit.
