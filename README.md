# DocQA — RAG-based Document Q&A

Ask questions about any PDF and get accurate, grounded answers with source citations.

Built with Flask, LangChain, Gemini Embeddings, Groq (Llama 3.1), and FAISS.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## What it does

Upload any PDF — research paper, textbook, report — and ask questions about it in natural language. The app finds the most relevant sections and generates a concise answer, along with citations showing exactly which pages the answer came from.

It uses **Retrieval-Augmented Generation (RAG)** to ground every answer in the document, preventing the model from hallucinating information that isn't there.

---

## Demo

> Upload a PDF → Ask a question → Get a cited answer

![Demo](demo.png)

---

## How it works

```
PDF upload
   │
   ▼
Text chunking (LangChain RecursiveCharacterTextSplitter)
   │
   ▼
Vector embeddings (Gemini embedding-001)
   │
   ▼
FAISS vector store (in-memory similarity index)
   │
   ▼
User asks a question
   │
   ▼
Similarity search → top 3 relevant chunks retrieved
   │
   ▼
Chunks + question → Groq (Llama 3.1) → grounded answer + page citations
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Flask |
| AI orchestration | LangChain |
| Embeddings | Google Gemini (`gemini-embedding-001`) |
| LLM | Groq — Llama 3.1 8B Instant |
| Vector store | FAISS (in-memory) |
| Frontend | Vanilla HTML/CSS/JS |

---

## Getting started

### Prerequisites
- Python 3.9+
- A [Gemini API key](https://aistudio.google.com) (free)
- A [Groq API key](https://console.groq.com) (free)

### Installation

```bash
# Clone the repo
git clone https://github.com/vanshika2608/rag-doc-qa.git
cd rag-doc-qa

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the root directory:
```
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
```

### Run

```bash
python app.py
```

Open `http://localhost:5001` in your browser.

---

## Project structure

```
rag-doc-qa/
├── app.py                 # Flask routes and app entry point
├── rag/
│   ├── loader.py          # PDF loading and text chunking
│   ├── embedder.py        # Gemini embeddings + FAISS index creation
│   └── chain.py           # LangChain LCEL pipeline + Groq LLM
├── templates/
│   └── index.html         # Chat UI
├── requirements.txt
└── .env                   # API keys (never committed)
```

---

## Key concepts

**Why RAG?**
Standard LLMs hallucinate when asked about specific documents they haven't seen. RAG solves this by retrieving relevant text from the document at query time and passing it as context to the model — grounding the answer in real content.

**Why FAISS?**
FAISS is an in-memory vector database that enables fast cosine similarity search across document chunks. No external database server needed — perfect for a self-contained project.

**Why Groq?**
Groq's inference hardware runs Llama 3.1 significantly faster than standard cloud providers, with a generous free tier — ideal for a project that needs low-latency responses.

---

## Limitations

- FAISS index is in-memory and resets on server restart (a persistent DB like ChromaDB or Pinecone would fix this)
- One document at a time — multi-document support would require session management
- Large PDFs (100+ pages) may be slow to index on first upload

---

## Future improvements

- [ ] Persist FAISS index to disk between sessions
- [ ] Multi-document support
- [ ] Streaming responses
- [ ] Deploy to Hugging Face Spaces

---

## Author

Vanshika Deswal — [GitHub](https://github.com/vanshika2608)
