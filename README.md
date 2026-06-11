# 🧠 EngBot – Internal Developer Copilot (RAG)

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-red?logo=streamlit)](https://your-demo-link.streamlit.app)  <!-- Replace with your real URL -->
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **Retrieval‑Augmented Generation (RAG)** chatbot that acts like an internal engineering assistant.  
Ask natural‑language questions about your infrastructure, and get **accurate, sourced answers** – only from the documents you’ve ingested.

---

## 🔍 What problem does it solve?

Engineers waste **15–20% of their time** hunting for answers in internal wikis, runbooks, and documentation.  
This bot ingests that knowledge, retrieves the most relevant chunks, and lets a state‑of‑the‑art LLM generate a precise answer **with citations**.

For this demo, I scraped **Kubernetes documentation** – but you can point it at any internal docs, engineering blogs, or incident post‑mortems.

---

## ✨ Features

- **RAG pipeline** – Chunking, embedding (all‑MiniLM‑L6‑v2), vector search (ChromaDB), and generation via Groq’s Llama 3.1 (free tier).
- **Grounded answers** – The LLM is forced to use only retrieved context; off‑topic questions are refused.
- **Transparent citations** – Every answer shows exactly which source files were used.
- **Interactive UI** – Clean chat interface built with Streamlit, ready for demos.
- **Lightweight & free** – No GPU needed; runs on CPU with free Groq API and local embeddings.

---

User Query
│
▼
┌─────────────┐ Similarity Search ┌──────────────┐
│ Streamlit │ ──────────────────────▶ │ ChromaDB │
│ Chat UI │ ◀── Top‑k chunks ───── │ (vector store) │
└─────────────┘ └──────────────┘
│ ▲
│ Retrieved context + Prompt │ Embeddings
▼ │
┌─────────────┐ ┌──────────────┐
│ Groq LLM │ │ Ingest.py │
│ (Llama 3.1) │ │ (scrape + │
│ Generates │ │ chunk + │
│ Answer │ │ embed) │
└─────────────┘ └──────────────┘


---

## 🧰 Tech Stack

| Component          | Technology                          |
|--------------------|-------------------------------------|
| Embedding Model    | `all-MiniLM-L6-v2` (HuggingFace)    |
| Vector Database    | ChromaDB                            |
| LLM                | Llama 3.1 8B via Groq Cloud (free)  |
| Orchestration      | LangChain (minimal)                 |
| Frontend           | Streamlit                           |
| Data (demo)        | Kubernetes official documentation   |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/engbot.git
cd engbot

2. Create & activate a virtual environment
bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
3. Install dependencies
bash
pip install -r requirements.txt
4. Set up Groq API key
Get a free key at console.groq.com

Create a .env file in the project root:

text
GROQ_API_KEY=gsk_your_key_here
5. Ingest documents (Kubernetes docs demo)
bash
python ingest.py
This scrapes a few Kubernetes pages, chunks them, embeds them, and stores them in ./chroma_db.

6. Launch the app
bash
streamlit run app.py
Open http://localhost:8501 and start asking questions! 

## 🏗️ Architecture
