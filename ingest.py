import os
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ========== 1. SCRAPE (same as above) ==========
urls = [
    "https://kubernetes.io/docs/concepts/overview/",
    "https://kubernetes.io/docs/concepts/architecture/",
    "https://kubernetes.io/docs/concepts/workloads/pods/",
]
os.makedirs("data", exist_ok=True)

for i, url in enumerate(urls):
    try:
        print(f"Scraping {url} ...")
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        content_div = soup.find("div", class_="td-content")
        if content_div is None:
            text = soup.body.get_text(separator="\n", strip=True)
        else:
            text = content_div.get_text(separator="\n", strip=True)
        with open(f"data/doc_{i}.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  -> saved ({len(text)} chars)")
    except Exception as e:
        print(f"  !! Failed: {e}")

# ========== 2. CHUNKING ==========
print("\nLoading and chunking documents...")
all_chunks = []
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # each chunk ~500 characters
    chunk_overlap=50,     # overlap to avoid cutting sentences
    separators=["\n\n", "\n", " ", ""]
)

for filename in os.listdir("data"):
    if filename.endswith(".txt"):
        with open(f"data/{filename}", "r", encoding="utf-8") as f:
            text = f.read()
        # Split this document into chunks
        chunks = splitter.split_text(text)
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": filename
            })

print(f"Total chunks created: {len(all_chunks)}")

# ========== 3. EMBED & STORE IN CHROMA DB ==========
print("Initializing embedding model (this may take a moment the first time)...")
# all-MiniLM-L6-v2 is small (80 MB) and runs on CPU, very fast
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Storing chunks in ChromaDB...")
# This creates a folder `chroma_db/` in your project
vectordb = Chroma(
    collection_name="kubernetes_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Add all chunks with metadata
texts = [c["text"] for c in all_chunks]
metadatas = [{"source": c["source"]} for c in all_chunks]

# If you have many chunks, you can add in batches, but for a few hundred it's fine to add all at once
vectordb.add_texts(texts=texts, metadatas=metadatas)

# Persist (optional, Chroma automatically saves)
vectordb.persist()
print(f"Ingestion complete. Your vector database is ready in ./chroma_db")