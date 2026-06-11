import os
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ---------- 1. SCRAPE ----------
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

# ---------- 2. CHUNKING ----------
print("\nLoading and chunking documents...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

all_texts = []
all_metadatas = []
for filename in os.listdir("data"):
    if filename.endswith(".txt"):
        with open(f"data/{filename}", "r", encoding="utf-8") as f:
            text = f.read()
        chunks = splitter.split_text(text)
        for chunk in chunks:
            all_texts.append(chunk)
            all_metadatas.append({"source": filename})

print(f"Total chunks: {len(all_texts)}")

# ---------- 3. EMBED & SAVE FAISS INDEX ----------
print("Initializing embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Creating FAISS index...")
faiss_db = FAISS.from_texts(all_texts, embeddings, metadatas=all_metadatas)

# Save to local directory
faiss_db.save_local("faiss_index")
print("FAISS index saved to ./faiss_index")