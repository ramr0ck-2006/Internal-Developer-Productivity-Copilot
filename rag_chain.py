import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq

# ---- Load .env explicitly ----
env_path = Path(__file__).parent / ".env"   # the .env file next to this script
print(f"Loading .env from: {env_path}")     # confirms path
load_dotenv(dotenv_path=env_path)

# 1. Vectorstore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(
    collection_name="kubernetes_docs",   # ← add this
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 2. Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Please set GROQ_API_KEY in your .env file")
groq_client = Groq(api_key=GROQ_API_KEY)

def answer_query(query):
    docs = vectordb.similarity_search(query, k=3)
    if not docs:
        return "I don't have that information in my knowledge base.", []

    context = "\n\n---\n\n".join([d.page_content for d in docs])
    sources = list(set([d.metadata["source"] for d in docs]))

    prompt = f"""You are an internal engineering assistant.
Answer the user's question based ONLY on the following context excerpts.
If the answer is not in the context, say exactly: "I don't have that information in my knowledge base."
Do not include any source references in your answer.

Context:
{context}

Question: {query}
Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500,
    )
    answer = response.choices[0].message.content
    return answer, sources

if __name__ == "__main__":
    q = "What is a Pod?"
    ans, srcs = answer_query(q)
    print("Answer:", ans)
    print("Sources:", srcs)