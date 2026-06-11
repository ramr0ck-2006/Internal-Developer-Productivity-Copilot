import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# ---- Load .env ----
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# ---- Load FAISS vector store ----
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
faiss_db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)

# ---- Groq client ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Please set GROQ_API_KEY in your .env file")
groq_client = Groq(api_key=GROQ_API_KEY)


def answer_query(query, chat_history=None, threshold=0.8):
    """
    Retrieves relevant chunks with FAISS, filters by L2 distance threshold,
    uses optional conversation history, and generates an answer via Groq.
    """
    # 1. FAISS similarity search with scores
    docs_and_scores = faiss_db.similarity_search_with_score(query, k=3)

    # ---- DEBUG: print all scores to Streamlit Cloud logs ----
    print(f"\nQuery: {query}")
    for doc, score in docs_and_scores:
        print(f"  Score: {score:.4f}  |  Source: {doc.metadata['source']}")
    print("-" * 40)
    # ---- END DEBUG ----

    if not docs_and_scores:
        return "I don't have that information in my knowledge base.", []

    # 2. TEMPORARILY DISABLE THRESHOLD – return all retrieved docs
    # relevant = [(doc, score) for doc, score in docs_and_scores if score <= threshold]
    # if not relevant:
    #     return "I don't have that information in my knowledge base.", []
    # docs = [doc for doc, _ in relevant]
    docs = [doc for doc, _ in docs_and_scores]

    sources = list(set([doc.metadata["source"] for doc in docs]))
    context = "\n\n---\n\n".join([d.page_content for d in docs])

    # 3. System prompt + history
    system_prompt = {
        "role": "system",
        "content": (
            "You are an internal engineering assistant. "
            "Use ONLY the provided context excerpts to answer the question. "
            "If the answer cannot be found in the context, say exactly: "
            "'I don't have that information in my knowledge base.' "
            "Do not include any source references in your answer."
        )
    }

    messages = [system_prompt]

    if chat_history:
        for turn in chat_history[-6:]:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})

    user_message = f"""Context:
{context}

Question: {query}
Answer:"""
    messages.append({"role": "user", "content": user_message})

    # 4. Call Groq
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.2,
        max_tokens=500,
    )
    answer = response.choices[0].message.content
    return answer, sources