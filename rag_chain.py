import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq

# ---- Load .env ----
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# ---- Vectorstore ----
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(
    collection_name="kubernetes_docs",
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# ---- Groq client ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Please set GROQ_API_KEY in your .env file")
groq_client = Groq(api_key=GROQ_API_KEY)


def answer_query(query, chat_history=None, threshold=0.5):
    """
    Retrieves relevant chunks, filters by relevance threshold,
    optionally includes conversation history, and generates an answer.
    """
    # 1. Retrieve with scores
    results = vectordb.similarity_search_with_score(query, k=3)

    # ---- DEBUG: show scores in terminal ----
    print(f"\nQuery: {query}")
    for doc, score in results:
        print(f"  Score: {score:.4f}  |  Source: {doc.metadata['source']}")
    print("-" * 40)
    # ---------------------------------------

    if not results:
        return "I don't have that information in my knowledge base.", []

    # 2. Filter by threshold
    relevant_docs = [(doc, score) for doc, score in results if score <= threshold]
    if not relevant_docs:
        return "I don't have that information in my knowledge base.", []

    docs = [doc for doc, _ in relevant_docs]
    sources = list(set([doc.metadata["source"] for doc in docs]))
    context = "\n\n---\n\n".join([d.page_content for d in docs])

    # 3. Build system prompt + history
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

    # 4. Call Groq LLM
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.2,
        max_tokens=500,
    )
    answer = response.choices[0].message.content
    return answer, sources


if __name__ == "__main__":
    # Simple test from command line
    q = "What is a Pod?"
    ans, srcs = answer_query(q)
    print("Answer:", ans)
    print("Sources:", srcs)