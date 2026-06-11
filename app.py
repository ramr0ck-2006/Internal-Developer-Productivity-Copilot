import streamlit as st
from rag_chain import answer_query
from feedback import log_feedback

# ---- Page config ----
st.set_page_config(
    page_title="EngBot – Internal Dev Copilot",
    page_icon="🤖",
    layout="wide"
)

# ---- Custom CSS for professional look ----
st.markdown("""
<style>
.chat-message {
    padding: 1rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.user-message {
    background-color: #e8f0fe;
    border-left: 4px solid #1a73e8;
}
.assistant-message {
    background-color: #f5f5f5;
    border-left: 4px solid #34a853;
}
.sources-box {
    background: #fff3e0;
    padding: 0.5rem;
    border-radius: 8px;
    font-size: 0.85rem;
    margin-top: 0.5rem;
    border-left: 3px solid #ff9800;
}
.sources-box summary {
    font-weight: bold;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# ---- Header ----
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("🤖 EngBot – Internal Developer Copilot")
    st.caption("Ask technical questions based on the ingested Kubernetes docs.")
with col2:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---- Initialize session state ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Display chat history ----
for msg in st.session_state.messages:
    role = msg["role"]
    css_class = "user-message" if role == "user" else "assistant-message"
    with st.container():
        st.markdown(f'<div class="chat-message {css_class}">', unsafe_allow_html=True)
        st.markdown(msg["content"])
        if role == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("📚 Sources & Scores"):
                if "scores" in msg and msg["scores"]:
                    st.markdown("**Top‑3 L2 distances (lower = better):**")
                    for i, (src, sc) in enumerate(zip(msg["sources"], msg["scores"])):
                        st.markdown(f"- Chunk {i+1}: source `{src}` — score `{sc}`")
                else:
                    for src in msg["sources"]:
                        st.markdown(f"- {src}")
        st.markdown('</div>', unsafe_allow_html=True)

# ---- Chat input ----
if prompt := st.chat_input("e.g., How do I expose a Kubernetes service?"):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build conversation history from previous turns
    history = []
    msgs = st.session_state.messages
    for i in range(0, len(msgs)-1, 2):
        if msgs[i]["role"] == "user" and i+1 < len(msgs) and msgs[i+1]["role"] == "assistant":
            history.append({
                "user": msgs[i]["content"],
                "assistant": msgs[i+1]["content"]
            })

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            answer, sources, scores = answer_query(prompt, chat_history=history)
        if not answer.strip():
            answer = "I couldn't generate an answer. Please rephrase your question."
        st.markdown(answer)
        if sources or scores:
            with st.expander("📚 Sources & Scores"):
                if scores:
                    st.markdown("**Top‑3 L2 distances (lower = better):**")
                    for i, (src, sc) in enumerate(zip(sources, scores)):
                        st.markdown(f"- Chunk {i+1}: source `{src}` — score `{sc}`")
                else:
                    for src in sources:
                        st.markdown(f"- {src}")

    # Store assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "scores": scores  # store for history display
    })

    # ---- Feedback buttons ----
    col1, col2, col3 = st.columns([0.05, 0.05, 0.9])
    with col1:
        if st.button("👍", key=f"up_{len(st.session_state.messages)}"):
            log_feedback(prompt, answer, sources, "positive")
            st.toast("Thanks for the positive feedback!")
    with col2:
        if st.button("👎", key=f"down_{len(st.session_state.messages)}"):
            log_feedback(prompt, answer, sources, "negative")
            st.toast("Thanks! We'll improve.")