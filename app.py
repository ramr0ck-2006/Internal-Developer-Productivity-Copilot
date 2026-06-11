import streamlit as st
from rag_chain import answer_query  # import function from previous file

st.set_page_config(page_title="EngBot - Internal Dev Copilot")
st.title("🚀 EngBot: Internal Dev Copilot")
st.markdown("Ask any technical question from our internal knowledge base (Kubernetes docs).")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
if query := st.chat_input("e.g., How do I create a pod?"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Get answer
    with st.spinner("Thinking..."):
        answer, sources = answer_query(query)

    # Format sources
    src_lines = "\n".join([f"- {s}" for s in sources])
    full_response = f"{answer}\n\n**Sources:**\n{src_lines}"

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    with st.chat_message("assistant"):
        st.markdown(full_response)