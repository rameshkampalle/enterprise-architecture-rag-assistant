import os
import streamlit as st

from enterprise_rag.config import Settings
from enterprise_rag.answer import ask_question

st.set_page_config(page_title="Enterprise Architecture RAG Assistant", page_icon="🏗️")
st.title("🏗️ Enterprise Architecture Q&A")
st.caption("Ask questions from your enterprise architecture policy documents.")


@st.cache_resource
def get_settings():
    # Fill from Streamlit Secrets in deployment. Keep .env for local runs.
    for key in [
        "PINECONE_API_KEY",
        "GEMINI_API_KEY",
        "PINECONE_INDEX_NAME",
        "PINECONE_CLOUD",
        "PINECONE_REGION",
        "PINECONE_NAMESPACE",
        "PINECONE_INDEX_DIMENSION",
        "EMBEDDING_MODEL",
        "LLM_MODEL",
        "LLM_TEMPERATURE",
        "TOP_K_RETRIEVAL",
        "CORPUS_PATH",
    ]:
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])

    return Settings()


settings = get_settings()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.caption(f"Sources: {msg['sources']}")

query = st.chat_input("Ask a question...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.spinner("Searching knowledge base..."):
        result = ask_question(query, settings)
        answer = result.get("answer", "")
        sources = ", ".join(result.get("retrieved_sources", []))

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    with st.chat_message("assistant"):
        st.write(answer)
        if sources:
            st.caption(f"Sources: {sources}")
