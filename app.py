"""Streamlit chatbot UI.   Run with:  streamlit run app.py"""
import streamlit as st

import rag_pipeline as rag

st.set_page_config(page_title="AI Document Assistant", page_icon="📄")
st.title("📄 AI Document Assistant")
st.caption("Ask questions about your PDF documents. Every answer shows its sources.")


@st.cache_resource(show_spinner="Loading the document index...")
def get_store():
    return rag.load_vectorstore()


# ---------------------------------------------------------------- sidebar: add documents
with st.sidebar:
    st.header("Documents")
    existing = sorted(p.name for p in rag.DATA_DIR.glob("*.pdf"))
    st.write("Indexed files:" if existing else "No documents yet.")
    for name in existing:
        st.write(f"- {name}")

    uploads = st.file_uploader("Add PDF documents", type="pdf", accept_multiple_files=True)
    if uploads and st.button("Add and rebuild index"):
        rag.DATA_DIR.mkdir(parents=True, exist_ok=True)
        for f in uploads:
            (rag.DATA_DIR / f.name).write_bytes(f.getvalue())
        try:
            with st.spinner("Processing documents..."):
                stats = rag.build_vectorstore()
            st.cache_resource.clear()
            st.success(f"Indexed {stats['documents']} document(s), {stats['chunks']} chunks.")
            st.rerun()
        except Exception as error:
            st.error(f"Could not process the documents: {error}")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------- load index (missing-docs handling)
try:
    store = get_store()
except FileNotFoundError:
    st.warning("No document index found. Add a PDF in the sidebar, or run `python build_index.py`.")
    st.stop()
except Exception as error:
    st.error(f"Could not load the document index: {error}")
    st.stop()

# ---------------------------------------------------------------- chat
if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources):
    if sources:
        with st.expander("Sources"):
            for name, page in sources:
                st.write(f"📄 {name} — page {page}")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        show_sources(msg.get("sources"))

question = st.chat_input("Ask a question about your documents")
if question is not None:
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            result = rag.ask_rag(question, store)
        if result["error"]:
            st.error(result["error"])
            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {result['error']}"})
        else:
            st.write(result["answer"])
            show_sources(result["sources"])
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "sources": result["sources"]})
