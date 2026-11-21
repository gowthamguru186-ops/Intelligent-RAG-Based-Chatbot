"""Core RAG pipeline: load PDFs -> clean -> chunk -> embed -> FAISS -> retrieve -> Gemini answer.

Used by build_index.py (creates the FAISS index) and app.py (Streamlit chatbot).
"""
import os
import re
import time
import unicodedata
import warnings
from collections import Counter
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------------------------------------------------------- settings
PROJECT_DIR = Path(os.getenv("RAG_PROJECT_DIR", Path(__file__).resolve().parent))
DATA_DIR = PROJECT_DIR / "data" / "documents"
VECTORSTORE_DIR = PROJECT_DIR / "vectorstore"
ENV_PATH = PROJECT_DIR / ".env"

load_dotenv(ENV_PATH, override=True)

EMBEDDING_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"  # accepts 512 tokens
CHUNK_SIZE = 512       # tokens
CHUNK_OVERLAP = 50     # tokens
TOP_K = 4
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

NOT_FOUND_MESSAGE = "I couldn't find this information in the provided document."
MAX_QUESTION_CHARS = 1000

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an AI Document Assistant.

Answer the question using ONLY the context below. Do not use outside knowledge.
If the answer is not available in the context, reply exactly:
"I couldn't find this information in the provided document."

Context:
{context}

Question:
{question}

Answer:
""")


# ----------------------------------------------------------------------------- cleaning
def find_repeated_lines(pages, min_share=0.5, min_len=5, max_len=120):
    """Short lines that repeat on many pages (headers / footers)."""
    counts = Counter()
    for page in pages:
        counts.update({ln.strip() for ln in page.page_content.splitlines() if ln.strip()})
    limit = max(2, int(len(pages) * min_share))
    return {ln for ln, n in counts.items() if n >= limit and min_len <= len(ln) <= max_len}


def clean_text(text, repeated_lines=frozenset()):
    text = unicodedata.normalize("NFKC", text)                     # ligatures: ﬁ -> fi
    text = re.sub(r"(?<=[A-Za-z])-\n(?=[a-z])", "", text)          # join hyphenated line breaks
    lines = [
        ln for ln in text.splitlines()
        if ln.strip() not in repeated_lines
        and not re.fullmatch(r"\s*\d{1,3}\s*", ln)                  # page numbers
    ]
    text = " ".join(lines)
    text = re.sub(r"arXiv:\d{4}\.\d{4,5}v\d+\s+\[[^\]]+\]\s+\d{1,2}\s+\w{3}\s+\d{4}", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ----------------------------------------------------------------------------- ingestion
def load_pdf(pdf_path):
    """Load one PDF as cleaned page Documents with file name + page number metadata."""
    pages = PyPDFLoader(str(pdf_path)).load()
    repeated = find_repeated_lines(pages)
    cleaned = []
    for i, page in enumerate(pages):
        number = page.metadata.get("page", i)
        text = clean_text(page.page_content, repeated)
        if text:  # skips empty pages (scanned pages need OCR, see README)
            cleaned.append(Document(
                page_content=text,
                metadata={"source": str(pdf_path), "source_name": Path(pdf_path).name,
                          "page": number, "page_number": number + 1},
            ))
    return cleaned


def split_documents(pages):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


@lru_cache(maxsize=1)
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(pdf_paths=None):
    """Read every PDF in data/documents (or the given paths), build and save the FAISS index."""
    pdf_paths = sorted(pdf_paths or DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {DATA_DIR}. Add at least one PDF first.")

    pages = []
    for path in pdf_paths:
        pages.extend(load_pdf(path))
    if not pages:
        raise ValueError("No text could be extracted from the PDFs (scanned PDFs need OCR).")

    chunks = split_documents(pages)
    model = get_embedding_model()
    texts = [c.page_content for c in chunks]
    vectors = model.embed_documents(texts)
    store = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, vectors)),
        embedding=model,
        metadatas=[c.metadata for c in chunks],
    )
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(VECTORSTORE_DIR))
    return {"documents": len(pdf_paths), "pages": len(pages), "chunks": len(chunks)}


def load_vectorstore():
    if not (VECTORSTORE_DIR / "index.faiss").exists():
        raise FileNotFoundError("Vector store not found. Run `python build_index.py` first.")
    # Pickle is used for metadata; safe because we only load an index we created ourselves.
    return FAISS.load_local(str(VECTORSTORE_DIR), get_embedding_model(),
                            allow_dangerous_deserialization=True)


# ----------------------------------------------------------------------------- answering
@lru_cache(maxsize=1)
def get_llm():
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError(
            f"GOOGLE_API_KEY is missing. Add GOOGLE_API_KEY=<your key> to {ENV_PATH}")
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, max_retries=2)


def format_docs(docs):
    return "\n\n".join(
        f"[Source: {d.metadata.get('source_name', 'unknown')}, Page {d.metadata.get('page_number', '?')}]\n"
        f"{d.page_content}" for d in docs)


def ask_rag(question, vectorstore, max_attempts=3):
    """Answer a question. Never raises: problems come back in result['error']."""
    result = {"question": question, "answer": "", "sources": [], "contexts": [], "error": None}

    if not isinstance(question, str) or not question.strip():
        result["error"] = "Please enter a question."
        return result
    question = question.strip()
    if len(question) > MAX_QUESTION_CHARS:
        result["error"] = f"The question is too long (maximum {MAX_QUESTION_CHARS} characters)."
        return result
    result["question"] = question

    try:
        docs = vectorstore.as_retriever(search_kwargs={"k": TOP_K}).invoke(question)
    except Exception as error:
        result["error"] = f"Could not search the vector store: {error}"
        return result
    if not docs:
        result["answer"] = NOT_FOUND_MESSAGE
        return result
    result["contexts"] = [d.page_content for d in docs]

    try:
        chain = RAG_PROMPT | get_llm() | StrOutputParser()
    except Exception as error:
        result["error"] = str(error)
        return result

    for attempt in range(1, max_attempts + 1):
        try:
            result["answer"] = chain.invoke(
                {"context": format_docs(docs), "question": question}).strip()
            break
        except Exception as error:
            message = str(error)
            temporary = any(c in message for c in ("429", "500", "502", "503", "504")) or any(
                w in message.upper() for w in ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "RATE LIMIT", "TIMEOUT"))
            if temporary and attempt < max_attempts:
                time.sleep(5 * attempt)
                continue
            result["error"] = f"LLM request failed: {message[:300]}"
            return result

    if NOT_FOUND_MESSAGE not in result["answer"]:
        seen = []
        for d in docs:
            src = (d.metadata.get("source_name", "unknown"), d.metadata.get("page_number", "?"))
            if src not in seen:
                seen.append(src)
        result["sources"] = seen
    return result
