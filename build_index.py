"""Build (or rebuild) the FAISS index from every PDF in data/documents.

Usage:  python build_index.py
"""
from rag_pipeline import DATA_DIR, VECTORSTORE_DIR, build_vectorstore

if __name__ == "__main__":
    print("Reading PDFs from:", DATA_DIR)
    try:
        stats = build_vectorstore()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}")
    print(f"Done: {stats['documents']} document(s), {stats['pages']} pages, {stats['chunks']} chunks")
    print("Index saved to:", VECTORSTORE_DIR)
