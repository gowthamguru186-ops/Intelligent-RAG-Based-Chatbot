# Intelligent RAG-Based Chatbot using LangChain & LLMs

An AI-powered **Document Assistant** that uses Retrieval-Augmented Generation (RAG) to answer user questions from PDF documents. It retrieves relevant information, generates context-based answers using Google Gemini, and displays source references.

## Project Overview

The application allows users to upload PDF documents and ask questions in natural language. Instead of relying only on the LLM's pre-trained knowledge, it retrieves relevant document chunks and generates answers grounded in the provided content.

## Key Features

- PDF document loading and text extraction
- Text cleaning and document chunking
- Semantic search using Sentence Transformers
- FAISS vector database for similarity search
- Google Gemini for answer generation
- Source document and page references
- Conversational question-answering
- Fallback response for questions not covered by the documents
- RAGAS-based evaluation

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| LLM Framework | LangChain |
| Embedding Model | Sentence Transformers |
| Vector Database | FAISS |
| LLM | Google Gemini |
| User Interface | Streamlit |
| Evaluation | RAGAS |
| Notebook | Jupyter Notebook |

## Architecture

1. **Document Loading:** Load PDF documents and extract text.
2. **Text Processing:** Clean and split the extracted text into chunks.
3. **Embedding Generation:** Convert text chunks into vector embeddings.
4. **Vector Storage:** Store embeddings in a FAISS vector database.
5. **Retrieval:** Retrieve the most relevant chunks based on the user query.
6. **Answer Generation:** Use Google Gemini to generate answers from retrieved context.
7. **Source References:** Display relevant document sources and page numbers.

## Project Structure

```text
RAG_CHATBOT/
│
├── app.py
├── rag_pipeline.py
├── build_index.py
├── requirements.txt
├──
