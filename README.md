# 📄 PDF Q&A with LLaMA 3 via Groq

This project showcases a Retrieval-Augmented Generation (RAG) pipeline built with [Groq](https://groq.com/) and the **LLaMA 3 model** for answering questions from a PDF document. It leverages LangChain, HuggingFace embeddings, Chroma vector DB, and PDF loading utilities to enable document understanding and contextual Q&A.

---

## 🚀 Features

- 🧠 LLaMA 3.1 8B (instant) model via Groq API
- 📘 Loads PDF documents using `PyPDFLoader`
- 🧩 Chunks + embeds text with HuggingFace MiniLM model
- 🔍 Vector search with Chroma
- 🤖 Custom RAG pipeline for concise, context-based answers

---

## 🧰 Tech Stack

- Python 3.10+
- [LangChain](https://python.langchain.com/)
- [Groq](https://groq.com/)
- [LLaMA 3.1 8B (Instant)](https://llama.meta.com/)
- [HuggingFace Sentence Transformers](https://www.sbert.net/)
- Chroma vector database
- dotenv for API key management

---

## 📁 Project Structure

```bash
.
├── data/
│   └── Be_Good.pdf         # Input PDF file for processing
├── main.py                 # Main script for loading + answering
├── .env                    # Secrets file
