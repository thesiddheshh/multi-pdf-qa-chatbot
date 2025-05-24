# PDF RAG Chatbot

A fully functional, locally hosted chatbot that allows users to upload one or more PDF documents and engage in natural language conversations with their content. Built using Streamlit, Hugging Face Transformers, FAISS, SentenceTransformers, and OpenRouter’s Mistral-7B-Instruct model, this tool demonstrates how Retrieval-Augmented Generation (RAG) can be applied to interactive document question answering.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Setup and Installation](#setup-and-installation)
- [How to Use](#how-to-use)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [API Configuration](#api-configuration)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [License](#license)
- [Contact](#contact)

---

## Project Overview

This application enables users to upload single or multiple PDF files and ask natural language questions about the content. The app performs document parsing, semantic chunking, embedding, indexing via FAISS, and retrieval-augmented response generation using an external LLM API.

Use cases include:

- Reading and understanding academic papers
- Extracting information from policy documents or legal contracts
- Internal document search for organizations
- Building educational tools for complex multi-source comprehension

---

## Key Features

- Upload one or multiple PDF files directly from the sidebar
- Select which document to query — individual PDFs or all at once
- Vector-based semantic search using FAISS
- Accurate chunking strategy for long-document retrieval
- Contextual answer generation using Mistral-7B via OpenRouter
- Chat history display with user and bot dialogue
- Option to download the entire Q&A session as a `.txt` file
- Minimal UI with a sidebar file manager and central chat window

---

## System Architecture

```text
+-------------------------+
|     User Interface      |
|     (Streamlit)         |
+-----------+-------------+
            |
            v
+-----------+-------------+
|   PDF Ingestion Layer   | <- Reads PDFs using PyMuPDF
+-----------+-------------+
            |
            v
+-----------+-------------+
| Text Chunking & Embedding |
| (SentenceTransformers)   |
+-----------+-------------+
            |
            v
+-----------+-------------+
|     FAISS Indexing       |
|   (Semantic Retrieval)   |
+-----------+-------------+
            |
            v
+-----------+-------------+
| Context Construction     |
| Prompt Engineering       |
+-----------+-------------+
            |
            v
+-----------+-------------+
|  OpenRouter LLM API      |
| (Mistral-7B-Instruct)    |
+-----------+-------------+
            |
            v
+-----------+-------------+
|        Chat Output       |
|   Display + Export File  |
+-------------------------+
```

---

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

### 2. Set Up a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Your OpenRouter API Key

In `app.py`, replace the placeholder with your OpenRouter API key:

```python
api_key = "sk-or-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

You can register and obtain a key from https://openrouter.ai/

---

## How to Use

1. Launch the app:

```bash
streamlit run app.py
```

2. In the sidebar:
   - Use **"Upload PDFs"** to upload one or more documents.
   - Use the **dropdown** to select which document(s) to query (including “All”).
3. Type your question in the chat input at the bottom center.
4. Press Enter to get a context-aware answer.
5. Use the **"Download Chat"** button to export the entire conversation.

---

## Deployment

### Deploy on Streamlit Cloud

1. Push this project to a public GitHub repository.
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign in.
3. Create a new app:
   - Repository: `yourusername/pdf-rag-chatbot`
   - Branch: `main`
   - File path: `app.py`
4. Add your `OPENROUTER_API_KEY` as a **secret environment variable**.
5. Click "Deploy."

---

## Project Structure

```text
pdf-rag-chatbot/
├── app.py                   # Main Streamlit application logic
├── requirements.txt         # Project dependencies
├── .gitignore               # Git ignore configuration
├── README.md                # Project documentation
├── .streamlit/
│   └── config.toml          # UI layout and settings
├── assets/                  # (Optional) Image or icon resources
│   ├── sidebar.png
│   └── chat.png
```

---

## Dependencies

- `streamlit` — for UI and application state
- `PyMuPDF` — PDF parsing
- `faiss-cpu` — vector search and nearest neighbor lookup
- `sentence-transformers` — embedding with MiniLM
- `transformers` — tokenizer utility
- `requests` — sending API requests to OpenRouter

Install with:

```bash
pip install streamlit PyMuPDF faiss-cpu numpy sentence-transformers transformers requests
```

---

## API Configuration

This app uses **OpenRouter** to access hosted language models like Mistral-7B.

- Create an account at: https://openrouter.ai/
- Obtain your API key
- Insert your key into the `api_key` field in `app.py`

Example:

```python
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
```

All requests are sent to: `https://openrouter.ai/api/v1/chat/completions`

---

## Limitations and Future Improvements

**Limitations:**
- No local model support (external API dependency)
- Retrieval limited to top-k chunks (semantic but not exhaustive)
- Long PDF parsing may slow down embedding step

**Future Enhancements:**
- Add OCR support for scanned PDFs
- Integrate file summarization as a separate task
- Improve chunking with adaptive splitting (based on semantics or sections)
- Switch to locally hosted models using LangChain or Ollama
- Persistent chat history using a database

---

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute it with attribution.

---

## Contact

**Author**: Siddhant  
GitHub: [@yourusername](https://github.com/yourusername)  
Email: your.email@example.com  

For suggestions, improvements, or issues, feel free to open a [GitHub Issue](https://github.com/yourusername/pdf-rag-chatbot/issues).
