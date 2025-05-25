import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
import time
import os
import requests
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from io import BytesIO
from datetime import datetime

# ------------------------------ #
#         PAGE SETUP            #
# ------------------------------ #
st.set_page_config(
    page_title="📚 PDF RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ------------------------------ #
#         CUSTOM CSS            #
# ------------------------------ #
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f4f6f8;
        }
        .main {
            padding: 2rem;
        }
        .stChatMessage {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05);
        }
        .stChatMessage.user {
            background-color: #d6eaff !important;
            color: black !important;
            border-left: 4px solid #1c77c3;
        }
        .stChatMessage.assistant {
            background-color: #d8f5e6 !important;
            color: black !important;
            border-left: 4px solid #28a745;
        }
        .sidebar-title {
            font-size: 34px;
            font-weight: 900;
            color: #1c77c3;
            text-transform: uppercase;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .sidebar-title img {
            width: 40px;
            height: 40px;
        }
        .block-container {
            padding-top: 1rem;
        }
        .css-18e3th9 {
            padding-top: 0rem !important;
        }
        .pdf-block {
            background-color: #222;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            border: 1px solid #333;
            color: white;
        }
        .stButton>button {
            background-color: #1c77c3;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.6em 1em;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #145a92;
        }
        .stDownloadButton>button {
            background-color: #28a745;
            color: white;
            border-radius: 6px;
        }
        .stSelectbox>div {
            border-radius: 6px;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------ #
#          LOAD MODELS          #
# ------------------------------ #
@st.cache_resource(show_spinner=False)
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return embedder

embedder = load_models()

# ------------------------------ #
#           HELPERS             #
# ------------------------------ #
def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def count_pdf_pages(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return doc.page_count

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve_top_k_chunks(question, chunks, index, k=3):
    q_emb = embedder.encode([question]).astype("float32")
    distances, indices = index.search(q_emb, k)
    top_chunks = [chunks[i] for i in indices[0]]
    return top_chunks

def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi"]:
        if abs(num) < 1024.0:
            return f"{num:.2f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.2f} Ti{suffix}"

def download_chat(chat_history):
    txt = ""
    for role, msg in chat_history:
        txt += f"{role.upper()}:\n{msg}\n\n"
    return txt

def query_openrouter_llm(prompt, context):
    api_key = "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"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Always provide detailed, complete answers with context and citation."},
            {"role": "user", "content": f"Answer this question using the following PDF context:\n\n{context}\n\nQuestion: {prompt}"},
        ],
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ------------------------------ #
#        SESSION STATE          #
# ------------------------------ #
if "pdf_docs" not in st.session_state:
    st.session_state.pdf_docs = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------------------ #
#             SIDEBAR           #
# ------------------------------ #
with st.sidebar:
    st.markdown(
        """
        <div class='sidebar-title'>
            <img src='https://cdn-icons-png.flaticon.com/512/4712/4712109.png' alt='icon' />
            RAG Chatbot
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.header("📂 Manage PDFs")

    st.subheader("📤 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.pdf_docs:
                file_bytes = file.read()
                file_size = sizeof_fmt(len(file_bytes))
                file_pages = count_pdf_pages(file_bytes)
                try:
                    text = extract_text_from_pdf(file_bytes)
                except:
                    st.warning(f"Failed to read {file.name}. Skipping.")
                    continue
                chunks = chunk_text(text)
                embeddings = embedder.encode(chunks).astype("float32")
                index = create_faiss_index(embeddings)

                st.session_state.pdf_docs[file.name] = {
                    "text": text,
                    "chunks": chunks,
                    "embeddings": embeddings,
                    "index": index,
                    "pages": file_pages,
                    "size": file_size,
                }

    for fname, meta in st.session_state.pdf_docs.items():
        st.markdown(
            f"<div class='pdf-block'><strong>{fname}</strong><br />📄 {meta['pages']} pages | 💾 {meta['size']}</div>",
            unsafe_allow_html=True,
        )

    pdf_options = list(st.session_state.pdf_docs.keys())
    multiselect_options = ["All"] + pdf_options
    selected_raw = st.multiselect("📄 Select one or more PDFs to query:", multiselect_options)
    selected_pdfs = pdf_options if "All" in selected_raw else selected_raw

# ------------------------------ #
#         MAIN CHAT AREA        #
# ------------------------------ #
st.title("📚 Ask Questions About Your PDFs")

for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

if prompt := st.chat_input("Ask a question..."):
    if not selected_pdfs:
        st.warning("⚠️ Please upload and select at least one PDF first.")
    else:
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append(("user", prompt))

        combined_chunks = []
        for pdf_name in selected_pdfs:
            meta = st.session_state.pdf_docs[pdf_name]
            combined_chunks.extend(meta["chunks"])

        combined_embeddings = embedder.encode(combined_chunks).astype("float32")
        combined_index = create_faiss_index(combined_embeddings)

        top_chunks = retrieve_top_k_chunks(prompt, combined_chunks, combined_index, k=3)

        context = ""
        for i, chunk in enumerate(top_chunks):
            context += f"[Source chunk {i+1}]\n{chunk}\n\n"

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = query_openrouter_llm(prompt, context)
                st.markdown(answer)
                st.session_state.chat_history.append(("assistant", answer))

if st.session_state.chat_history:
    txt = download_chat(st.session_state.chat_history)
    st.download_button(
        label="💾 Download Chat History",
        data=txt,
        file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
    )
