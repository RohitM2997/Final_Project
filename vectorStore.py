# vectoreStore.py

import tempfile
import fitz  # PyMuPDF
import hashlib          # Streamlit's built-in caching for reference chunking functions.
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from newspaper import Article                                       # to extract text well from url and works faster. These libraries focus only on extracting the main article/content — not scripts, sidebars, etc.



# The _hash_bytes function generates an MD5 hash of byte data. This is used in conjunction with st.cache_data to cache the results of expensive operations (like PDF parsing or chunking) based on the content of the file. If the file content hasn't changed (same hash), Streamlit will return the cached result instead of re-running the function, improving performance.
def _hash_bytes(data:bytes) -> str:
    """Generate a hash from byte data for caching consistency."""
    return hashlib.md5(data).hexdigest()


# function that dynamically selects chunk_size and chunk_overlap
def get_dynamic_chunk_params(text: str):
    word_count = len(text.split())
    if word_count < 500:
        return 300, 50
    elif word_count < 2000:
        return 700, 150
    elif word_count < 5000:
        return 1000, 200
    else:
        return 1500, 300


# A helper function that takes raw text and splits it into smaller, overlapping chunks.
def _split_into_chunks(text: str):
    """Split text into overlapping chunks using LangChain."""
    chunk_size, chunk_overlap = get_dynamic_chunk_params(text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


# SentenceTransformersTokenTextSplitter Uses nltk or similar tools to split text by sentences. Groups sentences into token-length chunks using a tokenizer
# def _split_into_chunks(text: str):
#     """Split text into overlapping chunks using LangChain."""
#     chunk_size, chunk_overlap = get_dynamic_chunk_params(text)
#     splitter = SentenceTransformersTokenTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap
#     )
#     return splitter.split_text(text)


#It is the cached function. It writes the uploaded PDF bytes to a temporary file, opens it with PyMuPDF (fitz), extracts text page by page, and then calls _split_into_chunks.
@st.cache_data(show_spinner=False)
def extract_chunks_from_pdf_cached(file_hash: str, file_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    doc = fitz.open(tmp_path)
    text = "\n".join([page.get_text() for page in doc])
    return _split_into_chunks(text)


#It acts as a wrapper. It calculates the file hash and then calls the cached version. st.info is used to give user feedback about caching.
def extract_chunks_from_pdf( file_bytes:bytes):       
    st.info("📦 Checking for cached PDF chunks...")
    file_hash = _hash_bytes(file_bytes)
    return extract_chunks_from_pdf_cached(file_hash, file_bytes)


@st.cache_data(show_spinner=False)
def extract_chunks_from_txt_cached(file_hash: str, file_bytes: bytes):
    text = file_bytes.decode("utf-8")
    return _split_into_chunks(text)


def extract_chunks_from_txt(file_bytes:bytes):
    st.info("📦 Checking for cached TXT chunks...")
    file_hash = _hash_bytes(file_bytes)
    return extract_chunks_from_txt_cached(file_hash, file_bytes)


# Directly takes a string (pasted text) and splits it into chunks
@st.cache_data(show_spinner=False)
def extract_chunks_from_text(text: str):
    st.info("📦 Checking for cached pasted text chunks...")
    return _split_into_chunks(text)


# newspaper is used to download and parse the content from URL. If successful, it splits the extracted text into chunks
@st.cache_data(show_spinner=False)
def extract_chunks_from_url(url:str):
    st.info("🌐 Downloading and extracting content from URL...")
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        if not text.strip():
            raise ValueError("No article content found.")
    except Exception as e:
        st.warning(f"⚠️ Failed to fetch or parse the URL: {e}")
        return []
    return _split_into_chunks(text)


# Creating vectore store and retriever setup
def get_retriever(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={"device": "cpu"}      # ensures it runs on CPU
    )
    # Create a FAISS in-memory vector store from the text chunks using the generated embeddings
    # default indexing for faiss is IndexFlatL2, euclidean distance.
    db = FAISS.from_texts(chunks, embedding=embeddings)

    return db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    # search_type="similarity": Specifies that the retriever should find chunks most similar to the query.
    #search_kwargs={"k": 5}: Instructs the retriever to return the top 5 most similar chunks.