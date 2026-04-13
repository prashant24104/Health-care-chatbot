from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document

# --- 1. LOAD PDFs ---
def load_pdf_files(data):
    loader = DirectoryLoader(
        data, 
        glob="**/*.pdf", 
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

# --- 2. FILTER (Optional) ---
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, returns a new list of Document objects 
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src} 
            )
        )
    return minimal_docs

# --- 3. SPLIT TEXT ---
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        separators=["\n\n", "\n", " ", ""]
    )
    text_chunks = text_splitter.split_documents(minimal_docs)
    return text_chunks

def normalize_docs(text_chunks):
    """Convert Document objects or dicts into plain strings."""
    texts = []
    for d in text_chunks:
        if hasattr(d, "page_content"):   # Document object
            texts.append(d.page_content)
        elif isinstance(d, dict) and "page_content" in d:  # dict
            texts.append(d["page_content"])
        elif isinstance(d, str):        # already a string
            texts.append(d)
    return texts

# --- 4. DOWNLOAD EMBEDDINGS ---
def download_embeddings():
    """ Downloads the HuggingFace embeddings model."""
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
    print("   ✅ Vectors uploaded.")