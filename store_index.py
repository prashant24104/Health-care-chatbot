from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, text_split, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# 1. Load Environment Variables
print("🔄 Loading environment...")
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not PINECONE_API_KEY or not HUGGINGFACE_API_KEY:
    raise RuntimeError("❌ Missing API keys. Check your .env file!")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY

# 2. Load PDFs
print("📂 Loading PDF files from 'data' folder...")
docs = load_pdf_files("data")
print(f"   ✅ Loaded {len(docs)} pages.")

# 3. Split text
print("✂️  Splitting text...")
chunks = text_split(docs)
print(f"   ✅ Split documents into {len(chunks)} chunks.")

# 4. Download embeddings
print("⬇️  Downloading Embeddings Model...")
embeddings = download_embeddings()
print("   ✅ Embeddings model ready.")

# 5. Connect to Pinecone
print("🌲 Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "healthcare-chatbot"

if index_name not in pc.list_indexes().names():
    print(f"   🔨 Creating new index: '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384,  # correct for all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("   ✅ Index created.")
else:
    print(f"   ✅ Index '{index_name}' already exists.")

# 6. Upload vectors
print("🚀 Uploading vectors to Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name
)

print(f"🎉 Success! Uploaded {len(chunks)} vectors to Pinecone.")