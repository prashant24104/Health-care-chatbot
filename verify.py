from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Connect
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("healthcare-chatbot")

# 2. Get Real Stats
stats = index.describe_index_stats()

print("\n📊 PINECONE INDEX STATS:")
print(stats)
print("\n--------------------------------")

# 3. Interpret Results
total_vectors = stats.get('total_vector_count', 0)
if total_vectors > 0:
    print(f"✅ SUCCESS: You have {total_vectors} vectors in your database!")
    print("   (The website is just lagging, ignore it.)")
else:
    print("❌ FAILURE: Your index is empty.")
    print("   This means your 'text_chunks' list was empty when you ran the upload.")