import flask
from flask import Flask, request, jsonify, render_template
from src.helper import download_embeddings
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from src.prompt import system_prompt

app = Flask(__name__)

# 1. Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not PINECONE_API_KEY or not HUGGINGFACE_API_KEY:
    raise RuntimeError("❌ Missing API keys. Check your .env file!")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY

# 2. Load embeddings
embeddings = download_embeddings()
index_name = "healthcare-chatbot"

# 3. Connect to Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# # 4. Define system prompt
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_promp),
#         ("human", "{input}")
#     ]
# )

# 5. Initialize the LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="conversational",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)
model = ChatHuggingFace(llm=llm)

# 6. Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Context:\n{context}\n\nQuestion:\n{input}")
])

question_answer_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 7. Format retriever output
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# 8. Build RAG chain
# (The retrieval chain is already configured above.)

# 9. Routes
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET","POST"])
def chat():

    data = request.get_json(silent=True)
    if data and 'message' in data:
        msg = data['message']
    else:
        msg = request.form.get("msg") or request.args.get("msg")
    
    if not msg:
        return jsonify({"error": "No message provided"}), 400

    # Handle greetings
    greetings = ['hi', 'hello', 'hii', 'hey', 'good morning', 'good afternoon', 'good evening']
    if msg.lower().strip() in greetings:
        return jsonify({"answer": "Hello! I'm your medical assistant. How can I help you with health-related questions today?"})

    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response :", response["answer"])

    return jsonify({"answer": response["answer"]})

if __name__ == "__main__":
    app.run( debug=True)