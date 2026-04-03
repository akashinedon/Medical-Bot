from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from threading import Lock


app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Missing required environment variable: PINECONE_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Missing required environment variable: GEMINI_API_KEY")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")


_rag_chain = None
_init_lock = Lock()


def get_rag_chain():
    """Initialize RAG components on first use to keep app startup fast."""
    global _rag_chain
    if _rag_chain is not None:
        return _rag_chain

    with _init_lock:
        if _rag_chain is not None:
            return _rag_chain

        embeddings = download_hugging_face_embeddings()
        index_name = "medical-chatbot"
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings,
        )
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        chatModel = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
        _rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return _rag_chain



@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    rag_chain = get_rag_chain()
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])



if __name__ == '__main__':
    port = int(os.environ.get("PORT", "8080"))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
