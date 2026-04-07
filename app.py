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
_last_chat_error = None


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


@app.route("/debug/last_error")
def last_error():
    return jsonify({"last_chat_error": _last_chat_error}), 200



@app.route("/get", methods=["GET", "POST"])
def chat():
    global _last_chat_error
    msg = request.form.get("msg") or request.args.get("msg")
    if not msg:
        return "Please enter a message.", 400

    try:
        print(msg)
        rag_chain = get_rag_chain()
        response = rag_chain.invoke({"input": msg})
        answer = str(response.get("answer", "")).strip()
        _last_chat_error = None
        return answer or "I couldn't find a clear answer. Please rephrase your question."
    except Exception as exc:
        _last_chat_error = f"{type(exc).__name__}: {str(exc)[:500]}"
        app.logger.exception("Chat request failed")
        return (
            "I'm having trouble reaching the medical knowledge service right now. "
            "Please try again in a moment."
        ), 200



if __name__ == '__main__':
    port = int(os.environ.get("PORT", "8080"))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
