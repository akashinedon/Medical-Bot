import json
import os
from threading import Lock

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

from src.helper import download_hugging_face_embeddings
from src.prompt import contextualize_q_system_prompt, system_prompt

app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Missing required environment variable: PINECONE_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Missing required environment variable: GEMINI_API_KEY")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

# Pre-rerank recall (cast a wider net) vs. final chunks actually handed to the LLM
# (kept tight so the prompt stays small and answers stay grounded).
RETRIEVER_K = int(os.environ.get("RETRIEVER_K", "6"))
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", "3"))
CROSS_ENCODER_MODEL = os.environ.get("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Escape hatches: both default to on, but can be turned off (e.g. for a faster
# local dev cold-start, or if the BM25 corpus / cross-encoder can't be loaded).
ENABLE_HYBRID_SEARCH = os.environ.get("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
ENABLE_RERANKER = os.environ.get("ENABLE_RERANKER", "true").lower() == "true"

# Chunks cached by store_index.py, used to build the BM25 (keyword) retriever.
# We read from this JSON cache instead of the original PDF because the PDF is
# excluded from the Docker build context (see .dockerignore) - only this small
# cache ships to production.
CHUNKS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "chunks.json")

# How many prior turns (user+assistant messages) we forward to the model for
# follow-up questions. Bounded so a long conversation doesn't blow up the
# prompt size / token cost on every single request.
MAX_HISTORY_MESSAGES = 12


_rag_chain = None
_init_lock = Lock()
_last_chat_error = None


def _build_bm25_retriever():
    """Build a keyword-search (BM25) retriever from the cached chunks, if available."""
    if not ENABLE_HYBRID_SEARCH:
        return None

    if not os.path.exists(CHUNKS_PATH):
        app.logger.warning(
            "%s not found - hybrid search disabled, falling back to vector-only "
            "retrieval. Run `python store_index.py` once to generate it.",
            CHUNKS_PATH,
        )
        return None

    try:
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)

        docs = [
            Document(page_content=c.get("page_content", ""), metadata=c.get("metadata") or {})
            for c in raw_chunks
            if c.get("page_content")
        ]
        if not docs:
            return None

        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = RETRIEVER_K
        return bm25_retriever
    except Exception:
        app.logger.exception("Failed to build BM25 retriever; falling back to vector-only retrieval.")
        return None


def _build_chat_history(raw_history):
    """Convert the frontend's plain [{role, content}, ...] into LangChain messages."""
    messages = []
    if not isinstance(raw_history, list):
        return messages

    for item in raw_history[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))
    return messages


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
        vector_retriever = docsearch.as_retriever(
            search_type="similarity", search_kwargs={"k": RETRIEVER_K}
        )

        # Hybrid search: combine dense (semantic/vector) with sparse (keyword/BM25)
        # retrieval so exact medical terms aren't missed just because their
        # embedding isn't the closest match.
        bm25_retriever = _build_bm25_retriever()
        if bm25_retriever is not None:
            base_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever], weights=[0.6, 0.4]
            )
        else:
            base_retriever = vector_retriever

        # Re-rank the wider candidate set with a cross-encoder (much more accurate
        # than raw similarity scores, but too slow to run over the whole index -
        # so it only scores the top RETRIEVER_K candidates already shortlisted above).
        if ENABLE_RERANKER:
            try:
                cross_encoder = HuggingFaceCrossEncoder(model_name=CROSS_ENCODER_MODEL)
                reranker = CrossEncoderReranker(model=cross_encoder, top_n=RERANK_TOP_N)
                retriever = ContextualCompressionRetriever(
                    base_compressor=reranker, base_retriever=base_retriever
                )
            except Exception:
                app.logger.exception("Failed to load cross-encoder reranker; skipping reranking.")
                retriever = base_retriever
        else:
            retriever = base_retriever

        chatModel = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0,
        )

        # Step 1: rewrite a follow-up question ("what about its symptoms?") into a
        # standalone one, using chat history, BEFORE it reaches the retriever.
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            chatModel, retriever, contextualize_q_prompt
        )

        # Step 2: answer using the retrieved context + chat history.
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(chatModel, qa_prompt)
        _rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

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
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        msg = (payload.get("msg") or request.form.get("msg") or "").strip()
        raw_history = payload.get("history", [])
    else:
        msg = (request.args.get("msg") or "").strip()
        raw_history = []

    if not msg:
        return "Please enter a message.", 400

    chat_history = _build_chat_history(raw_history)

    def generate():
        global _last_chat_error
        sources = []
        seen_sources = set()
        answered = False
        try:
            rag_chain = get_rag_chain()
            for chunk in rag_chain.stream({"input": msg, "chat_history": chat_history}):
                # The first chunk(s) to carry data include "context" (the
                # retrieved documents) before any "answer" tokens start flowing.
                if "context" in chunk and not sources:
                    for doc in chunk["context"]:
                        source_name = os.path.basename(str(doc.metadata.get("source") or "Medical_book.pdf"))
                        page = doc.metadata.get("page")
                        key = (source_name, page)
                        if key in seen_sources:
                            continue
                        seen_sources.add(key)
                        snippet = doc.page_content.strip().replace("\n", " ")[:180]
                        sources.append(
                            {
                                "source": source_name,
                                # Pinecone round-trips numeric metadata as float (e.g. 528.0),
                                # while BM25 (reading straight from chunks.json) keeps it as
                                # int - accept both so citations don't silently lose the page
                                # just because a chunk happened to come from the vector side.
                                "page": (int(page) + 1) if isinstance(page, (int, float)) else None,
                                "snippet": snippet + ("…" if len(doc.page_content.strip()) > 180 else ""),
                            }
                        )

                answer_piece = chunk.get("answer")
                if answer_piece:
                    answered = True
                    yield answer_piece

            if not answered:
                yield "I couldn't find a clear answer. Please rephrase your question."

            _last_chat_error = None
            # Sentinel the frontend splits on to pull out structured citation data
            # from the plain-text stream, without needing a second round trip.
            yield "\n<<<SOURCES>>>" + json.dumps(sources)
        except Exception as exc:
            _last_chat_error = f"{type(exc).__name__}: {str(exc)[:500]}"
            app.logger.exception("Chat request failed")
            yield (
                "I'm having trouble reaching the medical knowledge service right now. "
                "Please try again in a moment."
            )

    return Response(stream_with_context(generate()), mimetype="text/plain")


if __name__ == '__main__':
    port = int(os.environ.get("PORT", "8080"))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
