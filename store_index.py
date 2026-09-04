from dotenv import load_dotenv
import os
import json
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()


PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Missing required environment variable: PINECONE_API_KEY")


extracted_data=load_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks=text_split(filter_data)

embeddings = download_hugging_face_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)



index_name = "medical-chatbot"  # change if desired

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    # PineconeVectorStore.from_documents() below always inserts fresh vectors -
    # it never checks for/overwrites existing ones. Without clearing first,
    # every re-run of this script would duplicate the whole book in the index.
    print(f"Index '{index_name}' already exists - clearing it so this script stays safe to re-run.")
    pc.Index(index_name).delete(delete_all=True)

index = pc.Index(index_name)


docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

# Also cache the raw chunks (text + metadata) as JSON alongside the dense
# Pinecone index. app.py loads this file to build a keyword-based BM25
# retriever for hybrid search, without needing the original PDF at
# request-serving time (the PDF is excluded from the Docker image).
chunks_dump_path = os.path.join("data", "chunks.json")
with open(chunks_dump_path, "w", encoding="utf-8") as f:
    json.dump(
        [{"page_content": c.page_content, "metadata": c.metadata} for c in text_chunks],
        f,
    )
print(f"Saved {len(text_chunks)} chunks to {chunks_dump_path} for hybrid (BM25) search.")