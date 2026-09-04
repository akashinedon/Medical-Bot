# Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS

# How to run?

### STEPS:

Clone the repository

```bash
git clonehttps://github.com/entbappy/Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS.git
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n medibot python=3.10 -y
```

```bash
conda activate medibot
```

### STEP 02- install the requirements

```bash
pip install -r requirements.txt
```

### Create a `.env` file in the root directory and add your Pinecone & Gemini credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GEMINI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,

```bash
open up localhost:
```

### Upgrading an existing setup (multi-turn memory, hybrid search + reranking, citations, streaming)

If you already had this repo running before, do this once after pulling the latest code:

```bash
pip install -r requirements.txt   # picks up the new rank_bm25 dependency
python store_index.py             # re-run LOCALLY (not via `docker run ... python store_index.py`,
                                   # the PDF is intentionally excluded from the Docker image -
                                   # see .dockerignore). This refreshes the Pinecone metadata with
                                   # page numbers and writes data/chunks.json, which app.py needs
                                   # at runtime to build the BM25 half of hybrid search.
```

New optional env vars (all have sane defaults, none are required):

| Variable              | Default                                | Purpose                                             |
|-----------------------|-----------------------------------------|------------------------------------------------------|
| `RETRIEVER_K`          | `6`                                      | Chunks fetched per retriever before reranking        |
| `RERANK_TOP_N`         | `3`                                      | Chunks kept after reranking, sent to the LLM         |
| `CROSS_ENCODER_MODEL`  | `cross-encoder/ms-marco-MiniLM-L-6-v2`   | Model used to rerank retrieved chunks                |
| `ENABLE_HYBRID_SEARCH` | `true`                                   | Set `false` to use vector-only retrieval             |
| `ENABLE_RERANKER`      | `true`                                   | Set `false` to skip the cross-encoder rerank step    |

Note: `/get` now expects a JSON body (`{"msg": "...", "history": [...]}`) and streams back
plain text (a trailing `\n<<<SOURCES>>>` + JSON carries citations) instead of returning a
single plain-text response - the bundled `chat.html` frontend already speaks this format.

### Techstack Used:

- Python
- LangChain
- Flask
- Gemini
- Pinecone

# Deployment on Vercel

This app deploys as a single [Vercel Function](https://vercel.com/docs/functions) running the
Flask app directly (Vercel auto-detects the `app` instance in `app.py`, zero extra
config needed for that part). The previous AWS EC2 + self-hosted-runner CI/CD pipeline has
been retired - `.github/workflows/cicd.yaml` is gone, and Vercel's own Git integration
handles deploys instead (push to `main`/`master` -> auto preview/production deploy, no
GitHub Actions required).

## One-time project setup

```bash
npm i -g vercel   # if you don't already have the CLI
vercel login
vercel link       # links this folder to a Vercel project
```

Add the required secrets (same two keys as local `.env`, plus any optional tuning vars
from the table above):

```bash
vercel env add PINECONE_API_KEY
vercel env add GEMINI_API_KEY
```

## Deploying

```bash
vercel          # preview deployment - get a URL, sanity check it first
vercel --prod   # promote to production once the preview looks good
```

Or just `git push` once the project is linked and Git integration is enabled in the
Vercel dashboard - every push gets a preview deployment, and pushes to the production
branch deploy to production automatically.

## Known constraints specific to this stack

- **Bundle size**: `torch` + `sentence-transformers` + `langchain*` are heavy. `requirements.txt`
  pins `torch==<version>+cpu` (via `--extra-index-url https://download.pytorch.org/whl/cpu`) to
  avoid accidentally pulling CUDA packages, which would blow past Vercel's function size limits.
  The standard limit is 500MB; if the deployed bundle exceeds that, enable **Large Functions**
  (Beta, up to 5GB) on the Vercel project.
- **`data/chunks.json`** (needed for BM25 hybrid search) is *not* excluded by `.vercelignore` and
  must be committed to git so it ships with the function. `data/*.pdf` *is* excluded - it's only
  needed offline by `store_index.py`, never at request time.
- **Cold starts**: the first request after a deploy builds the whole RAG chain (downloads the
  MiniLM embedding + cross-encoder models, connects to Pinecone) - `vercel.json` sets
  `maxDuration: 120` on `app.py` to give it room. Fluid Compute then reuses that warm instance
  for subsequent requests.
- **Streaming**: `/get` streams its response via a Flask generator. This is standard WSGI
  streaming and is expected to work, but verify it on your first real preview deploy - stream
  end-to-end with `curl -N` before trusting it in the UI.
