1. Clone / Open Project

Navigate to the project folder:

cd knowledge-search
2. Backend Setup

Go to backend folder:

cd backend

Create virtual environment (Python 3.12):

py -3.12 -m venv .venv

Activate virtual environment:

.\.venv\Scripts\Activate.ps1

Upgrade pip:

python -m pip install --upgrade pip

Install dependencies:

pip install -r requirements.txt

Install FAISS:

pip install faiss-cpu
3. Generate Document Corpus

Go back to project root:

cd ..

Generate dataset for search:

python scripts\generate_corpus.py
4. Start Backend Server

Go back to backend:

cd backend

Run API server:

uvicorn app.main:app --reload

Backend will run at:

http://127.0.0.1:8000

API docs:

http://127.0.0.1:8000/docs

![alt text](image.png)

![alt text](image-1.png)


┌─────────────────────────────────────────────────────────────────┐
│                    YOUR PROJECT FLOW                             │
│                                                                  │
│  STEP 1: DATA                                                    │
│  ─────────────────────────────────────────────────               │
│  300+ .txt files (science, history, tech articles)               │
│         │                                                        │
│         ▼                                                        │
│  python -m app.ingest → docs.jsonl                               │
│  Each doc: { doc_id, title, text, source, created_at }          │
│                                                                  │
│  STEP 2: INDEXING                                                │
│  ─────────────────────────────────────────────────               │
│  docs.jsonl                                                      │
│       │                                                          │
│       ├──► BM25Index.build()                                     │
│       │    - tokenize: "Python programming" → ["python","prog…"] │
│       │    - rank-bm25 scores term freq × IDF                    │
│       │    - saved to data/index/bm25/bm25.pkl                   │
│       │                                                          │
│       └──► VectorIndex.build()                                   │
│            - sentence-transformers encodes each doc              │
│            - "Python guide" → [0.12, -0.45, 0.87, ...] 384 dims │
│            - FAISS stores all vectors                            │
│            - saved to data/index/vector/faiss.index              │
│                                                                  │
│  STEP 3: SEARCH REQUEST                                          │
│  ─────────────────────────────────────────────────               │
│  User types: "machine learning neural networks"                  │
│                                                                  │
│       ┌─────────────────────────────────────┐                   │
│       │         POST /api/search             │                   │
│       └─────────────────────────────────────┘                   │
│              │                   │                               │
│              ▼                   ▼                               │
│         BM25.query()       VectorIndex.query()                   │
│         (exact words)      (meaning/concept)                     │
│              │                   │                               │
│              ▼                   ▼                               │
│       [(doc3, 2.5),        [(doc5, 0.92),                        │
│        (doc1, 1.8), ...]    (doc3, 0.78), ...]                   │
│              │                   │                               │
│              └────────┬──────────┘                               │
│                       ▼                                          │
│              normalize both to [0,1]                             │
│              hybrid = 0.5×bm25_norm + 0.5×vec_norm              │
│                       │                                          │
│                       ▼                                          │
│              ranked results with score breakdown                 │
│              logged to SQLite                                    │
│                                                                  │
│  STEP 4: DASHBOARD (KPI page)                                    │
│  ─────────────────────────────────────────────────               │
│  Reads SQLite → shows charts:                                    │
│  • p50/p95 latency  • request volume  • top queries              │
│                                                                  │
│  STEP 5: EVALUATION                                              │
│  ─────────────────────────────────────────────────               │
│  python -m app.eval → nDCG@10, Recall@10, MRR@10                │
│  Results saved to data/metrics/experiments.csv                   │
│  Visible in the Evaluation page                                  │
└─────────────────────────────────────────────────────────────────┘
