# 🌏 Climate Knowledge Graph RAG System

A production-oriented **Climate Research Assistant** powered by:

- **Neo4j Knowledge Graph**
- **Vector Search (Chunk + Document embeddings)**
- **LLM-based Metadata Normalization**
- **Scrapy Web Crawler**
- **OpenWebUI (Frontend RAG Interface)**

This system ingests trusted climate PDFs and web domains, extracts structured metadata, builds a graph, embeds semantic vectors, and enables controlled retrieval for grounded conversational answers.

---

# 🎯 Project Objective

The objective of this project is to build a:

> Trust-aware, document-grounded Climate Knowledge Graph  
> that powers a controlled RAG-based research assistant.

The system:

- Ingests climate PDFs (local or crawled)
- Extracts structured metadata using an LLM
- Canonicalizes publishers and authors
- Embeds document identity and text chunks
- Builds graph relationships (Author, Publisher, Year)
- Creates Neo4j vector indexes
- Enables OpenWebUI to retrieve only grounded sources
- Returns document-level citations only (no chunk leakage)

This is **not** a generic chatbot.  
It is a **source-controlled climate research system**.

---

# 🧠 System Architecture

The system is now split cleanly into backend modules plus a notebook orchestrator.

```
CCRC_Agentic_AI/
│
├── db_manager.py
├── ingestor.py
├── main.ipynb
├── tools/                 (OpenWebUI tool layer)
│   │
│   └──climate_knowledge_graph.py
└── README.md
```

---

# 📦 Backend Modules

## 1️⃣ `db_manager.py`

Contains:

- `Neo4jManager` class
- Constraint setup
- Schema management
- Graph linking logic
- Publisher retrieval
- Database reset utilities

Responsible for:
- All direct Neo4j operations
- Graph structure management
- Author/Publisher/Year relationships

---

## 2️⃣ `ingestor.py`

Contains:

- `store_in_neo4j`
- `SmartIngestor`
- `EduSpider`
- `Neo4jPipeline`
- `run_spider_process`
- `SmartURLIngestor`

Responsible for:

- PDF ingestion (local + streamed)
- Scrapy domain crawling
- Metadata extraction via LLM
- Trust scoring
- Chunking and embedding
- Storing documents + chunks
- Immediate graph linking

---

## 3️⃣ `main.ipynb`

This notebook:

- Configures endpoints
- Instantiates backend classes
- Runs ingestion
- Creates vector indexes
- Controls ingestion order

It is the orchestration layer.

---

# 🗄️ Data Model (Neo4j)

## Nodes

- `Document`
- `Chunk`
- `Author`
- `Publisher`
- `Year`
- `WebPage`

## Relationships

```cypher
(Author)-[:WROTE]->(Document)
(Publisher)-[:PUBLISHED]->(Document)
(Document)-[:PUBLISHED_IN]->(Year)
(Document)-[:HAS_CHUNK]->(Chunk)
```

---

# 🔍 Ingestion Pipeline

### Step 1 — Load Document
Uses `UnstructuredLoader`:
- Strategy: `hi_res`
- Chunking: `by_title`
- Max ~1000 characters per chunk

---

### Step 2 — Metadata Extraction (LLM)

The local LLM extracts structured JSON:

```json
{
  "title": "Annual Climate Statement 2022",
  "authors": ["T. Lane"],
  "publishers": ["Bureau of Meteorology"],
  "year": 2022,
  "full_date": "12 March 2023"
}
```

Rules enforced:
- Authors = humans only
- Publishers = agencies/journals only
- Canonicalization enforced
- Reporting year vs publication year logic handled

---

### Step 3 — Trust Scoring

Trust Level 1 if:
- Domain matches trusted list (.gov.au, .csiro.au, etc.)
- OR publisher matches trusted list

Otherwise:
- Trust Level 2

---

### Step 4 — Embedding

Two embeddings generated:

1. Document identity embedding  
   (Title + Authors + Publishers + Year)

2. Chunk embeddings  
   For semantic retrieval

---

### Step 5 — Storage

`store_in_neo4j`:

- Merges Document node
- Merges Authors
- Merges Publishers
- Merges Year
- Deletes old chunks
- Creates new Chunk nodes
- Links everything atomically

---

# 🔎 Vector Index Creation

After ingestion completes, run in the notebook:

```python
create_index()
```

This creates:

```cypher
CREATE VECTOR INDEX chunk_vector_index
FOR (c:Chunk) ON (c.embedding)

CREATE VECTOR INDEX document_vector_index
FOR (d:Document) ON (d.embedding)
```

- Similarity: cosine
- Waits until index is ONLINE
- Drops old index if present

⚠️ Neo4j 5+ required.

---

# 🧭 climate_orchestration.ipynb Usage

## 1️⃣ Configure Endpoints

Edit this cell:

```python
# Neo4j
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"

# Embedding API
EMBED_URL = "http://localhost:8000/v1/embeddings"
EMBED_MODEL = "your-embedding-model"

# LLM API (for metadata extraction)
LLM_URL = "http://localhost:8000/v1/chat/completions"
LLM_MODEL = "your-llm-model"

VECTOR_DIMENSION = 1024
```

Adjust only:
- Neo4j URI / credentials
- Embedding endpoint
- LLM endpoint
- Model names
- Vector dimension

No backend file edits required.

---

## 2️⃣ Initialize DB

```python
db_manager = Neo4jManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
db_manager.setup_constraints()
```

---

## 3️⃣ Ingest Local PDFs

```python
ingestor = SmartIngestor(
    db_manager,
    EMBED_URL,
    EMBED_MODEL,
    LLM_URL,
    LLM_MODEL
)

ingestor.process_local_queue()
```

---

## 4️⃣ Ingest a Domain

```python
url_ingestor = SmartURLIngestor(
    db_config={
        "uri": NEO4J_URI,
        "user": NEO4J_USER,
        "password": NEO4J_PASSWORD
    },
    embed_url=EMBED_URL,
    embed_model=EMBED_MODEL,
    llm_url=LLM_URL,
    llm_model=LLM_MODEL,
    Neo4jManagerClass=Neo4jManager
)

url_ingestor.ingest_domain("https://www.bom.gov.au")
```

---

## 5️⃣ Create Vector Indexes

```python
create_index()
```

Wait until both indexes are ONLINE before querying.

---

# 🌐 OpenWebUI Integration

Frontend:

- Lives entirely inside OpenWebUI
- Uses `tools.py`
- Calls Neo4j for:
  - Document vector search
  - Chunk vector search
- Returns:
  - Document-level citations only
  - No chunk text exposed
- LLM restricted to retrieved context

This keeps:
- Clean UI
- No hallucinated sources
- Deterministic grounding

---

# 🔒 Design Principles

- Deterministic ingestion
- Canonicalized metadata
- Trust-aware ranking
- Hybrid graph + vector retrieval
- Chunk grounding without chunk exposure
- Document-level citations only
- Backend/frontend separation

---

# 📌 Why This Architecture?

Pure chunk RAG causes:
- Citation hallucination
- Identity drift
- Weak document grounding

Pure graph search lacks:
- Semantic matching

Hybrid approach gives:

- Structured entity linking
- Semantic similarity
- Reliable citations
- Trust-aware filtering
- Clean conversational UX

---

# 🚀 Current Capabilities

✔ PDF ingestion  
✔ Web domain crawling  
✔ Metadata normalization  
✔ Publisher canonicalization  
✔ Trust scoring  
✔ Graph linking  
✔ Chunk + document embeddings  
✔ Neo4j vector indexes  
✔ OpenWebUI RAG integration  

---

# 🧪 Requirements

- Neo4j 5+
- Vector index support enabled
- Local embedding endpoint
- Local LLM endpoint
- Python 3.10+

---

# 📚 Intended Use

- Climate policy analysis
- Government reporting comparison
- Cross-publisher linking
- Research assistant for Australian climate data
- Structured climate knowledge exploration

---

# 🧾 Summary

This project builds a:

> Neo4j-backed, vector-indexed, trust-aware  
> Climate Knowledge Graph powering a grounded RAG assistant.

It combines:

- Knowledge Graph structure
- Vector search
- LLM metadata normalization
- Controlled retrieval
- Clean citation presentation

Designed for reliability for use in academia.
