# BI Agent — Proof-of-Concept Artifact

> **Master's Thesis Artifact**
> This system is the proof-of-concept artifact constructed for a master's thesis at Copenhagen Business School. The thesis investigates the feasibility of building an AI-powered business intelligence (BI) agent for small and medium-sized enterprises (SMEs) using existing open-source and commercially available components — without relying on costly proprietary enterprise platforms.

---

## Research Context

Small and medium-sized enterprises account for 98% of all enterprises in the EU and generate 65% of private-sector employment, yet they consistently under-invest in business intelligence relative to their larger counterparts. The dominant barriers are cost, technical complexity, and a shortage of in-house data skills. Enterprise analytics platforms impose licensing costs, implementation overhead, and require SQL-literate employees to extract value — a combination that largely excludes SMEs from the kind of flexible, query-driven insight that data-rich competitors enjoy.

Natural language interfaces to relational databases offer a promising remedy: if a decision-maker could ask a question in plain language and receive a correctly executed query result, the SQL-literacy barrier would be substantially lowered. State-of-the-art text-to-SQL systems now report execution accuracy above 80% on established academic benchmarks, but translating this benchmark performance into deployable systems for real SME databases remains an open problem. Operational schemas in practice contain inconsistent naming conventions, undocumented relationships, and structures that diverge sharply from the clean tables of academic benchmarks.

This artifact addresses that gap by composing retrieval-augmented generation (RAG), LLM-based text-to-SQL, agent orchestration (LangGraph), and standardized tool integration (Model Context Protocol) into a coherent BI agent that is feasible for SME deployment. The research adopts a Design Science Research (DSR) methodology and evaluates the artifact across three feasibility dimensions: **technical**, **architectural**, and **operational**.

---

## System Overview

```
User Query (natural language)
        │
        ▼
  Streamlit UI 
        │                                                         
        ▼                                                         
  LangGraph Pipeline                                              
        │                                                         
        ├──▶ Orchestrator Agent                                   
        │       └── Classifies intent: rag | sql | chart | hybrid 
        │                                                        
        ├──▶ RAG Agent                                           
        │       └── Semantic search → pgvector → LLM reranking  
        │                                                        
        ├──▶ SQL Agent                                           
        │       └── LLM → sqlglot validation → MCP → PostgreSQL  
        │                                                        
        ├──▶ Chart Agent                                         
        │       └── LLM chart selection → Plotly rendering       
        │                                                        
        └──▶ Response Agent                                      
                └── Python-computed facts + LLM synthesis
```

All SQL execution is isolated behind a **real MCP server** running as a subprocess. Agents never import database tools directly — they communicate exclusively via the Model Context Protocol (MCP) client, enforcing a clean separation of concerns and providing a foundation for future tool extensibility.

---

## Architecture

### Agent Orchestration (LangGraph)

The pipeline is implemented as a LangGraph `StateGraph`. A shared `AgentState` TypedDict is passed between nodes and updated at each step. Routing is fully conditional based on intent classification.

```
START
  └──▶ orchestrator
          ├──(rag | hybrid)──▶ rag_agent
          │                      ├──(hybrid)─▶ sql_agent ─▶ chart_agent
          │                      │                             └──▶ response_agent
          │                      │                                        └──▶ END
          │                      └──(rag)────▶ response_agent ──▶ END
          └──(sql | chart)──▶ sql_agent
                                   ├──(chart)──▶ chart_agent ──▶ response_agent
                                   │                                     └──▶ END
                                   └──(sql)────────────────▶ response_agent
                                                                    └──▶ END
```

### Shared State (`state.py`)

| Field            | Description                                        |
|------------------|----------------------------------------------------|
| `user_query`     | Original natural language question                 |
| `intent`         | Classified intent: rag / sql / chart / hybrid      |
| `plan`           | Orchestrator's reasoning plan                      |
| `schema_context` | Compact DB schema string injected into SQL prompts |
| `sql_query`      | Generated SQL                                      |
| `sql_result`     | Executed query result (list of row dicts)          |
| `error`          | Error message if SQL execution failed              |
| `retry_count`    | Number of SQL generation retries (max 3)           |
| `rag_context`    | Concatenated retrieved document chunks             |
| `rag_chunks`     | Raw chunk metadata (source, score, text)           |
| `rag_fallback`   | True if fallback retrieval was used                |
| `chart_spec`     | Chart type(s) and Plotly figure spec               |
| `final_answer`   | Synthesized response shown to the user             |

---

## Components

### Orchestrator (`agents/orchestrator.py`)

Classifies the user's question into one of four intent categories:

| Intent  | Meaning: 
|---------|-----------------------------------------------------------|
| `sql`   | Requires structured database query                        |
| `rag`   | Requires document retrieval (policies, KPIs, definitions) |
| `chart` | Requires visualization of query results                   |
| `hybrid`| Requires both document context and a database query       |

The orchestrator calls the LLM with the user query and returns `{intent, plan}`. The plan is a short natural-language description of how the system will approach the question.

### SQL Agent (`agents/sql_agent.py`)

The SQL agent translates natural language questions into valid PostgreSQL queries and executes them through the MCP server. It is the most complex component in the system.

**Generation:** The LLM (LLaMA 4 Scout via Groq by default) receives a detailed system prompt that includes:
- Fully-qualified table and column names from the schema snapshot
- Foreign key relationships for JOIN guidance
- Time handling rules (EXTRACT, INTERVAL, relative date subqueries)
- Aggregation and grouping rules
- Domain-specific formulas (e.g. revenue = unitprice × qty × (1 − discount))
- Safety constraints and ranking conventions

**Validation:** Generated SQL is parsed by `sqlglot` before execution. If the parse fails, the agent retries with a corrected prompt. `sqlglot` also transpiles T-SQL idioms to PostgreSQL syntax, improving robustness against LLM-generated dialect mismatches.

**Safety:** The agent blocks all destructive SQL operations (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `TRUNCATE`) and access to sensitive columns (password hashes, national ID numbers).

**Retry loop (max 3 attempts):**
1. Parse error → LLM retried with error message and semantic search context.
2. Execution error (undefined column) → column name recorded in `banned_columns.json`; LLM retried with updated schema.
3. Empty result → semantic search context injected; LLM retried.

**Date range awareness:** The agent caches the min/max date range from fact tables at startup and warns if a query references years outside the DB's actual date range.

**MCP execution:** The validated query is sent to the MCP server via `mcp_client.call_tool("run_sql_query", ...)` — never via direct function call.

### RAG Agent (`agents/rag_agent.py`)

Handles questions that require business context from ingested documents (KPI definitions, company policies, domain terminology).

**Retrieval strategy:**
1. Primary search: `semantic_search()` with cosine similarity threshold ≥ 0.55.
2. If threshold returns no results: fallback to `semantic_search_no_threshold()` (top 10 regardless of score), flagging `rag_fallback=True`.
3. LLM reranker filters the retrieved chunks for genuine relevance.

**Vector store:** BAAI/bge-large-en-v1.5 embeddings (1024 dimensions) generated locally via Ollama, stored in PostgreSQL with the `pgvector` extension (HNSW index on cosine distance).

### Chart Agent (`agents/chart_agent.py`)

Activated when intent is `chart` or `hybrid`. Receives the SQL result set and generates interactive Plotly visualizations.

**Supported chart types:**

| Category     | Types                                                    |
|--------------|----------------------------------------------------------|
| Bar          | bar, grouped\_bar, stacked\_bar, normalized\_bar         |
| Line/Area    | line, area                                               |
| Distribution | scatter, histogram, box                                  |
| Specialized  | waterfall, treemap, donut, small\_multiples, bar-line    |

**Smart defaults:**
- Detects year+month column combinations and creates readable period labels (`Jan '22`).
- Detects rate/change/growth columns and prioritizes them as the y-axis over raw totals.
- Auto-pivots wide data (multiple metric columns) to long format for grouped/stacked charts.
- Adds an OLS trend line to scatter plots when r² > 0.05.
- Renders wide charts (>40 unique x-values) with horizontal scroll.

The LLM selects 1–3 appropriate chart types for the data; the Streamlit UI displays all options with a type-selector so the user can switch between them.

### Response Agent (`agents/response_agent.py`)

Always runs last. Synthesizes a final, business-readable answer from whatever the upstream agents produced.

**Response paths:**

| Condition                     | Behavior                                            |
|-------------------------------|-----------------------------------------------------|
| SQL results, large (>12 rows) | Python-computed key facts (highest, lowest, average,|
|                               |   top-3, total) — no LLM, no hallucination          |  
| SQL results, small (≤12 rows) | Formatted table of all rows, optional LLM           |
|                               |   interpretation grounded on facts                  |
| SQL returned no rows          | Reports empty result, shows the query               |
| SQL execution error           | Friendly error with diagnostic classification       |
|                               |   (missing column, syntax, undefined table)         |
| RAG only                      | LLM synthesizes answer from retrieved document      |
|                               |   chunks                                            |
| Nothing                       | Generic "no data available" fallback                |

The Python-only fact extraction path (`_extract_key_facts()`) is deliberately chosen for large result sets to eliminate LLM hallucination on numeric data.

### MCP Server (`mcp_server/server.py`)

A FastMCP server running as a subprocess on stdio transport. Exposes a single registered tool:

 `run_sql_query` 

Description: Executes a SQL string via SQLAlchemy and returns results as a list of row dicts |

The MCP client (`mcp_client.py`) provides `call_tool()` and `list_tools()` wrappers used by the SQL agent to communicate with the subprocess server without direct imports.

### Document Ingestion (`rag/ingest.py`)

Populates the vector store with domain knowledge. Called automatically at startup via `ingest_knowledge_base()`, which scans the `knowledge_base/` directory and skips already-ingested files.

**Pipeline:**
1. Load document: `TextLoader` (txt, md) or `PyPDFLoader` (pdf).
2. Chunk: `RecursiveCharacterTextSplitter` (400 characters, 60-character overlap).
3. Embed: Ollama (`mxbai-embed-large`, 1024 dimensions, local).
4. Store: pgvector with source filename and chunk index as JSONB metadata.

Manual ingestion: `python -m rag.ingest path/to/document.txt`

### Database Layer (`db/`)

| Module               | Responsibility                                                 |
|----------------------|----------------------------------------------------------------|
| `connection.py`      | SQLAlchemy engine factory, `init_pgvector()`                   |
| `schema_snapshot.py` | Introspects `information_schema`, caches compact schema string |
|                      |   for LLM prompts, saves to `schema_snapshot.json`             |
| `fk_snapshot.py`     | Reads declared FK constraints, merges with manually defined    |
|                      |   logical FKs, saves to `fk_snapshot.json`                     |
| `vector_store.py`    | Cosine similarity search, embedding storage, threshold and a   |
|                      |   no-threshold retrieval                                       |
| `banned_columns.py`  | Persistent registry of confirmed non-existent column names;    |
|                      |  prevents repeated LLM hallucination on the same column        |

### LLM Configuration (`llm_config.py`)

Centralized LLM provider configuration supporting Groq (cloud) and Ollama (local) backends.

| Setting            | Default       | Description                                      |
|--------------------|---------------|--------------------------------------------------|
| `PROVIDER`         | `groq`        | LLM backend                                      |
| `GROQ_MODEL`       | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq cloud model   |
| `OLLAMA_MODEL`     | `llama3.1`    | Local model fallback                             |
| `EMBEDDING_MODEL`  | `mxbai-embed-large` | Local embedding model (Ollama)             |
| `RAG_TOP_K`        | `10`          | Max chunks to retrieve                           |
| `RAG_SIMILARITY_THRESHOLD`| `0.55` | Min cosine similarity for primary retrieval      |

Supports **Groq API key rotation** (`GroqKeyPool`) for rate-limit resilience when running evaluations or high-volume queries. Per-agent model and temperature overrides are configurable via `AGENT_MODELS` and `AGENT_TEMPERATURES` dicts.

---

## Project Structure

```
prototype2_agent/
├── main.py                   # Entry point: DB init, schema capture, UI launch
├── graph.py                  # LangGraph StateGraph: nodes + conditional routing
├── state.py                  # Shared AgentState TypedDict
├── llm_config.py             # LLM/embedding provider config, key rotation
├── mcp_client.py             # MCP client helper (used by all agents)
│
├── agents/
│   ├── orchestrator.py       # Intent classification & routing plan
│   ├── sql_agent.py          # Text-to-SQL: generation, validation, retry, MCP execution
│   ├── rag_agent.py          # Document Q&A: semantic search, reranking
│   ├── chart_agent.py        # Plotly chart generation: 15+ chart types
│   └── response_agent.py     # Final answer synthesis: Python facts + LLM
│
├── db/
│   ├── connection.py         # SQLAlchemy engine, pgvector init
│   ├── schema_snapshot.py    # DB schema introspection → schema_snapshot.json
│   ├── fk_snapshot.py        # FK introspection + manual FKs → fk_snapshot.json
│   ├── vector_store.py       # pgvector cosine search, embed & store
│   └── banned_columns.py     # Persistent registry of non-existent columns
│
├── mcp_server/
│   ├── server.py             # FastMCP server (stdio transport)
│   └── tools/
│       └── sql_tools.py      # run_sql_query tool implementation
│
├── rag/
│   └── ingest.py             # Document chunking & embedding ingestion
│
├── ui/
│   └── app.py                # Streamlit chat UI
│
├── evals/
│   ├── run_evals.py          # Evaluation runner
│   ├── golden_dataset.py     # Golden question-answer pairs
│   ├── groq_judge.py         # LLM-as-judge evaluation
│   ├── langsmith_tracing.py  # LangSmith observability integration
│   ├── promptfoo_provider.py # PromptFoo evaluation provider
│   ├── score_recorder.py     # Evaluation score tracking
│   └── test_*.py             # Agent-level and end-to-end test suites
│
├── knowledge_base/
│   ├── adventureworks_kpis.txt  # KPI definitions and business metrics
│   └── company_policies.txt     # Company policies for RAG context
│
├── schema_snapshot.json      # Cached DB schema (generated at startup)
├── fk_snapshot.json          # Cached FK relationships (generated at startup)
├── banned_columns.json       # Persistent non-existent column registry
├── requirements.txt          # Python dependencies
└── .env.example              # Environment variable template
```

---

## Prerequisites

### 1. Ollama (local embeddings)

```bash
# macOS
brew install ollama
ollama serve
ollama pull mxbai-embed-large
```

### 2. PostgreSQL with pgvector

```bash
# macOS via Homebrew
brew install postgresql@16
brew install pgvector
brew services start postgresql@16

createdb your_database_name
psql your_database_name -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 3. Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_groq_key_here
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password
```

---

## Usage

### Start the application

```bash
python main.py
```

On startup this will:
1. Initialize the pgvector extension and create the `rag_chunks` table.
2. Capture a schema snapshot of your PostgreSQL database (`schema_snapshot.json`).
3. Capture foreign key relationships (`fk_snapshot.json`).
4. Ingest any new documents in `knowledge_base/` into the vector store.
5. Launch the Streamlit chat UI.

### Ingest documents manually

```bash
# Plain text or Markdown
python -m rag.ingest path/to/document.txt

# PDF
python -m rag.ingest path/to/document.pdf
```

### Run the MCP server standalone (debugging)

```bash
python -m mcp_server.server
```

### Run the Streamlit UI directly

```bash
streamlit run ui/app.py
```

### Run evaluations

```bash
cd evals
python run_evals.py
```

---

## Key Design Decisions

**Why MCP for SQL execution?**
The Model Context Protocol enforces a clean boundary between agent logic and database execution. Agents cannot call database functions directly — they must go through the registered tool interface. This makes the SQL execution surface explicit and auditable, and allows new tools to be added to the MCP server without modifying agent code.

**Why pgvector instead of a dedicated vector database?**
Using PostgreSQL for both relational data and vector storage eliminates an operational dependency. For an SME deployment, maintaining a single database system is significantly simpler than operating a separate vector store alongside a relational database.

**Why local embeddings (Ollama)?**
All embedding operations use a locally-running model (Ollama). This keeps document content off third-party APIs, avoids per-token embedding costs, and allows the system to operate in environments with data privacy constraints — a common SME concern.

**Why Python-computed facts in the response agent?**
For large result sets, the response agent extracts key metrics (highest, lowest, average, total, top-3) in pure Python rather than asking the LLM to summarize raw rows. This eliminates a class of numeric hallucination where LLMs confidently report wrong aggregations.

**Why a banned columns registry?**
LLMs occasionally hallucinate column names that do not exist in the schema. When PostgreSQL returns an `UndefinedColumn` error, the column name is parsed from the error message and recorded in `banned_columns.json`. On subsequent queries, the system injects the banned column list into the prompt, preventing the same hallucination from repeating.

---

## Evaluation

The `evals/` directory contains a full evaluation framework:

| Component               | Description                                                 |
|-------------------------|-------------------------------------------------------------|
| `golden_dataset.py`     | Curated question–answer pairs with expected SQL and results |
| `groq_judge.py`         | LLM-as-judge scoring (correctness, relevance, faithfulness) |
| `langsmith_tracing.py`  | LangSmith run tracing and observability                     |
| `promptfoo_provider.py` | PromptFoo integration for systematic prompt evaluation      |
| `score_recorder.py`     | Tracks scores across model configurations and runs          |
| `test_*.py`             | Unit and integration tests for each agent and end-to-end    |
|                         |   flows                                                     |

The evaluation framework was used in the thesis to compare performance across LLM configurations (Groq cloud models vs. local Ollama models) on the golden dataset, measuring execution accuracy, result correctness, and cost per query.

---

## Technology Stack

| Layer                | Technology                                        |
|----------------------|---------------------------------------------------|
| Agent orchestration  | LangGraph                                         |
| LLM (cloud)          | LLaMA 4 Scout via Groq API                        |
| LLM (local fallback) | Ollama (llama3.1)                                 |
| Embeddings           | BAAI/mxbai-embed-large via Ollama (local)         |
| Vector store         | PostgreSQL + pgvector (HNSW index)                |
| Relational database  | PostgreSQL                                        |
| SQL validation       | sqlglot                                           |
| Tool integration     | Model Context Protocol (FastMCP, stdio transport) |
| Visualization        | Plotly                                            |
| UI                   | Streamlit                                         |
| ORM                  | SQLAlchemy                                        |
