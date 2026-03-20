# prototype2_agent — Multi-Agent BI Assistant

A production-ready multi-agent Business Intelligence assistant built with LangChain, LangGraph, and a real MCP server. Uses LLaMA 4 Scout (via Groq) for reasoning and BAAI/bge-large-en-v1.5 (via Ollama) for embeddings.

## Architecture

```
User → Streamlit UI → LangGraph Pipeline
                          ├── Orchestrator (routes by intent)
                          ├── RAG Agent (document Q&A via MCP → pgvector)
                          ├── SQL Agent (query generation via MCP → PostgreSQL)
                          ├── Chart Agent (Plotly visualization)
                          └── Response Agent (business-friendly synthesis)
```

All database and vector-store operations go through a **real MCP server** running as a subprocess. Agents connect via MCP client protocol — they never import tool functions directly.

## Prerequisites

### 1. Install Ollama and pull the embedding model

```bash
# Install Ollama (macOS)
brew install ollama

# Start the Ollama service
ollama serve

# Pull the embedding model (1024 dimensions)
ollama pull bge-large-en-v1.5
```

### 2. Set up PostgreSQL with pgvector

```bash
# macOS via Homebrew
brew install postgresql@16
brew install pgvector

# Start PostgreSQL
brew services start postgresql@16

# Create the database
createdb your_database_name

# Enable pgvector extension (done automatically on first run, but can be done manually)
psql your_database_name -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 3. Install Python dependencies

```bash
cd prototype2_agent
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your actual values:
#   GROQ_API_KEY=your_groq_key_here
#   DB_HOST=localhost
#   DB_PORT=5432
#   DB_NAME=your_database_name
#   DB_USER=your_db_user
#   DB_PASSWORD=your_db_password
```

## Usage

### Run the full application

```bash
python main.py
```

This will:
1. Initialize pgvector extension and create the `documents` table
2. Capture a schema snapshot of your database
3. Launch the Streamlit UI

### Ingest documents for RAG

```bash
# Ingest a text file
python -m rag.ingest path/to/document.txt

# Ingest a PDF
python -m rag.ingest path/to/document.pdf
```

### Run the MCP server standalone (for debugging)

```bash
python -m mcp_server.server
```

### Run the Streamlit UI directly

```bash
streamlit run ui/app.py
```

## Project Structure

```
prototype2_agent/
├── .env.example              # Environment variable template
├── requirements.txt          # Python dependencies
├── state.py                  # Shared AgentState TypedDict
├── mcp_client.py             # MCP client helper (used by all agents)
├── mcp_server/
│   ├── server.py             # Real MCP server (stdio transport)
│   └── tools/
│       ├── sql_tools.py      # SQL query execution, schema snapshot
│       └── rag_tools.py      # Semantic search, embed & store
├── agents/
│   ├── orchestrator.py       # Intent classification & routing
│   ├── rag_agent.py          # Document Q&A specialist
│   ├── sql_agent.py          # SQL generation with retry loop
│   ├── chart_agent.py        # Plotly chart generation
│   └── response_agent.py     # Business-friendly answer synthesis
├── graph.py                  # LangGraph StateGraph wiring
├── db/
│   ├── connection.py         # SQLAlchemy + pgvector setup
│   └── schema_snapshot.py    # DB schema introspection
├── rag/
│   └── ingest.py             # Document chunking & embedding ingestion
├── ui/
│   └── app.py                # Streamlit chat interface
└── main.py                   # Entrypoint
```

## How It Works

1. **Orchestrator** classifies the user's question as `rag`, `sql`, `chart`, or `hybrid`
2. Based on intent, the graph routes to the appropriate specialist agent(s)
3. **SQL Agent** generates PostgreSQL queries with sqlglot validation and a 3-retry loop
4. **RAG Agent** retrieves relevant document chunks via semantic search
5. **Chart Agent** renders Plotly visualizations from query results
6. **Response Agent** always runs last, synthesizing everything into a clear business answer
