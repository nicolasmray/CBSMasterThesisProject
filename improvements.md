# Improvements & Architecture Comparison

---

## Why `thesisProject_langChain` gets correct numbers

The LangChain project uses three layers of protection that prevent the LLM from
ever inventing or misquoting numbers:

- **Programmatic statistics** — Python computes count, sum, min, max, and mean
  *before* the LLM is called. The LLM receives verified figures and is told to
  quote them verbatim.

- **Scalar fast-path** — single-cell results (COUNT, SUM, MAX, …) bypass the
  LLM entirely and are displayed directly as a metric card. No language model
  involved at all.

- **Separate data display** — the raw database table is rendered in the UI
  independently of the LLM answer, so users can cross-check every number on
  screen.

---

## Side-by-side comparison

### Numbers & accuracy

- **Number source**
  - *Before:* LLM "summarises" raw JSON — free to round or confabulate
  - *LangChain project:* Python computes stats first; LLM only interprets pre-verified figures

- **Scalar results** (COUNT, SUM, …)
  - *Before:* passed through the LLM for summarisation
  - *LangChain project:* detected and displayed directly — LLM not involved

- **Prompt instruction**
  - *Before:* `"summarize the key findings with numbers"` — vague, invites invention
  - *LangChain project:* `"direct answer"` + pre-computed stats block injected into prompt

- **LLM temperature**
  - *Before:* 0.3 — higher creative latitude
  - *LangChain project:* 0.2 with structured JSON output mode

### Data visibility

- **Where numbers appear**
  - *Before:* LLM answer only — no way to verify
  - *LangChain project:* raw DB table shown separately; LLM answer is the interpretation layer

- **Statistics module**
  - *Before:* none — all computation left to the LLM
  - *LangChain project:* dedicated `analysis/` package; Python computes, LLM interprets

### Validation & robustness

- **SQL validation**
  - *Before:* 3-retry loop in SQL agent
  - *LangChain project:* 5-pass validator (syntax, schema, security, performance) + 1 corrective retry with detailed error hints

- **Analysis layer**
  - *Before:* Response Agent synthesises directly from raw rows
  - *LangChain project:* `AnalysisOrchestrator` runs statistical analysis first, then the LLM interprets those stats

---

## Changes made to this project

### 1. Updated — `agents/response_agent.py`

**Root cause fixed:** the old prompt said *"summarize the key findings with numbers"*,
giving the LLM latitude to invent or round values.

- **Split answer into two independent parts** — raw query results are written
  directly into the answer as JSON (no LLM involved); the LLM is then called
  separately and receives only column names and row count, not the values, so
  it cannot hallucinate figures
- **Stricter system prompt** — model explicitly forbidden from restating numbers,
  scoped purely to business interpretation (trends, patterns, recommendations)
- **Temperature lowered** from `0.3` → `0.1` to reduce creative deviation
- **Clear error and empty-result paths** — instead of a useless "context is
  insufficient" fallback, the agent now reports exactly what happened (query
  returned no rows, execution error with message, etc.)

---

### 2. Updated — `ui/app.py`

- **Raw data table** — `st.dataframe(sql_result)` rendered in an expander
  directly below each answer so users can verify every number on screen
- **Session replay** — `sql_result` stored in `st.session_state` so the data
  table survives page refreshes and Streamlit reruns
- **`_render_assistant_turn()` helper** — extracted so both live turns and
  replayed history use identical rendering logic with no duplication

---

### 3. Updated — `mcp_server/tools/rag_tools.py`

- **Lazy Ollama initialisation** — `OllamaEmbeddings` was previously
  instantiated at module import time, crashing the entire MCP server process on
  startup whenever Ollama was not running; now created on first RAG call only,
  so SQL queries work independently of Ollama

---

### 4. Updated — `agents/sql_agent.py`

- **Compact schema format** — schema was sent as indented JSON (~30 000 tokens
  for AdventureWorks); now formatted as `schema.table: col(type), ...` lines,
  reducing token usage by ~70% and preventing 413 rate-limit errors
- **Dict→list normalisation** — FastMCP returns a bare dict for single-row
  results; now detected and wrapped into a one-element list so all downstream
  code receives a consistent `list[dict]`
- **DB error surfacing** — previously a dict with an `"error"` key was silently
  converted to `sql_result=[]` with no error recorded; now correctly extracted
  and stored in the `error` state field
- **ThreadPoolExecutor** — `asyncio.run()` now runs in a dedicated worker thread,
  isolating the MCP event loop from Streamlit's own loop and fixing the
  "unhandled errors in a TaskGroup" shutdown hang

---

### 5. Updated — `agents/rag_agent.py`

- **ThreadPoolExecutor** — same fix as `sql_agent.py`; async MCP calls run in
  a dedicated thread to avoid event loop conflicts with Streamlit




