# Improvements & Architecture Comparison

-----21.3.2026------

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

------23.3.2026---------

### Fix: Multi-row query results now returned in full (`mcp_client.py`)
- **Problem:** Queries returning multiple rows (e.g. sales per category) only showed the first row.
- **Root cause:** FastMCP serializes each dict in a `list[dict]` return as a separate MCP content block. `call_tool` only read `content[0]`, discarding all subsequent rows.
- **Fix:** `call_tool` now iterates over all content blocks and assembles them into a list, so all rows are returned correctly.

### Fix: Clean result display instead of raw JSON (`response_agent.py`)
- **Problem:** Results were displayed as raw JSON, e.g. `Query returned 1 row(s): [ { "count": 19820 } ]`.
- **Fix:** Added `_format_rows` helper that formats results as a plain number for single-value results (e.g. `19820`) or a plain-text `|`-separated table for multi-row/multi-column results.

---

------23.3.2026---------

### Interactive Plotly charts with 3 chart type options (`ui/app.py`, `chart_agent.py`)
- Replaced static PNG with interactive Plotly charts (`fig.to_json()` / `pio.from_json()`).
- LLM ranks top 3 chart types; user picks via radio button. Options stored in `chart_spec["options"]`.

### Expanded chart library (`chart_agent.py`)
- Added: `grouped_bar`, `small_multiples`, `area`, `scatter`, `histogram`, `box`, `waterfall`, `treemap`. Removed: `funnel`.
- System prompt updated with BA best-practice triplet recommendations per question type.

### Y-axis auto-scaling (`chart_agent.py`)
- `_compute_y_max` replaces LLM-estimated `y_max` (which was based on a 5-row sample and clipped data).
- Waterfall: y_max covers the cumulative sum, fixing invisible intermediate bars.

### Multi-year line/area charts (`chart_agent.py`, `sql_agent.py`)
- SQL returns separate `year` / `month` integer columns; chart agent draws one line per year with Jan–Dec x-axis ordering.
- `_to_label` normalises PostgreSQL float EXTRACT results (e.g. `2024.0` → `"2024"`) for consistent series matching.

### Fix: Various chart rendering bugs (`chart_agent.py`)
- Bar chart: constructs `YYYY-MM` labels when month numbers repeat across years.
- Column name fallback: validates `x_col`/`y_col` against actual columns before rendering.
- `_format_rows`: added missing 3+ column markdown table return (was returning `None`, crashing response).
- Year numbers no longer formatted as `2,022` — thousand separators only applied for values ≥ 10,000.
- sqlglot `TO_CHAR` transpilation error suppressed with `unsupported_level=IGNORE`.
