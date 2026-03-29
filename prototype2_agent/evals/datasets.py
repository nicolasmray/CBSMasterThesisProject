"""Ground-truth evaluation datasets for all agents.

Each dataset is a list of dicts. Tests iterate over them parametrically.
Keep this file as the single source of truth for expected behaviours.
"""

# ─── Orchestrator: intent classification ──────────────────────────────────────

ORCHESTRATOR_CASES = [
    # SQL intent
    {"query": "How many customers do we have?", "expected_intent": "sql"},
    {"query": "What is the total revenue for 2024?", "expected_intent": "sql"},
    {"query": "Show me the top 10 products by sales volume", "expected_intent": "sql"},
    {"query": "Average order value by territory last year", "expected_intent": "sql"},
    {"query": "Which salesperson had the highest quota attainment?", "expected_intent": "sql"},
    {"query": "Count employees by department", "expected_intent": "sql"},
    {"query": "What is the total freight cost on purchase orders?", "expected_intent": "sql"},
    {"query": "List all vendors with credit rating 5", "expected_intent": "sql"},
    # RAG intent
    {"query": "What is the company policy on PII data?", "expected_intent": "rag"},
    {"query": "How is revenue defined according to our KPIs?", "expected_intent": "rag"},
    {"query": "What are the rules for showing salary data in reports?", "expected_intent": "rag"},
    {"query": "What is the fiscal year calendar?", "expected_intent": "rag"},
    {"query": "What is the minimum group size for demographic reports?", "expected_intent": "rag"},
    {"query": "Explain the vendor rejection threshold policy", "expected_intent": "rag"},
    # Chart intent
    {"query": "Show me a bar chart of revenue by territory", "expected_intent": "chart"},
    {"query": "Plot monthly sales trend for 2024 as a line chart", "expected_intent": "chart"},
    {"query": "Visualize product category revenue as a pie chart", "expected_intent": "chart"},
    {"query": "Create a chart of employee count by department", "expected_intent": "chart"},
    # Hybrid intent
    {"query": "What were our top products by revenue, and what does our policy say about discontinued products?", "expected_intent": "hybrid"},
    {"query": "Show me vendor spend and explain the vendor compliance policy", "expected_intent": "hybrid"},
]


# ─── SQL agent: query correctness ────────────────────────────────────────────
# Each case has the user query and either:
#   - expected_tables: tables that MUST appear in the generated SQL
#   - expected_columns: columns that MUST appear
#   - forbidden_patterns: patterns that must NOT appear
#   - min_rows: minimum expected row count (0 = allow empty)
#   - result_check: callable(result) -> bool for custom validation

SQL_AGENT_CASES = [
    {
        "query": "How many customers are there?",
        "expected_tables": ["sales.customer"],
        "expected_columns": ["customerid"],
        "min_rows": 1,
        "result_check": lambda r: len(r) == 1 and list(r[0].values())[0] > 0,
        "description": "Simple COUNT on customer table",
    },
    {
        "query": "What is the total revenue for 2024?",
        "expected_tables": ["sales.salesorderdetail", "sales.salesorderheader"],
        "expected_columns": ["unitprice", "orderqty"],
        "forbidden_patterns": ["CURRENT_DATE", "NOW()"],
        "min_rows": 1,
        "description": "Revenue calculation with correct formula",
    },
    {
        "query": "Top 5 products by units sold",
        "expected_tables": ["sales.salesorderdetail"],
        "expected_columns": ["orderqty"],
        "min_rows": 5,
        "description": "Product ranking by volume",
    },
    {
        "query": "Average order value by territory",
        "expected_tables": ["sales.salesorderheader"],
        "min_rows": 1,
        "description": "AOV grouped by territory",
    },
    {
        "query": "Count of employees by department",
        "expected_tables": ["humanresources.employee"],
        "min_rows": 1,
        "description": "Employee headcount by department",
    },
    {
        "query": "Monthly revenue trend for 2024",
        "expected_tables": ["sales.salesorderdetail", "sales.salesorderheader"],
        "forbidden_patterns": ["TO_CHAR", "CURRENT_DATE", "NOW()"],
        "min_rows": 1,
        "description": "Time-series with EXTRACT, not TO_CHAR",
    },
    {
        "query": "Which vendors have a credit rating of 5?",
        "expected_tables": ["purchasing.vendor"],
        "expected_columns": ["creditrating"],
        "min_rows": 0,
        "description": "Simple filter on vendor table",
    },
    {
        "query": "Revenue by product category",
        "expected_tables": ["production.product"],
        "min_rows": 1,
        "description": "Joins through product hierarchy",
    },
    {
        "query": "Show me the last 12 months of sales",
        "expected_tables": ["sales.salesorderheader"],
        "forbidden_patterns": ["CURRENT_DATE", "NOW()"],
        "min_rows": 1,
        "description": "Relative time anchored to MAX(orderdate)",
    },
    {
        "query": "Total purchase spend by vendor",
        "expected_tables": ["purchasing.purchaseorderheader"],
        "min_rows": 1,
        "description": "Purchasing aggregation",
    },
]


# ─── RAG agent: retrieval quality ────────────────────────────────────────────
# Each case has the query and the expected content that SHOULD be in retrieved chunks.

RAG_RETRIEVAL_CASES = [
    {
        "query": "How is revenue calculated?",
        "must_contain": ["unitprice", "orderqty", "unitpricediscount"],
        "expected_source": "adventureworks_kpis.txt",
        "min_chunks": 1,
        "description": "Revenue formula retrieval",
    },
    {
        "query": "What is the PII policy for employee data?",
        "must_contain": ["passwordhash", "nationalidnumber"],
        "expected_source": "company_policies.txt",
        "min_chunks": 1,
        "description": "PII policy retrieval",
    },
    {
        "query": "What is the fiscal year calendar?",
        "must_contain": ["July", "June", "Q1"],
        "expected_source": "company_policies.txt",
        "min_chunks": 1,
        "description": "Fiscal year definition",
    },
    {
        "query": "How do I calculate gross margin per product?",
        "must_contain": ["listprice", "standardcost"],
        "expected_source": "adventureworks_kpis.txt",
        "min_chunks": 1,
        "description": "Gross margin formula",
    },
    {
        "query": "What is the vendor rejection threshold?",
        "must_contain": ["rejectedqty", "5%"],
        "expected_source": "company_policies.txt",
        "min_chunks": 1,
        "description": "Vendor quality threshold",
    },
    {
        "query": "What are the rules for showing salary data?",
        "must_contain": ["confidential", "aggregate"],
        "expected_source": "company_policies.txt",
        "min_chunks": 1,
        "description": "Salary confidentiality policy",
    },
    {
        "query": "What are the order status codes?",
        "must_contain": ["Shipped", "Rejected"],
        "expected_source": "company_policies.txt",
        "min_chunks": 1,
        "description": "Order status definitions",
    },
    {
        "query": "How is customer lifetime value computed?",
        "must_contain": ["customerid", "revenue"],
        "expected_source": "adventureworks_kpis.txt",
        "min_chunks": 1,
        "description": "CLV formula retrieval",
    },
    {
        "query": "What is the reorder point policy?",
        "must_contain": ["reorderpoint", "inventory"],
        "expected_source": "adventureworks_kpis.txt",
        "min_chunks": 1,
        "description": "Inventory reorder policy",
    },
    {
        "query": "What fields are considered high sensitivity PII?",
        "must_contain": ["passwordhash", "nationalidnumber"],
        "expected_source": "company_policies.txt",
        "min_chunks": 1,
        "description": "PII classification levels",
    },
]


# ─── End-to-end: full pipeline flows ─────────────────────────────────────────

E2E_CASES = [
    {
        "query": "How many customers do we have?",
        "expected_intent": "sql",
        "answer_must_contain_number": True,
        "should_have_sql": True,
        "should_have_rag": False,
        "description": "Simple SQL count",
    },
    {
        "query": "What is the company policy on employee PII?",
        "expected_intent": "rag",
        "should_have_sql": False,
        "should_have_rag": True,
        "answer_keywords": ["password", "pii"],
        "description": "RAG policy lookup",
    },
    {
        "query": "Show me a chart of revenue by territory",
        "expected_intent": "chart",
        "should_have_sql": True,
        "should_have_chart": True,
        "description": "Chart generation flow",
    },
    {
        "query": "Total revenue for 2024",
        "expected_intent": "sql",
        "should_have_sql": True,
        "answer_must_contain_number": True,
        "description": "Revenue calculation e2e",
    },
    {
        "query": "What is the fiscal year calendar and how many employees are in each department?",
        "expected_intent": "hybrid",
        "should_have_sql": True,
        "should_have_rag": True,
        "description": "Hybrid SQL+RAG flow",
    },
]


# ─── Performance baseline thresholds ─────────────────────────────────────────

PERF_THRESHOLDS = {
    "orchestrator_max_seconds": 30,
    "sql_agent_max_seconds": 60,
    "rag_agent_max_seconds": 30,
    "chart_agent_max_seconds": 30,
    "e2e_max_seconds": 120,
    "semantic_search_max_seconds": 20,  # first call has Ollama cold start
    "schema_load_max_seconds": 0.5,
}
