"""LangSmith tracing utilities for evaluation runs.

Provides helpers to wrap agent calls with LangSmith traces so that
every evaluation run is logged, including inputs, outputs, latency,
and token usage (when available from Groq).

Usage:
  1. Set LANGSMITH_API_KEY in .env
  2. Traces appear at https://smith.langchain.com under project "prototype2-evals"

If LANGSMITH_API_KEY is not set, tracing is silently disabled.
"""

import os
import functools
import time
from contextlib import contextmanager

# Optional import — graceful degradation if langsmith not configured
try:
    from langsmith import Client, traceable
    from langsmith.run_helpers import get_current_run_tree

    LANGSMITH_AVAILABLE = bool(os.getenv("LANGSMITH_API_KEY"))
except ImportError:
    LANGSMITH_AVAILABLE = False
    traceable = lambda **kwargs: lambda f: f  # no-op decorator


def get_langsmith_client():
    """Return a LangSmith client if configured, else None."""
    if not LANGSMITH_AVAILABLE:
        return None
    return Client()


@contextmanager
def trace_eval_run(test_name: str, metadata: dict | None = None):
    """Context manager that wraps a block in a LangSmith trace.

    Usage:
        with trace_eval_run("test_orchestrator_intent", {"query": "..."}) as run:
            result = orchestrator_agent(state)
            run["output"] = result
    """
    run_data = {"output": None, "elapsed": 0.0}
    start = time.perf_counter()

    try:
        yield run_data
    finally:
        run_data["elapsed"] = time.perf_counter() - start

        if LANGSMITH_AVAILABLE:
            try:
                client = Client()
                client.create_run(
                    name=test_name,
                    run_type="eval",
                    inputs=metadata or {},
                    outputs={"result": str(run_data.get("output", ""))[:500]},
                    extra={"elapsed_seconds": run_data["elapsed"]},
                    project_name="prototype2-evals",
                )
            except Exception:
                pass  # Don't let tracing failures break tests


def traced_agent_call(agent_name: str):
    """Decorator that traces an agent call to LangSmith.

    Usage:
        @traced_agent_call("orchestrator")
        def test_something():
            ...
    """
    def decorator(func):
        if not LANGSMITH_AVAILABLE:
            return func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with trace_eval_run(f"eval/{agent_name}/{func.__name__}"):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def log_eval_summary(results: dict):
    """Log a summary of evaluation results to LangSmith as a dataset run.

    Args:
        results: Dict with test names as keys and pass/fail as values.
    """
    if not LANGSMITH_AVAILABLE:
        print("LangSmith not configured — skipping summary upload.")
        return

    try:
        client = Client()
        passed = sum(1 for v in results.values() if v)
        total = len(results)

        client.create_run(
            name="eval_summary",
            run_type="eval",
            inputs={"total_tests": total},
            outputs={
                "passed": passed,
                "failed": total - passed,
                "pass_rate": f"{passed / total:.1%}" if total else "N/A",
                "details": {k: "PASS" if v else "FAIL" for k, v in results.items()},
            },
            project_name="prototype2-evals",
        )
        print(f"Eval summary uploaded to LangSmith: {passed}/{total} passed")
    except Exception as e:
        print(f"Failed to upload eval summary: {e}")
