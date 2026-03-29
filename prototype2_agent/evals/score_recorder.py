"""Score recorder — captures DeepEval metric scores and writes them to disk.

Every DeepEval test calls `record_and_assert()` instead of `assert_test()`.
This measures the metric, records the score to a shared JSON file, and then
asserts. The run_evals.py script reads this file to build the scoreboard.

Scores are written to evals/reports/_scores_current.json during the run,
then the runner moves them into the history directory with a timestamp.
"""

import json
import os
import threading
from datetime import datetime

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
SCORES_FILE = os.path.join(REPORTS_DIR, "_scores_current.json")

_lock = threading.Lock()


def _ensure_dir():
    os.makedirs(REPORTS_DIR, exist_ok=True)


def _load_scores() -> list[dict]:
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE) as f:
            return json.load(f)
    return []


def _save_scores(scores: list[dict]):
    _ensure_dir()
    with open(SCORES_FILE, "w") as f:
        json.dump(scores, f, indent=2)


def record_score(
    test_name: str,
    metric_name: str,
    score: float,
    threshold: float,
    passed: bool,
    reason: str = "",
):
    """Append a metric score to the current run's score file."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "test_name": test_name,
        "metric_name": metric_name,
        "score": round(score, 4),
        "threshold": threshold,
        "passed": passed,
        "reason": reason[:300],
    }
    with _lock:
        scores = _load_scores()
        scores.append(entry)
        _save_scores(scores)


def record_and_assert(test_case, metrics: list, test_name: str = ""):
    """Measure each metric, record the score, then assert.

    Replaces deepeval.assert_test() — does the same thing but captures scores.

    Args:
        test_case: DeepEval LLMTestCase
        metrics: List of DeepEval metric instances
        test_name: Human-readable test name for the scoreboard
    """
    if not test_name:
        test_name = test_case.input[:80]

    all_passed = True
    failure_reasons = []

    for metric in metrics:
        metric.measure(test_case)
        score = metric.score
        passed = score >= metric.threshold
        reason = getattr(metric, "reason", "") or ""

        record_score(
            test_name=test_name,
            metric_name=metric.__class__.__name__,
            score=score,
            threshold=metric.threshold,
            passed=passed,
            reason=reason,
        )

        if not passed:
            all_passed = False
            failure_reasons.append(
                f"{metric.__class__.__name__}: score={score:.4f}, "
                f"threshold={metric.threshold}, reason={reason[:150]}"
            )

    if not all_passed:
        raise AssertionError(
            f"Metrics failed:\n" + "\n".join(failure_reasons)
        )


def clear_current_scores():
    """Delete the current scores file (called at the start of each run)."""
    if os.path.exists(SCORES_FILE):
        os.remove(SCORES_FILE)
