#!/usr/bin/env python3
"""Evaluation runner script.

Every run automatically:
  1. Saves results to a timestamped folder in evals/reports/history/
  2. Generates a scoreboard with actual DeepEval metric scores
  3. Maintains a run log comparing current vs previous runs
  4. Never overwrites previous results

Reports structure:
  evals/reports/
  ├── latest/                    ← always points to the most recent run
  │   ├── summary.txt
  │   ├── scoreboard.txt
  │   ├── eval_report.json
  │   └── scores.json
  ├── history/
  │   ├── 2026-03-29_17-30-00/   ← timestamped, never deleted
  │   │   ├── summary.txt
  │   │   ├── scoreboard.txt
  │   │   ├── eval_report.json
  │   │   └── scores.json
  │   └── 2026-03-29_18-00-00/
  │       └── ...
  └── run_history.txt            ← one-line-per-run comparison table

Usage:
    python evals/run_evals.py              # Run everything
    python evals/run_evals.py --sql        # SQL agent tests only
    python evals/run_evals.py --rag        # RAG tests only
    python evals/run_evals.py --e2e        # End-to-end tests only
    python evals/run_evals.py --perf       # Performance benchmarks
    python evals/run_evals.py --fast       # Skip slow/LLM tests
    python evals/run_evals.py -v           # Verbose output
"""

import argparse
import json
import os
import shutil
import sys
import subprocess
from datetime import datetime

from dotenv import load_dotenv

EVALS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(EVALS_DIR, "..")
REPORTS_DIR = os.path.join(EVALS_DIR, "reports")
HISTORY_DIR = os.path.join(REPORTS_DIR, "history")
LATEST_DIR = os.path.join(REPORTS_DIR, "latest")
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def _build_scoreboard(scores_path: str, output_path: str):
    """Build a scoreboard showing every DeepEval metric with its actual score."""
    if not os.path.exists(scores_path):
        with open(output_path, "w") as f:
            f.write("No DeepEval metrics were recorded in this run.\n")
        return

    with open(scores_path) as f:
        scores = json.load(f)

    if not scores:
        with open(output_path, "w") as f:
            f.write("No DeepEval metrics were recorded in this run.\n")
        return

    lines = []
    lines.append("=" * 90)
    lines.append("  DEEPEVAL SCOREBOARD")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"  {'Test Name':<45s} {'Metric':<25s} {'Score':>7s} {'Thresh':>7s} {'Result':>7s}")
    lines.append(f"  {'─' * 45} {'─' * 25} {'─' * 7} {'─' * 7} {'─' * 7}")

    for s in scores:
        name = s["test_name"][:45]
        metric = s["metric_name"][:25]
        score = f"{s['score']:.4f}"
        thresh = f"{s['threshold']:.2f}"
        result = "PASS" if s["passed"] else "FAIL"
        lines.append(f"  {name:<45s} {metric:<25s} {score:>7s} {thresh:>7s} {result:>7s}")

    # Summary stats
    lines.append("")
    lines.append("─" * 90)
    total = len(scores)
    passed = sum(1 for s in scores if s["passed"])
    failed = total - passed

    # Average scores by metric type
    by_metric: dict[str, list[float]] = {}
    for s in scores:
        by_metric.setdefault(s["metric_name"], []).append(s["score"])

    lines.append(f"  Total metrics evaluated: {total}  |  Passed: {passed}  |  Failed: {failed}")
    lines.append("")
    lines.append("  Average scores by metric type:")
    for metric_name, metric_scores in sorted(by_metric.items()):
        avg = sum(metric_scores) / len(metric_scores)
        lines.append(f"    {metric_name:<30s}  avg={avg:.4f}  (n={len(metric_scores)})")

    lines.append("")

    # Show failures with reasons
    failures = [s for s in scores if not s["passed"]]
    if failures:
        lines.append("  FAILED METRICS (with reasons):")
        for s in failures:
            lines.append(f"    {s['test_name']}")
            lines.append(f"      {s['metric_name']}: {s['score']:.4f} < {s['threshold']:.2f}")
            if s.get("reason"):
                lines.append(f"      Reason: {s['reason'][:200]}")
        lines.append("")

    lines.append("=" * 90)

    text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(text)


def _generate_summary(json_path: str, output_path: str):
    """Read the JSON report and write a human-readable summary."""
    with open(json_path) as f:
        data = json.load(f)

    summary = data.get("summary", {})
    tests = data.get("tests", [])
    duration = data.get("duration", 0)

    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    skipped = summary.get("skipped", 0)
    errors = summary.get("error", 0)
    total = summary.get("total", passed + failed + skipped + errors)
    pass_rate = (passed / total * 100) if total else 0

    lines = []
    lines.append("=" * 78)
    lines.append(f"  EVALUATION REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"  Total tests:   {total}")
    lines.append(f"  Passed:        {passed}")
    lines.append(f"  Failed:        {failed}")
    lines.append(f"  Skipped:       {skipped}")
    lines.append(f"  Errors:        {errors}")
    lines.append(f"  Pass rate:     {pass_rate:.1f}%")
    lines.append(f"  Duration:      {duration:.1f}s ({duration / 60:.1f}m)")
    lines.append("")

    by_file: dict[str, list] = {}
    for t in tests:
        nodeid = t.get("nodeid", "")
        filename = nodeid.split("::")[0] if "::" in nodeid else nodeid
        by_file.setdefault(filename, []).append(t)

    for filename, file_tests in sorted(by_file.items()):
        file_passed = sum(1 for t in file_tests if t.get("outcome") == "passed")
        file_total = len(file_tests)
        lines.append("-" * 78)
        lines.append(f"  {filename}  ({file_passed}/{file_total} passed)")
        lines.append("-" * 78)

        for t in file_tests:
            nodeid = t.get("nodeid", "")
            test_name = nodeid.split("::")[-1] if "::" in nodeid else nodeid
            outcome = t.get("outcome", "?").upper()
            dur = t.get("duration", 0)

            icon = {"PASSED": "PASS", "FAILED": "FAIL", "SKIPPED": "SKIP"}.get(outcome, outcome)
            lines.append(f"  [{icon}] {test_name:<65s} ({dur:.1f}s)")

            if outcome == "FAILED":
                crash = t.get("call", {}).get("crash", {})
                message = crash.get("message", "")
                if message:
                    msg_lower = message.lower()
                    if "429" in msg_lower or "rate_limit" in msg_lower:
                        reason = "RATE LIMITED (Groq 429)"
                    else:
                        reason = message[:200]
                    lines.append(f"         Reason: {reason}")
        lines.append("")

    rate_limited = sum(1 for t in tests
                       if t.get("outcome") == "failed"
                       and ("429" in str(t.get("call", {}).get("crash", {}).get("message", "")).lower()
                            or "rate_limit" in str(t.get("call", {}).get("crash", {}).get("message", "")).lower()))
    real_failures = failed - rate_limited

    lines.append("=" * 78)
    lines.append("  FAILURE BREAKDOWN")
    lines.append("=" * 78)
    lines.append(f"  Rate limit errors (429):  {rate_limited}")
    lines.append(f"  Real test failures:       {real_failures}")
    lines.append(f"  Skipped:                  {skipped}")

    if real_failures > 0:
        lines.append("")
        lines.append("  REAL FAILURES:")
        for t in tests:
            if t.get("outcome") == "failed":
                crash_msg = str(t.get("call", {}).get("crash", {}).get("message", "")).lower()
                if "429" not in crash_msg and "rate_limit" not in crash_msg:
                    nodeid = t.get("nodeid", "")
                    test_name = nodeid.split("::")[-1] if "::" in nodeid else nodeid
                    msg = t.get("call", {}).get("crash", {}).get("message", "")[:150]
                    lines.append(f"    - {test_name}")
                    lines.append(f"      {msg}")

    lines.append("")
    lines.append("=" * 78)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def _append_run_history(run_dir: str, json_path: str, scores_path: str):
    """Append a one-line summary to the run history log."""
    history_file = os.path.join(REPORTS_DIR, "run_history.txt")

    # Read test results
    with open(json_path) as f:
        data = json.load(f)
    s = data.get("summary", {})
    duration = data.get("duration", 0)

    total = s.get("total", 0)
    passed = s.get("passed", 0)
    failed = s.get("failed", 0)
    skipped = s.get("skipped", 0)
    rate = f"{passed / total * 100:.0f}%" if total else "N/A"

    # Read DeepEval scores
    avg_score = "N/A"
    if os.path.exists(scores_path):
        with open(scores_path) as f:
            scores = json.load(f)
        if scores:
            avg_score = f"{sum(sc['score'] for sc in scores) / len(scores):.3f}"

    timestamp = os.path.basename(run_dir)
    line = (f"  {timestamp}  |  {total:>4d} tests  |  {passed:>3d} pass  |  "
            f"{failed:>3d} fail  |  {skipped:>3d} skip  |  {rate:>5s}  |  "
            f"avg_metric={avg_score:>6s}  |  {duration:>6.0f}s")

    # Write header if file doesn't exist
    if not os.path.exists(history_file):
        header = (
            "=" * 110 + "\n"
            "  RUN HISTORY — Each line is one evaluation run. Compare pass rates and metric scores over time.\n"
            "=" * 110 + "\n"
            f"  {'Run Timestamp':<22s}  |  {'Tests':>10s}  |  {'Pass':>8s}  |  "
            f"{'Fail':>8s}  |  {'Skip':>8s}  |  {'Rate':>5s}  |  "
            f"{'Avg Metric':>16s}  |  {'Duration':>9s}\n"
            + "-" * 110 + "\n"
        )
        with open(history_file, "w") as f:
            f.write(header)

    with open(history_file, "a") as f:
        f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run prototype2 evaluation suite")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--sql", action="store_true", help="Run only SQL agent tests")
    parser.add_argument("--rag", action="store_true", help="Run only RAG tests")
    parser.add_argument("--perf", action="store_true", help="Run only performance tests")
    parser.add_argument("--e2e", action="store_true", help="Run only end-to-end tests")
    parser.add_argument("--fast", action="store_true", help="Skip slow and LLM tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Create timestamped run directory
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(HISTORY_DIR, run_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(LATEST_DIR, exist_ok=True)

    json_report_path = os.path.join(run_dir, "eval_report.json")
    scores_src = os.path.join(REPORTS_DIR, "_scores_current.json")

    # Clear scores from previous run
    if os.path.exists(scores_src):
        os.remove(scores_src)

    cmd = [
        sys.executable, "-m", "pytest",
        EVALS_DIR,
        "-c", os.path.join(EVALS_DIR, "pytest.ini"),
        "--tb=short",
        "-v" if args.verbose else "-q",
        "--json-report",
        f"--json-report-file={json_report_path}",
    ]

    markers = []
    if args.unit:
        markers.append("unit")
    if args.sql:
        markers.append("sql")
    if args.rag:
        markers.append("rag")
    if args.perf:
        markers.append("perf")
    if args.e2e:
        markers.append("e2e")
    if args.fast:
        markers.append("not slow and not llm")

    if markers:
        cmd.extend(["-m", " or ".join(markers)])

    # Print header once
    groq_keys = len([k for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()])
    header = (
        "\n" + "=" * 70 + "\n"
        "  PROTOTYPE 2 — EVALUATION SUITE\n"
        + "=" * 70 + "\n"
        f"  Started:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  Run ID:      {run_timestamp}\n"
        f"  Reports:     {run_dir}/\n"
        f"  LangSmith:   {'enabled' if os.getenv('LANGSMITH_API_KEY') else 'disabled'}\n"
        f"  Groq keys:   {groq_keys} in rotation pool\n"
        + "=" * 70 + "\n"
    )
    print(header)

    # Run pytest — capture to log file AND show on terminal
    log_path = os.path.join(run_dir, "terminal_output.txt")
    with open(log_path, "w") as log_file:
        log_file.write(header + "\n")
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
        process.wait()

    # Move scores file into run directory
    scores_dst = os.path.join(run_dir, "scores.json")
    if os.path.exists(scores_src):
        shutil.move(scores_src, scores_dst)

    # Generate reports
    if os.path.exists(json_report_path):
        summary_path = os.path.join(run_dir, "summary.txt")
        scoreboard_path = os.path.join(run_dir, "scoreboard.txt")

        _generate_summary(json_report_path, summary_path)
        _build_scoreboard(scores_dst, scoreboard_path)
        _append_run_history(run_dir, json_report_path, scores_dst)

        # Copy to latest/
        for fname in ["summary.txt", "scoreboard.txt", "eval_report.json", "scores.json"]:
            src = os.path.join(run_dir, fname)
            dst = os.path.join(LATEST_DIR, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)

        # Print scoreboard
        if os.path.exists(scoreboard_path):
            with open(scoreboard_path) as f:
                print("\n" + f.read())

        # Print summary
        with open(summary_path) as f:
            print(f.read())

        print(f"\nRun saved to: {run_dir}/")
        print(f"Latest copy:  {LATEST_DIR}/")
        print(f"Run history:  {os.path.join(REPORTS_DIR, 'run_history.txt')}")
    else:
        print("\nWarning: JSON report not generated.")

    sys.exit(process.returncode)


if __name__ == "__main__":
    main()
