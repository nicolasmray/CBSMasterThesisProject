#!/usr/bin/env python3
"""Evaluation runner script.

Every run automatically:
  1. Saves full terminal output to evals/reports/last_run.txt
  2. Generates a JSON report at evals/reports/eval_report.json
  3. Generates a human-readable summary at evals/reports/summary.txt

Usage:
    python evals/run_evals.py              # Run everything
    python evals/run_evals.py --sql        # SQL agent tests only
    python evals/run_evals.py --rag        # RAG tests only
    python evals/run_evals.py --e2e        # End-to-end tests only
    python evals/run_evals.py --perf       # Performance benchmarks
    python evals/run_evals.py --fast       # Skip slow/LLM tests (no tokens used)
    python evals/run_evals.py -v           # Verbose (show each test name)
"""

import argparse
import json
import os
import sys
import subprocess
from datetime import datetime

from dotenv import load_dotenv

EVALS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(EVALS_DIR, "..")
REPORTS_DIR = os.path.join(EVALS_DIR, "reports")
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


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
    lines.append(f"  Duration:      {duration:.1f}s ({duration/60:.1f}m)")
    lines.append("")

    # Group by file
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

            # Status indicator
            if outcome == "PASSED":
                icon = "PASS"
            elif outcome == "FAILED":
                icon = "FAIL"
            elif outcome == "SKIPPED":
                icon = "SKIP"
            else:
                icon = outcome

            lines.append(f"  [{icon}] {test_name:<65s} ({dur:.1f}s)")

            # Show failure reason
            if outcome == "FAILED":
                call_info = t.get("call", {})
                crash = call_info.get("crash", {})
                message = crash.get("message", "")
                if not message:
                    longrepr = call_info.get("longrepr", "")
                    if isinstance(longrepr, str):
                        message = longrepr[:200]

                if message:
                    # Categorize the failure
                    msg_lower = message.lower()
                    if "429" in msg_lower or "rate_limit" in msg_lower or "rate limit" in msg_lower:
                        reason = "RATE LIMITED (Groq 429 — token quota exhausted)"
                    elif "assert" in msg_lower:
                        reason = message[:200]
                    elif "attribute" in msg_lower:
                        reason = f"CODE ERROR: {message[:150]}"
                    else:
                        reason = message[:200]
                    lines.append(f"         Reason: {reason}")

        lines.append("")

    # Summary of failure categories
    rate_limited = sum(1 for t in tests
                       if t.get("outcome") == "failed"
                       and ("429" in str(t.get("call", {}).get("crash", {}).get("message", "")).lower()
                            or "rate_limit" in str(t.get("call", {}).get("crash", {}).get("message", "")).lower()))
    real_failures = failed - rate_limited

    lines.append("=" * 78)
    lines.append("  FAILURE BREAKDOWN")
    lines.append("=" * 78)
    lines.append(f"  Rate limit errors (429):  {rate_limited}  (not real failures — retry with fresh quota)")
    lines.append(f"  Real test failures:       {real_failures}")
    lines.append(f"  Skipped:                  {skipped}")
    lines.append("")

    if real_failures > 0:
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
    lines.append(f"  Reports saved to: {REPORTS_DIR}/")
    lines.append(f"    summary.txt      — this file")
    lines.append(f"    eval_report.json — full JSON (parseable)")
    lines.append(f"    last_run.txt     — raw terminal output")
    lines.append("=" * 78)

    text = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(text)

    # Also print the summary to stdout
    print("\n" + text)


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

    # Ensure reports directory exists
    os.makedirs(REPORTS_DIR, exist_ok=True)

    json_report_path = os.path.join(REPORTS_DIR, "eval_report.json")
    summary_path = os.path.join(REPORTS_DIR, "summary.txt")
    log_path = os.path.join(REPORTS_DIR, "last_run.txt")

    cmd = [
        sys.executable, "-m", "pytest",
        EVALS_DIR,
        "-c", os.path.join(EVALS_DIR, "pytest.ini"),
        "--tb=short",
        "-v" if args.verbose else "-q",
        "--json-report",
        f"--json-report-file={json_report_path}",
    ]

    # Marker filters
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

    # Print header
    header = (
        "=" * 70 + "\n"
        "  PROTOTYPE 2 — EVALUATION SUITE\n"
        "=" * 70 + "\n"
        f"  Started:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  Working dir: {PROJECT_ROOT}\n"
        f"  Reports:     {REPORTS_DIR}/\n"
        f"  LangSmith:   {'enabled' if os.getenv('LANGSMITH_API_KEY') else 'disabled'}\n"
        f"  Groq keys:   {len(os.getenv('GROQ_API_KEYS', '').split(','))} in rotation pool\n"
        "=" * 70
    )
    print(header)

    # Run pytest, capture output to file AND show on terminal
    with open(log_path, "w") as log_file:
        log_file.write(header + "\n\n")
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

    # Generate human-readable summary from JSON report
    if os.path.exists(json_report_path):
        _generate_summary(json_report_path, summary_path)
        print(f"\nReports saved to {REPORTS_DIR}/")
    else:
        print("\nWarning: JSON report not generated — summary unavailable.")

    sys.exit(process.returncode)


if __name__ == "__main__":
    main()
