"""
Eval Orchestrator — AI Career Coach

Runs all five evals in sequence and prints a combined pass/fail report.
Exits with code 1 if any eval fails its quality gates — safe to wire into CI.

Run:
  python evals/run_all.py

Individual evals can still be run directly, e.g.:
  python evals/job_match_eval.py
  python evals/chat_eval.py
"""

import os
import subprocess
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ── Eval registry ──────────────────────────────────────────────────────────────
# (script_path relative to project root, display_label)
# Each eval runs in its own subprocess — avoids RAGAS circular import when
# multiple modules are loaded in the same process.

EVALS = [
    ("evals/job_match_eval.py",             "Job Match Quality     "),
    ("evals/tailoring_eval.py",             "Resume Tailoring      "),
    ("evals/chat_eval.py",                  "Chat Response Quality "),
    ("evals/memory_eval.py",                "Memory Quality        "),
    ("resume_eval_check/eval_resume_qa.py", "Resume Q&A (RAG)      "),
]


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_eval(script_path: str, label: str) -> bool:
    """
    Run an eval script as a fresh subprocess.
    Each eval gets its own Python process — no shared import state,
    no RAGAS circular-import issues between evals.
    Returns True if the script exited with code 0 (all gates passed).
    """
    full_path = os.path.join(PROJECT_ROOT, script_path)
    if not os.path.exists(full_path):
        print(f"  ERROR: Script not found: {full_path}")
        return False

    result = subprocess.run(
        [sys.executable, full_path],
        cwd=PROJECT_ROOT,
    )
    return result.returncode == 0


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'=' * 62}")
    print(f"  AI Career Coach — Full Eval Suite")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'=' * 62}\n")

    if not os.environ.get("LANGSMITH_API_KEY"):
        print("ERROR: LANGSMITH_API_KEY not set in .env")
        sys.exit(1)
    if not os.environ.get("XAI_API_KEY"):
        print("ERROR: XAI_API_KEY not set in .env")
        sys.exit(1)

    results: dict[str, bool] = {}

    for script_path, label in EVALS:
        print(f"\n{'─' * 62}")
        print(f"  Running: {label.strip()}")
        print(f"{'─' * 62}")
        passed                 = run_eval(script_path, label)
        results[label.strip()] = passed

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print(f"  FINAL RESULTS")
    print(f"{'=' * 62}")

    all_passed = True
    for label, passed in results.items():
        icon = "PASS" if passed else "FAIL"
        print(f"  [{icon}]  {label}")
        if not passed:
            all_passed = False

    project = os.environ.get("LANGSMITH_PROJECT", "default")
    print(f"\nAll traces: https://smith.langchain.com  (project: {project})")

    if not all_passed:
        print("\nOne or more evals failed — see output above for details.")
        sys.exit(1)
    else:
        print("\nAll evals passed.")


if __name__ == "__main__":
    main()
