"""
LangSmith Evaluation — Resume Tailoring Quality

Evaluates whether ATS tailoring results actually improve the resume.

Ground truth: JobMatch.tailoring_result (stored JSON from past tailoring calls).
No new LLM tailoring call — we evaluate the results that are already stored.

Evaluators:
  ats_improvement       — does ats_after > ats_before for each result?
  keyword_coverage      — LLM-as-judge: are the missing keywords present in tailored sections?
  summary_faithfulness  — RAGAS: is the tailored summary grounded in the original resume?

Quality gates:
  ats_improvement      ≥ 0.70  (70% of tailorings must show a score increase)
  keyword_coverage     ≥ 0.75
  summary_faithfulness ≥ 0.80

Run:
  python evals/tailoring_eval.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_xai import ChatXAI
from langsmith import Client
from langsmith.evaluation import evaluate
from ragas.metrics.collections import Faithfulness

from evals.datasets import create_or_get_dataset, get_ragas_llm, get_event_loop

DATASET_NAME = "tailoring-quality-eval-v1"

QUALITY_THRESHOLDS = {
    "ats_improvement":      0.70,
    "keyword_coverage":     0.75,
    "summary_faithfulness": 0.80,
}


# ── Dataset — pulled live from the DB ─────────────────────────────────────────

def _pull_examples_from_db():
    """Query stored tailoring results from the DB and format them as examples."""
    from factory import create_app
    from models import JobMatch, Resume

    app = create_app()
    with app.app_context():
        matches = (
            JobMatch.query
            .filter(JobMatch.tailoring_result.isnot(None))
            .limit(50)
            .all()
        )

        inputs, outputs = [], []
        for m in matches:
            try:
                tailoring = json.loads(m.tailoring_result)
            except (json.JSONDecodeError, TypeError):
                continue

            ats        = tailoring.get("ats_score", {})
            kw_analysis = tailoring.get("keyword_analysis", {})
            sections   = tailoring.get("tailored_sections", {})

            ats_before = ats.get("before")
            ats_after  = ats.get("after")
            if ats_before is None or ats_after is None:
                continue

            missing_keywords = kw_analysis.get("missing_from_resume", [])
            tailored_summary = sections.get("professional_summary", "")

            # Load resume text for faithfulness check
            resume_snippet = ""
            if m.resume_id:
                resume = Resume.query.get(m.resume_id)
                if resume and resume.text_content:
                    resume_snippet = resume.text_content[:1500]

            inputs.append({
                "job_title":        m.job.title if m.job else "Unknown",
                "ats_before":       ats_before,
                "ats_after":        ats_after,
                "missing_keywords": missing_keywords,
                "tailored_summary": tailored_summary,
                "resume_snippet":   resume_snippet,
            })
            outputs.append({})  # no separate ground-truth label — self-evaluated

        return inputs, outputs


# ── Target — passthrough, no new LLM call ─────────────────────────────────────

def tailoring_target(inputs: dict) -> dict:
    """
    Passthrough — we evaluate stored tailoring results.
    The target makes the stored fields available to evaluators via run.outputs.
    """
    return {
        "ats_before":       inputs["ats_before"],
        "ats_after":        inputs["ats_after"],
        "missing_keywords": inputs["missing_keywords"],
        "tailored_summary": inputs["tailored_summary"],
        "resume_snippet":   inputs.get("resume_snippet", ""),
    }


# ── Evaluators ─────────────────────────────────────────────────────────────────

def make_evaluators(xai_api_key: str):
    judge              = ChatXAI(model="grok-3", temperature=0, api_key=xai_api_key)
    faithfulness_metric = Faithfulness(llm=get_ragas_llm())
    loop               = get_event_loop()

    def ats_improvement(run, example) -> dict:
        """SCORE IMPROVEMENT: did the ATS score go up after tailoring?"""
        before   = run.outputs.get("ats_before", 0)
        after    = run.outputs.get("ats_after",  0)
        improved = after > before
        return {
            "key":     "ats_improvement",
            "score":   1 if improved else 0,
            "comment": f"before={before}, after={after}, delta={after - before:+d}",
        }

    def keyword_coverage(run, example) -> dict:
        """
        KEYWORD INSERTION: are the missing keywords present in the tailored summary?
        Uses LLM-as-judge since keyword matching needs synonym awareness.
        """
        missing = run.outputs.get("missing_keywords", [])
        summary = run.outputs.get("tailored_summary", "")
        if not missing or not summary:
            return {"key": "keyword_coverage", "score": None, "comment": "missing data — skipped"}

        keywords_str = ", ".join(missing[:10])
        prompt = f"""A resume tailoring system was given these keywords to insert into the resume:
Missing keywords: {keywords_str}

This is the tailored Professional Summary it produced:
{summary[:800]}

What fraction of the missing keywords (or their close synonyms / rewordings) appear in the tailored summary?
Score 0.0 = none present, 1.0 = all present.

Think step by step, then end your response with exactly: SCORE: <float between 0 and 1>"""

        result = judge.invoke(prompt)
        content = result.content.strip()
        try:
            score_str = content.split("SCORE:")[-1].strip().split()[0]
            score = max(0.0, min(1.0, float(score_str)))
        except (ValueError, IndexError):
            score = 0.0
        return {"key": "keyword_coverage", "score": score, "comment": content[:200]}

    def summary_faithfulness(run, example) -> dict:
        """
        HALLUCINATION CHECK: is the tailored summary grounded in the original resume?
        The tailoring must reframe existing experience — not invent new ones.
        """
        summary = run.outputs.get("tailored_summary", "")
        resume  = run.outputs.get("resume_snippet",   "")
        if not summary or not resume:
            return {"key": "summary_faithfulness", "score": None, "comment": "missing data — skipped"}

        result = loop.run_until_complete(faithfulness_metric.ascore(
            user_input="Rewrite the professional summary using the job's keywords.",
            response=summary,
            retrieved_contexts=[resume],
        ))
        return {"key": "summary_faithfulness", "score": float(result)}

    return [ats_improvement, keyword_coverage, summary_faithfulness]


# ── Quality gates ──────────────────────────────────────────────────────────────

def check_quality_gates(df) -> bool:
    print("\n── Quality Gates ────────────────────────────────────────────────")
    all_passed = True
    for metric, threshold in QUALITY_THRESHOLDS.items():
        col = next((c for c in df.columns if metric in c.lower()), None)
        if col is None:
            print(f"  ?  {metric}: column not found in results (skipped)")
            continue
        valid = df[col].dropna()
        if valid.empty:
            print(f"  ?  {metric}: no valid scores")
            continue
        actual = valid.mean()
        passed = actual >= threshold
        icon   = "PASS" if passed else "FAIL"
        print(f"  [{icon}]  {metric}: {actual:.2f}  (min: {threshold:.2f})")
        if not passed:
            all_passed = False
    return all_passed


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("ERROR: LANGSMITH_API_KEY not set in .env")
        sys.exit(1)
    if not os.environ.get("XAI_API_KEY"):
        print("ERROR: XAI_API_KEY not set in .env")
        sys.exit(1)

    print("Pulling stored tailoring results from database...")
    inputs, outputs = _pull_examples_from_db()

    if len(inputs) < 3:
        print(
            f"WARNING: Only {len(inputs)} tailoring results found.\n"
            f"Use the 'Tailor Resume' feature in the chat to generate more data."
        )
        if not inputs:
            sys.exit(0)

    print(f"Found {len(inputs)} tailoring results to evaluate.")

    client       = Client()
    dataset_name = create_or_get_dataset(
        client, DATASET_NAME,
        description=(
            "Resume tailoring quality eval — ATS score improvement, "
            "keyword coverage, and summary hallucination check."
        ),
        inputs=inputs,
        outputs=outputs,
    )

    evaluators = make_evaluators(os.environ["XAI_API_KEY"])

    print(f"\nRunning evaluators over {len(inputs)} tailoring results...")
    results = evaluate(
        tailoring_target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix="tailoring-quality",
        metadata={"model": "grok-3", "eval_version": "v1"},
    )

    df = results.to_pandas()

    # ── Per-result summary ─────────────────────────────────────────────────────
    print("\n── Per-result Summary ───────────────────────────────────────────")
    ats_col   = next((c for c in df.columns if "ats_improvement"      in c.lower()), None)
    kw_col    = next((c for c in df.columns if "keyword_coverage"     in c.lower()), None)
    faith_col = next((c for c in df.columns if "summary_faithfulness" in c.lower()), None)

    import pandas as pd

    for _, row in df.iterrows():
        title  = row.get("inputs.job_title",  "?")
        before = row.get("inputs.ats_before", "?")
        after  = row.get("inputs.ats_after",  "?")
        ats    = f"{int(row[ats_col])}"       if ats_col   and pd.notna(row[ats_col])   else "?"
        kw     = f"{row[kw_col]:.2f}"         if kw_col    and pd.notna(row[kw_col])    else "?"
        faith  = f"{row[faith_col]:.2f}"      if faith_col and pd.notna(row[faith_col]) else "?"
        print(f"  ats={ats}  kw={kw}  faithful={faith}  {before}→{after}  {title[:45]}")

    # ── Aggregate ──────────────────────────────────────────────────────────────
    print("\n── Aggregate Scores ─────────────────────────────────────────────")
    for col, label in [
        (ats_col,   "ATS Improvement      (after > before?)          "),
        (kw_col,    "Keyword Coverage     (missing kws in summary?)  "),
        (faith_col, "Summary Faithfulness (no hallucinated content?)  "),
    ]:
        if col:
            valid = df[col].dropna()
            print(f"  {label}: {valid.mean():.2f}  (over {len(valid)} results)")

    passed  = check_quality_gates(df)
    project = os.environ.get("LANGSMITH_PROJECT", "default")
    print(f"\nFull details: https://smith.langchain.com  (project: {project})")

    if not passed:
        print("\nOne or more quality gates failed.")
        sys.exit(1)
    else:
        print("\nAll quality gates passed.")


if __name__ == "__main__":
    main()
