"""
LangSmith Evaluation — Job Match Quality

Evaluates whether LLM-assigned match scores correlate with actual
user feedback (interested / applied = positive; not_interested = negative).

Ground truth: JobMatch.user_feedback rows in the DB.
No new LLM call in the target — we evaluate the stored scores directly.

Evaluators:
  precision_at_75  — did a score ≥ 75 correctly predict a positive reaction?
  score_alignment  — LLM-as-judge: does the recommendation text match user's reaction?

Quality gates:
  precision_at_75 ≥ 0.60
  score_alignment ≥ 0.65

Run:
  python evals/job_match_eval.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_xai import ChatXAI
from langsmith import Client
from langsmith.evaluation import evaluate

from evals.datasets import create_or_get_dataset

DATASET_NAME = "job-match-quality-eval-v1"
SCORE_THRESHOLD = 75.0

QUALITY_THRESHOLDS = {
    "precision_at_75": 0.60,
    "score_alignment":  0.65,
}


# ── Dataset — pulled live from the DB ─────────────────────────────────────────

def _pull_examples_from_db():
    """Query rated JobMatch rows and format them as LangSmith dataset examples."""
    from factory import create_app
    from models import JobMatch

    app = create_app()
    with app.app_context():
        rated = (
            JobMatch.query
            .filter(JobMatch.user_feedback.in_(['interested', 'applied', 'not_interested']))
            .filter(JobMatch.match_score.isnot(None))
            .limit(100)
            .all()
        )

        if not rated:
            return [], []

        inputs, outputs = [], []
        for m in rated:
            if not m.job:
                continue
            is_positive = m.user_feedback in ('interested', 'applied')
            inputs.append({
                "job_title":      m.job.title,
                "company":        m.job.company or "Unknown",
                "match_score":    m.match_score,
                "recommendation": m.recommendation or "",
            })
            outputs.append({
                "user_feedback": m.user_feedback,
                "is_positive":   is_positive,
            })

        return inputs, outputs


# ── Target — passthrough, no new LLM call ─────────────────────────────────────

def job_match_target(inputs: dict) -> dict:
    """
    No LLM call needed — we evaluate scores that are already stored in the DB.
    The target simply passes through the stored prediction for evaluators to inspect.
    """
    return {
        "match_score":    inputs["match_score"],
        "recommendation": inputs.get("recommendation", ""),
    }


# ── Evaluators ─────────────────────────────────────────────────────────────────

def make_evaluators(xai_api_key: str):
    judge = ChatXAI(model="grok-3", temperature=0, api_key=xai_api_key)

    def precision_at_75(run, example) -> dict:
        """
        PREDICTION ACCURACY: did a high match score correctly predict the user liked it?

        True positive  = score ≥ 75 AND user was interested → score 1
        True negative  = score <  75 AND user was not interested → score 1
        False positive = score ≥ 75 AND user was not interested → score 0
        False negative = score <  75 AND user was interested → score 0
        """
        match_score  = run.outputs.get("match_score", 0)
        is_positive  = example.outputs.get("is_positive", False)
        predicted_positive = match_score >= SCORE_THRESHOLD
        correct = predicted_positive == is_positive
        comment = (
            f"score={match_score:.1f}, "
            f"predicted={'positive' if predicted_positive else 'negative'}, "
            f"actual={'positive' if is_positive else 'negative'}"
        )
        return {"key": "precision_at_75", "score": 1 if correct else 0, "comment": comment}

    def score_alignment(run, example) -> dict:
        """
        RECOMMENDATION ALIGNMENT: does the recommendation text match the user's reaction?
        Catches cases where the score is right but the written recommendation is off-tone.
        """
        recommendation = run.outputs.get("recommendation", "")
        if not recommendation:
            return {"key": "score_alignment", "score": None, "comment": "no recommendation text"}

        actual_feedback = example.outputs.get("user_feedback", "")
        job_title       = example.inputs.get("job_title", "?")

        prompt = f"""You are evaluating a job matching system.

Job title: {job_title}
Match recommendation written by the system: {recommendation[:400]}
User's actual feedback after seeing the job: {actual_feedback}

Does the recommendation's tone and content align with the user's reaction?
- A positive recommendation (highlighting strong fit) should align with 'interested' or 'applied'.
- A cautious or negative recommendation (highlighting gaps) should align with 'not_interested'.

Think step by step, then end your response with exactly one of:
ALIGNED
MISALIGNED"""

        result  = judge.invoke(prompt)
        verdict = result.content.strip().upper()
        score   = 1 if ("ALIGNED" in verdict and "MISALIGNED" not in verdict) else 0
        return {"key": "score_alignment", "score": score, "comment": result.content.strip()[:200]}

    return [precision_at_75, score_alignment]


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

    print("Pulling rated job matches from database...")
    inputs, outputs = _pull_examples_from_db()

    positive_count = sum(1 for o in outputs if o.get("is_positive"))
    negative_count = len(outputs) - positive_count

    if len(inputs) < 5:
        print(
            f"WARNING: Only {len(inputs)} rated matches found (need ≥ 5 for a meaningful eval).\n"
            f"Rate some job matches as 'interested' or 'not interested' in the app first."
        )
        if not inputs:
            sys.exit(0)

    print(f"Found {len(inputs)} rated matches ({positive_count} positive, {negative_count} negative).")

    client      = Client()
    dataset_name = create_or_get_dataset(
        client, DATASET_NAME,
        description=(
            "Job match quality eval — compares LLM-assigned match scores against "
            "actual user feedback (interested / not_interested)."
        ),
        inputs=inputs,
        outputs=outputs,
    )

    evaluators = make_evaluators(os.environ["XAI_API_KEY"])

    print(f"\nRunning evaluators over {len(inputs)} job matches...")
    results = evaluate(
        job_match_target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix="job-match-quality",
        metadata={"model": "grok-3", "score_threshold": SCORE_THRESHOLD},
    )

    df = results.to_pandas()

    # ── Per-match summary ──────────────────────────────────────────────────────
    print("\n── Per-match Results ────────────────────────────────────────────")
    prec_col  = next((c for c in df.columns if "precision" in c.lower()), None)
    align_col = next((c for c in df.columns if "score_alignment" in c.lower()), None)

    import pandas as pd

    for _, row in df.iterrows():
        title    = row.get("inputs.job_title", "?")
        score    = row.get("inputs.match_score", 0)
        feedback = row.get("outputs.user_feedback", "?")
        prec     = f"{int(row[prec_col])}"   if prec_col  and pd.notna(row[prec_col])  else "?"
        align    = f"{int(row[align_col])}"  if align_col and pd.notna(row[align_col]) else "?"
        print(f"  precision={prec}  alignment={align}  score={score:.0f}  "
              f"feedback={feedback:<15}  {title[:45]}")

    # ── Aggregate ──────────────────────────────────────────────────────────────
    print("\n── Aggregate Scores ─────────────────────────────────────────────")
    for col, label in [
        (prec_col,  "Precision@75   (score≥75 → user liked?)      "),
        (align_col, "Score Alignment (rec. text matches reaction?) "),
    ]:
        if col:
            valid = df[col].dropna()
            print(f"  {label}: {valid.mean():.2f}  (over {len(valid)} matches)")

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
