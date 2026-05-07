"""
LangSmith Evaluation — Chat Response Quality

Evaluates whether the career coaching LLM gives responses that are:
  1. Grounded in the resume context (not hallucinated)
  2. Actually relevant to the user's question

Uses a synthetic dataset — 12 questions against a fixed fake resume context.
No DB access needed; the LLM is called directly (no full agent stack).

Evaluators:
  faithfulness — RAGAS: is the response grounded in the resume context provided?
  relevance    — LLM-as-judge: does the response actually answer the question?

Quality gates:
  faithfulness ≥ 0.80
  relevance    ≥ 0.75

Run:
  python evals/chat_eval.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_xai import ChatXAI
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import Client
from langsmith.evaluation import evaluate
from ragas.metrics.collections import Faithfulness

from evals.datasets import create_or_get_dataset, get_ragas_llm, get_event_loop

DATASET_NAME = "chat-response-quality-eval-v1"

# ── Synthetic dataset ──────────────────────────────────────────────────────────
# A fixed fake resume is shared across all questions so evaluators can check
# faithfulness against a known ground truth.

FAKE_RESUME_CONTEXT = """
Name: Alex Johnson
Experience:
  - Senior Software Engineer at DataCorp (2021–Present):
    Built Python microservices handling 10k req/s. Led a team of 4 engineers.
    Introduced CI/CD with GitHub Actions, reducing deploy time by 40%.
  - Backend Developer at StartupXYZ (2019–2021):
    Developed REST APIs with Django and PostgreSQL.
    Migrated legacy monolith to microservices architecture.
Skills: Python, Django, PostgreSQL, Docker, AWS (EC2, S3, Lambda), REST APIs, Git, GitHub Actions
Education: B.Sc. Computer Science, State University (2019)
Certifications: AWS Solutions Architect Associate
Languages: English (native)
""".strip()

CHAT_EXAMPLES = [
    # ── Grounded (answer must come from resume) ────────────────────────────────
    {
        "question": "What programming languages do I know?",
        "expected_topics": "Python — should not hallucinate Go, Java, or other languages",
    },
    {
        "question": "How many years of experience do I have?",
        "expected_topics": "approximately 5-6 years across DataCorp and StartupXYZ",
    },
    {
        "question": "What cloud platforms have I worked with?",
        "expected_topics": "AWS specifically EC2, S3, Lambda — no Azure or GCP",
    },
    {
        "question": "What is my strongest skill for a backend engineering role?",
        "expected_topics": "Python, microservices, REST APIs, Django",
    },
    {
        "question": "Do I have any team leadership experience?",
        "expected_topics": "led a team of 4 at DataCorp",
    },
    {
        "question": "What certifications do I hold?",
        "expected_topics": "AWS Solutions Architect Associate — only one certification",
    },
    {
        "question": "What measurable impact have I had in my career?",
        "expected_topics": "10k req/s, 40% deploy time reduction — should cite these numbers",
    },
    {
        "question": "Can you summarise my career trajectory in two sentences?",
        "expected_topics": "progression from backend dev to senior engineer, Python/AWS focus",
    },
    {
        "question": "What skill gaps do I have if I want a data engineering role?",
        "expected_topics": (
            "should mention Spark, Kafka, Airflow, or data pipelines — "
            "skills NOT in the resume, so must reason about gaps"
        ),
    },
    # ── Boundary (should NOT hallucinate) ─────────────────────────────────────
    # These questions have no answer in the resume. The model must say so clearly.
    {
        "question": "What is my expected salary?",
        "expected_topics": "should say salary is not in the resume — must not invent a number",
    },
    {
        "question": "Do I have any mobile development experience?",
        "expected_topics": "should say no or not found — must not invent mobile experience",
    },
    {
        "question": "What companies have I interviewed at previously?",
        "expected_topics": "should say not known — must not invent company names",
    },
]

SYSTEM_PROMPT = f"""You are Career Coach AI, a helpful career coaching assistant.
The user's resume is provided below. Only answer based on information in the resume.
If something is not in the resume, say clearly that you don't have that information — do not invent it.

<untrusted_data source="resume">
{FAKE_RESUME_CONTEXT}
</untrusted_data>"""

QUALITY_THRESHOLDS = {
    "faithfulness": 0.80,
    "relevance":    0.75,
}


# ── Target ─────────────────────────────────────────────────────────────────────

def chat_target(inputs: dict) -> dict:
    """
    Call Grok-3 with the career coach system prompt and the question.
    Returns the LLM response plus the resume context (used by RAGAS faithfulness).
    """
    llm = ChatXAI(
        model="grok-3",
        temperature=0,
        api_key=os.environ["XAI_API_KEY"],
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=inputs["question"]),
    ]
    result = llm.invoke(messages)
    return {
        "output":             result.content,
        "retrieved_contexts": [FAKE_RESUME_CONTEXT],
    }


# ── Evaluators ─────────────────────────────────────────────────────────────────

def make_evaluators(xai_api_key: str):
    judge               = ChatXAI(model="grok-3", temperature=0, api_key=xai_api_key)
    faithfulness_metric = Faithfulness(llm=get_ragas_llm())
    loop                = get_event_loop()

    def faithfulness(run, example) -> dict:
        """
        HALLUCINATION CHECK: is the response grounded in the resume?
        High score = only said things that are in the resume context.
        Low score  = invented facts not in the resume.
        """
        result = loop.run_until_complete(faithfulness_metric.ascore(
            user_input=example.inputs["question"],
            response=run.outputs.get("output", ""),
            retrieved_contexts=run.outputs.get("retrieved_contexts", []),
        ))
        return {"key": "faithfulness", "score": float(result)}

    def relevance(run, example) -> dict:
        """
        ANSWER QUALITY: does the response actually address the question?
        Also checks that boundary cases (no answer in resume) are handled correctly.
        """
        expected_topics = example.outputs.get("expected_topics", "")
        response        = run.outputs.get("output", "")

        prompt = f"""You are evaluating a career coaching AI assistant.

User question: {example.inputs["question"]}
What a good answer should cover: {expected_topics}
AI response: {response[:600]}

Score from 0.0 to 1.0:
  1.0 = response fully addresses the question and covers the expected topics
  0.5 = partial answer or misses some expected topics
  0.0 = off-topic, refuses to answer when it should, or hallucinates when it should decline

Think step by step, then end your response with exactly: SCORE: <float between 0 and 1>"""

        result  = judge.invoke(prompt)
        content = result.content.strip()
        try:
            score_str = content.split("SCORE:")[-1].strip().split()[0]
            score     = max(0.0, min(1.0, float(score_str)))
        except (ValueError, IndexError):
            score = 0.0
        return {"key": "relevance", "score": score, "comment": content[:200]}

    return [faithfulness, relevance]


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

    client       = Client()
    dataset_name = create_or_get_dataset(
        client, DATASET_NAME,
        description=(
            "Chat response quality eval — faithfulness (grounded in resume?) "
            "and relevance (answers the question?) over 12 synthetic examples."
        ),
        inputs=[{"question": ex["question"]} for ex in CHAT_EXAMPLES],
        outputs=[{"expected_topics": ex["expected_topics"]} for ex in CHAT_EXAMPLES],
    )

    evaluators = make_evaluators(os.environ["XAI_API_KEY"])

    grounded_count  = sum(1 for ex in CHAT_EXAMPLES if "must not" not in ex["expected_topics"])
    boundary_count  = len(CHAT_EXAMPLES) - grounded_count
    print(f"\nRunning {len(CHAT_EXAMPLES)} questions through chat LLM "
          f"({grounded_count} grounded, {boundary_count} boundary/hallucination checks)...")

    results = evaluate(
        chat_target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix="chat-quality",
        metadata={"model": "grok-3", "eval_version": "v1"},
    )

    df = results.to_pandas()

    # ── Per-question summary ───────────────────────────────────────────────────
    print("\n── Per-question Results ─────────────────────────────────────────")
    faith_col = next((c for c in df.columns if "faithfulness" in c.lower()), None)
    rel_col   = next((c for c in df.columns if "relevance"    in c.lower()), None)

    import pandas as pd

    for _, row in df.iterrows():
        question = row.get("inputs.question", "?")
        faith    = f"{row[faith_col]:.2f}" if faith_col and pd.notna(row[faith_col]) else "?"
        rel      = f"{row[rel_col]:.2f}"   if rel_col   and pd.notna(row[rel_col])   else "?"

        # Flag suspiciously low faithfulness — possible hallucination
        flag = "  <- possible hallucination" if faith_col and pd.notna(row[faith_col]) and row[faith_col] < 0.5 else ""
        print(f"  faithful={faith}  relevant={rel}  {question[:55]}{flag}")

    # ── Aggregate ──────────────────────────────────────────────────────────────
    print("\n── Aggregate Scores ─────────────────────────────────────────────")
    for col, label in [
        (faith_col, "Faithfulness  (grounded in resume?)     "),
        (rel_col,   "Relevance     (answers the question?)   "),
    ]:
        if col:
            valid = df[col].dropna()
            print(f"  {label}: {valid.mean():.2f}  (over {len(valid)} questions)")

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
