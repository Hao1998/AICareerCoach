"""
LangSmith Evaluation — Memory Quality (Session Summarisation + Fact Extraction)

Evaluates whether the session summariser and fact extractor produce
accurate, faithful, non-hallucinated output.

Uses 4 synthetic conversations with known expected facts.
No DB access needed.

Evaluators:
  summary_faithfulness — RAGAS: does the summary only contain things actually said?
  fact_recall          — LLM-as-judge: are the key expected facts captured?
  no_hallucination     — LLM-as-judge: does the summary invent anything not in the conversation?

Quality gates:
  summary_faithfulness ≥ 0.85
  fact_recall          ≥ 0.70
  no_hallucination     ≥ 0.80

Run:
  python evals/memory_eval.py
"""

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

DATASET_NAME = "memory-quality-eval-v1"

# ── Synthetic dataset ──────────────────────────────────────────────────────────
# Each entry is a fake conversation + the facts that SHOULD be extracted.
# The facts are the ground truth for the fact_recall evaluator.

MEMORY_EXAMPLES = [
    {
        "conversation": [
            ("user",      "I've been working as a Python backend developer for 6 years."),
            ("assistant", "That's solid experience. What kinds of projects have you focused on?"),
            ("user",      "Mostly fintech APIs and payment systems. I also lead a team of 5 engineers."),
            ("assistant", "Great leadership experience. Are you looking to stay in fintech?"),
            ("user",      "Actually I want to move into AI/ML engineering. Python is still relevant so that's good."),
            ("assistant", "Makes sense. We can look for ML engineering roles that leverage your backend skills."),
        ],
        "expected_facts": [
            "User has 6 years of Python backend experience.",
            "User works in fintech, specifically on APIs and payment systems.",
            "User leads a team of 5 engineers.",
            "User wants to transition into AI/ML engineering.",
        ],
    },
    {
        "conversation": [
            ("user",      "I only want remote jobs. I'm based in Berlin but the timezone works fine for US companies."),
            ("assistant", "Got it — fully remote, Berlin timezone. Any preference on company size?"),
            ("user",      "I prefer startups, 50 to 200 people. I don't want big tech — too slow and bureaucratic."),
            ("assistant", "Noted. I'll focus on remote-friendly startups in that range."),
            ("user",      "Also, I had a really bad experience with micromanagement at my last company. That's a dealbreaker for me."),
            ("assistant", "Understood. I'll flag that in your preferences and avoid companies known for that culture."),
        ],
        "expected_facts": [
            "User requires fully remote work.",
            "User is based in Berlin.",
            "User prefers startups with 50 to 200 employees.",
            "User does not want to work at big tech companies.",
            "Micromanagement is a dealbreaker for the user.",
        ],
    },
    {
        "conversation": [
            ("user",      "I have an interview at Stripe next week for a senior backend role."),
            ("assistant", "Exciting! Stripe is known for rigorous system design interviews. Want to prep?"),
            ("user",      "Yes please. My biggest weakness is distributed systems — I haven't done much of that."),
            ("assistant", "Let's focus on CAP theorem, consistency models, and Stripe's payment infrastructure patterns."),
            ("user",      "I'm actually quite good on algorithms — I've done about 150 LeetCode problems."),
            ("assistant", "Great — then we'll spend most of our time on system design and distributed concepts."),
        ],
        "expected_facts": [
            "User has an upcoming interview at Stripe for a senior backend role.",
            "User's weakness is distributed systems.",
            "User has completed approximately 150 LeetCode problems.",
            "User is strong on algorithms.",
        ],
    },
    {
        "conversation": [
            ("user",      "My current salary is $95k and I'm targeting at least $130k in my next role."),
            ("assistant", "That's a reasonable jump for a senior position. Are you open to equity as part of the comp?"),
            ("user",      "Yes, RSUs are fine but I won't accept a base below $120k even with equity."),
            ("assistant", "Understood — $120k base minimum, RSUs acceptable on top of that."),
            ("user",      "Exactly. And I'd prefer options over RSUs if I'm joining a pre-IPO startup."),
            ("assistant", "Good distinction. I'll keep that in mind when evaluating startup offers."),
        ],
        "expected_facts": [
            "User's current salary is $95k.",
            "User's target salary is at least $130k.",
            "User requires a minimum base salary of $120k even with equity.",
            "User is open to RSUs as part of compensation.",
            "User prefers options over RSUs for pre-IPO startups.",
        ],
    },
]

QUALITY_THRESHOLDS = {
    "summary_faithfulness": 0.85,
    "fact_recall":          0.70,
    "no_hallucination":     0.80,
}


# ── Mock ChatMessage ───────────────────────────────────────────────────────────

class _FakeMessage:
    """Minimal stand-in for ChatMessage — summarize_session only needs .role and .content."""
    def __init__(self, role: str, content: str):
        self.role    = role
        self.content = content


# ── Target ─────────────────────────────────────────────────────────────────────

def memory_target(inputs: dict) -> dict:
    """
    Run summarize_session and extract_memory_facts on the synthetic conversation.
    Creates ChatXAI directly — no Flask app context needed.
    """
    from langchain_xai import ChatXAI as _ChatXAI
    from chatbot.memory import summarize_session, extract_memory_facts

    messages = [_FakeMessage(role, content) for role, content in inputs["conversation"]]
    llm      = _ChatXAI(model="grok-3", temperature=0, api_key=os.environ["XAI_API_KEY"])

    summary = summarize_session(messages, llm) or ""
    facts   = extract_memory_facts(messages, llm)

    # Build a plain-text version of the conversation for RAGAS faithfulness checks
    conversation_text = "\n".join(
        f"{'User' if role == 'user' else 'Assistant'}: {content}"
        for role, content in inputs["conversation"]
    )

    return {
        "summary":           summary,
        "facts":             facts,
        "conversation_text": conversation_text,
    }


# ── Evaluators ─────────────────────────────────────────────────────────────────

def make_evaluators(xai_api_key: str):
    judge               = ChatXAI(model="grok-3", temperature=0, api_key=xai_api_key)
    faithfulness_metric = Faithfulness(llm=get_ragas_llm())
    loop                = get_event_loop()

    def summary_faithfulness(run, example) -> dict:
        """
        FAITHFULNESS: is every claim in the summary something that was actually said?
        A score of 1.0 means the summary contains no invented content.
        """
        summary      = run.outputs.get("summary", "")
        conversation = run.outputs.get("conversation_text", "")
        if not summary or not conversation:
            return {"key": "summary_faithfulness", "score": None, "comment": "missing data — skipped"}

        result = loop.run_until_complete(faithfulness_metric.ascore(
            user_input="Summarise the career coaching conversation.",
            response=summary,
            retrieved_contexts=[conversation],
        ))
        return {"key": "summary_faithfulness", "score": float(result)}

    def fact_recall(run, example) -> dict:
        """
        RECALL: what fraction of the expected facts were captured in the extracted list?
        Uses LLM-as-judge since facts may be paraphrased rather than verbatim.
        """
        facts    = run.outputs.get("facts", [])
        expected = example.outputs.get("expected_facts", [])
        if not expected:
            return {"key": "fact_recall", "score": None, "comment": "no expected facts defined"}

        facts_str    = "\n".join(f"- {f}" for f in facts) if facts else "(none extracted)"
        expected_str = "\n".join(f"- {f}" for f in expected)

        prompt = f"""You are evaluating a memory extraction system.

Facts that SHOULD have been extracted from the conversation:
{expected_str}

Facts that WERE actually extracted:
{facts_str}

What fraction of the expected facts are captured (exact or as a close paraphrase)?
Score 0.0 = none captured, 1.0 = all captured.

Think step by step, then end your response with exactly: SCORE: <float between 0 and 1>"""

        result  = judge.invoke(prompt)
        content = result.content.strip()
        try:
            score_str = content.split("SCORE:")[-1].strip().split()[0]
            score     = max(0.0, min(1.0, float(score_str)))
        except (ValueError, IndexError):
            score = 0.0
        return {"key": "fact_recall", "score": score, "comment": content[:200]}

    def no_hallucination(run, example) -> dict:
        """
        HALLUCINATION CHECK: does the summary claim anything NOT in the conversation?
        Score 1.0 = summary only states what was actually said.
        Score 0.0 = summary invents facts, numbers, or statements.
        """
        summary      = run.outputs.get("summary", "")
        conversation = run.outputs.get("conversation_text", "")
        if not summary or not conversation:
            return {"key": "no_hallucination", "score": None, "comment": "missing data — skipped"}

        prompt = f"""You are auditing a conversation summary for hallucinated content.

Original conversation:
{conversation}

Generated summary:
{summary}

Does the summary contain ANY facts, numbers, or claims that are NOT in the original conversation?
Score 1.0 = the summary contains nothing invented (only restates what was said).
Score 0.0 = the summary clearly invents facts not present in the conversation.

Think step by step, then end your response with exactly: SCORE: <float between 0 and 1>"""

        result  = judge.invoke(prompt)
        content = result.content.strip()
        try:
            score_str = content.split("SCORE:")[-1].strip().split()[0]
            score     = max(0.0, min(1.0, float(score_str)))
        except (ValueError, IndexError):
            score = 0.5
        return {"key": "no_hallucination", "score": score, "comment": content[:200]}

    return [summary_faithfulness, fact_recall, no_hallucination]


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
            "Memory quality eval — session summary faithfulness, fact recall, "
            "and hallucination check over 4 synthetic conversations."
        ),
        inputs=[{"conversation": ex["conversation"]} for ex in MEMORY_EXAMPLES],
        outputs=[{"expected_facts": ex["expected_facts"]} for ex in MEMORY_EXAMPLES],
    )

    evaluators = make_evaluators(os.environ["XAI_API_KEY"])

    print(f"\nRunning memory eval over {len(MEMORY_EXAMPLES)} synthetic conversations...")
    results = evaluate(
        memory_target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix="memory-quality",
        metadata={"model": "grok-3", "eval_version": "v1"},
    )

    df = results.to_pandas()

    import pandas as pd

    # ── Per-conversation summary ───────────────────────────────────────────────
    print("\n── Per-conversation Results ─────────────────────────────────────")
    faith_col  = next((c for c in df.columns if "summary_faithfulness" in c.lower()), None)
    recall_col = next((c for c in df.columns if "fact_recall"          in c.lower()), None)
    halluc_col = next((c for c in df.columns if "no_hallucination"     in c.lower()), None)

    for i, row in df.iterrows():
        # Pull first user message as a label for the row
        first_msg = (
            MEMORY_EXAMPLES[i]["conversation"][0][1]
            if i < len(MEMORY_EXAMPLES) else "?"
        )
        faith  = f"{row[faith_col]:.2f}"  if faith_col  and pd.notna(row[faith_col])  else "?"
        recall = f"{row[recall_col]:.2f}" if recall_col and pd.notna(row[recall_col]) else "?"
        halluc = f"{row[halluc_col]:.2f}" if halluc_col and pd.notna(row[halluc_col]) else "?"
        print(f"  faithful={faith}  recall={recall}  no_halluc={halluc}  \"{first_msg[:50]}...\"")

    # ── Aggregate ──────────────────────────────────────────────────────────────
    print("\n── Aggregate Scores ─────────────────────────────────────────────")
    for col, label in [
        (faith_col,  "Summary Faithfulness (grounded in conversation?) "),
        (recall_col, "Fact Recall          (key facts captured?)        "),
        (halluc_col, "No Hallucination     (nothing invented?)           "),
    ]:
        if col:
            valid = df[col].dropna()
            print(f"  {label}: {valid.mean():.2f}  (over {len(valid)} conversations)")

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
