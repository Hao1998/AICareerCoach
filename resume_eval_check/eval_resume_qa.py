"""
LangSmith Evaluation — Resume Q&A RAG Pipeline

Evaluates two things separately:
  1. Retrieval quality  — did the right chunks come back? (RAGAS ContextRecall)
  2. Answer correctness — is the final answer right?     (LLM-as-judge)

Run (from project root, with .venv active):
  python resume_eval_check/eval_resume_qa.py

Results: https://smith.langchain.com  → project testProjectAICareerCoach
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_xai import ChatXAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langsmith import Client
from langsmith.evaluation import evaluate
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall, Faithfulness

from services.resume_service import extract_text_from_pdf


RESUME_PDF_PATH = os.path.join(os.path.dirname(__file__), "LyHienHao_Resume.pdf")
DATASET_NAME = "hao-resume-qa-eval-v2"

# ── Dataset ───────────────────────────────────────────────────────────────────
#
# Factual cases: answerable from the resume.
#   Evaluated with: context_recall + faithfulness + correctness
#
# Adversarial cases: NOT answerable from the resume.
#   Evaluated with: faithfulness only.
#   The LLM should say "I don't know" rather than invent facts.
#   context_recall and correctness are skipped (no meaningful reference exists).

QA_EXAMPLES = [
    # ── Factual ───────────────────────────────────────────────────────────────
    {
        "question": "What is the candidate's primary programming language?",
        "answer": "Java",
    },
    {
        "question": "How many years of experience does the candidate have?",
        "answer": "4+ years of enterprise experience",
    },
    {
        "question": "What is the candidate's current job title and company?",
        "answer": "Software Developer — GenAI & Enterprise Automation at Zoi / Kärcher",
    },
    {
        "question": "What GenAI tools and frameworks does the candidate know?",
        "answer": (
            "LangChain, LangChain Agents, RAG Pipelines, FAISS, Pinecone, "
            "Google Gemini API, OpenAI Embeddings, Prompt Engineering, "
            "Semantic Search, Hallucination Prevention"
        ),
    },
    {
        "question": "How many AWS certifications does the candidate hold?",
        "answer": "5 AWS certifications",
    },
    {
        "question": "What was the impact of the GenAI ticket summarization system at Zoi/Kärcher?",
        "answer": "Reduced agent ticket-reading time by approximately 30%",
    },
    {
        "question": "What was the event processing throughput the candidate handled at Edulog?",
        "answer": "5,000 to 6,000 events per second via Kafka-based event streaming",
    },
    {
        "question": "What university did the candidate attend and what did they study?",
        "answer": "B.Sc. Information Technology at University of Pedagogy, Ho Chi Minh City",
    },
    {
        "question": "What is the candidate's IELTS score?",
        "answer": "IELTS 7.0",
    },
    {
        "question": "What languages does the candidate speak professionally?",
        "answer": "English and French",
    },
    {
        "question": "What AWS services has the candidate worked with?",
        "answer": "Lambda, SageMaker, EC2, ECS, S3, RDS, SQS, SNS, API Gateway, IAM",
    },
    {
        "question": "Where is the candidate located and are they open to remote work?",
        "answer": "Ho Chi Minh City, Vietnam — open to remote and freelance contract",
    },
    # ── Adversarial (NOT in resume) ───────────────────────────────────────────
    # A faithfulness score of 1.0 here means the LLM did NOT invent an answer.
    {
        "question": "What is the candidate's expected salary?",
        "answer": "NOT_IN_RESUME",
        "adversarial": True,
    },
    {
        "question": "What does the candidate think about Python versus Java?",
        "answer": "NOT_IN_RESUME",
        "adversarial": True,
    },
    {
        "question": "What is the candidate's LinkedIn profile URL?",
        "answer": "NOT_IN_RESUME",
        "adversarial": True,
    },
    {
        "question": "Has the candidate ever worked at Google or Amazon?",
        "answer": "NOT_IN_RESUME",
        "adversarial": True,
    },
]

FACTUAL_EXAMPLES     = [ex for ex in QA_EXAMPLES if not ex.get("adversarial")]
ADVERSARIAL_EXAMPLES = [ex for ex in QA_EXAMPLES if ex.get("adversarial")]


# ── Step 1: Chain that returns source documents ───────────────────────────────

def _build_qa_chain():
    print(f"Loading resume from: {RESUME_PDF_PATH}")
    resume_text = extract_text_from_pdf(RESUME_PDF_PATH)

    embeddings = HuggingFaceEmbeddings()
    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=2000, chunk_overlap=200, length_function=len
    )
    docs = splitter.create_documents([resume_text])
    vector_db = FAISS.from_documents(docs, embeddings)
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = ChatXAI(model="grok-3", temperature=0, api_key=os.environ["XAI_API_KEY"])
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,  # ← was False — we now keep retrieved chunks
    )


_qa_chain = None


def resume_qa_target(inputs: dict) -> dict:
    """
    LangSmith target function.
    Input:  {"question": "..."}
    Output: {"output": "...", "retrieved_contexts": ["chunk1", "chunk2", ...]}
    """
    global _qa_chain
    if _qa_chain is None:
        _qa_chain = _build_qa_chain()

    result = _qa_chain.invoke(inputs["question"])
    return {
        "output": result["result"],
        "retrieved_contexts": [doc.page_content for doc in result["source_documents"]],
    }


# ── Dataset management ────────────────────────────────────────────────────────

def create_or_get_dataset(client: Client) -> str:
    existing = [d for d in client.list_datasets() if d.name == DATASET_NAME]
    if existing:
        print(f"Reusing existing dataset: '{DATASET_NAME}'")
        return DATASET_NAME

    print(f"Creating dataset '{DATASET_NAME}' with {len(QA_EXAMPLES)} examples...")
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "Resume Q&A eval — factual questions + adversarial (out-of-corpus) questions. "
            "Tests retrieval recall and hallucination resistance."
        ),
    )
    client.create_examples(
        inputs=[{"question": ex["question"]} for ex in QA_EXAMPLES],
        outputs=[
            {"answer": ex["answer"], "adversarial": ex.get("adversarial", False)}
            for ex in QA_EXAMPLES
        ],
        dataset_id=dataset.id,
    )
    print(f"Dataset created: {len(FACTUAL_EXAMPLES)} factual + {len(ADVERSARIAL_EXAMPLES)} adversarial.")
    return DATASET_NAME


# ── Step 3: Evaluators ────────────────────────────────────────────────────────

def make_evaluators(xai_api_key: str):
    """
    Returns three evaluators.

    context_recall  — RAGAS. Did the retriever return chunks that contain the
                      facts needed to answer the question?
                      Score 0.0–1.0. Skipped for adversarial cases.

    faithfulness    — RAGAS. Did the LLM only use what was in the retrieved
                      chunks, or did it add facts from its own memory?
                      Score 0.0–1.0. Run on ALL cases, including adversarial.

    correctness     — LLM-as-judge. Is the final answer factually right?
                      Score 0 or 1. Skipped for adversarial cases.
    """
    # XAI (Grok) is OpenAI-compatible — point ragas at the XAI base URL.
    # We use AsyncOpenAI so ragas can run its async scoring calls.
    async_client = AsyncOpenAI(
        api_key=xai_api_key,
        base_url="https://api.x.ai/v1",
    )
    # max_tokens=8192: RAGAS faithfulness sends all statement verdicts in one structured
    # JSON call. Verbose answers (10+ tools, long descriptions) produce 15-20 statements
    # whose verdict JSON exceeds the instructor default of 1024 tokens, causing
    # IncompleteOutputException on every retry. 8192 is enough for any realistic answer.
    ragas_llm = llm_factory("grok-3", provider="openai", client=async_client, max_tokens=8192)

    recall_metric      = ContextRecall(llm=ragas_llm)
    faithfulness_metric = Faithfulness(llm=ragas_llm)

    # Reuse one event loop across all evaluator calls in this process.
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    def _run(coro):
        return loop.run_until_complete(coro)

    def context_recall(run, example) -> dict:
        """
        RETRIEVAL CHECK: Did the right chunks come back?

        This is the metric the original eval was missing entirely.
        If this score is low but correctness is high, the LLM is answering
        from its own training data — not from the resume. Silent risk in prod.
        """
        if example.outputs.get("adversarial"):
            return {"key": "context_recall", "score": None, "comment": "skipped (adversarial)"}

        result = _run(recall_metric.ascore(
            user_input=example.inputs["question"],
            retrieved_contexts=run.outputs.get("retrieved_contexts", []),
            reference=example.outputs["answer"],
        ))
        return {"key": "context_recall", "score": float(result)}

    def faithfulness(run, example) -> dict:
        """
        HALLUCINATION CHECK: Did the LLM stick to the retrieved chunks?

        For adversarial cases: if the resume has no answer, the LLM should
        say so. A high faithfulness score here = LLM correctly abstained.
        """
        result = _run(faithfulness_metric.ascore(
            user_input=example.inputs["question"],
            response=run.outputs.get("output", ""),
            retrieved_contexts=run.outputs.get("retrieved_contexts", []),
        ))
        label = " (adversarial)" if example.outputs.get("adversarial") else ""
        return {"key": "faithfulness", "score": float(result), "comment": label.strip()}

    def correctness(run, example) -> dict:
        """
        END-TO-END CHECK: Is the answer correct?
        This is what the original eval measured alone. Now it sits alongside
        context_recall and faithfulness so you can see all three together.
        """
        if example.outputs.get("adversarial"):
            return {"key": "correctness", "score": None, "comment": "skipped (adversarial)"}

        judge = ChatXAI(model="grok-3", temperature=0, api_key=xai_api_key)

        prompt = f"""You are evaluating a RAG system that answers questions about a resume.

Question: {example.inputs["question"]}
Reference answer: {example.outputs["answer"]}
RAG system answer: {run.outputs.get("output", "")}

Does the RAG system's answer convey the same core information as the reference?
It does NOT need to be word-for-word identical.

Think step by step, then end with exactly one of:
CORRECT
INCORRECT"""

        result = judge.invoke(prompt)
        verdict = result.content.strip().upper()
        score = 1 if verdict.endswith("CORRECT") and not verdict.endswith("INCORRECT") else 0
        return {"key": "correctness", "score": score, "comment": result.content.strip()}

    return [context_recall, faithfulness, correctness]


# ── Step 4: Quality gates ─────────────────────────────────────────────────────
#
# Minimum acceptable scores. If any metric falls below its threshold the script
# exits with code 1 — which fails a CI/CD pipeline job automatically.
#
# How to read these:
#   context_recall 0.75 → retriever must find the right chunks 75% of the time
#   faithfulness   0.85 → LLM must not hallucinate more than 15% of claims
#   correctness    0.75 → end-to-end answers must be correct 75% of the time

QUALITY_THRESHOLDS = {
    "context_recall": 0.75,
    "faithfulness":   0.85,
    "correctness":    0.75,
}


def check_quality_gates(df) -> bool:
    print("\n── Quality Gates ────────────────────────────────────────────────")
    all_passed = True

    for metric, threshold in QUALITY_THRESHOLDS.items():
        col = next((c for c in df.columns if metric in c.lower()), None)
        if col is None:
            print(f"  ?  {metric}: not found in results (skipped)")
            continue

        valid_scores = df[col].dropna()
        if valid_scores.empty:
            print(f"  ?  {metric}: no valid scores")
            continue

        actual = valid_scores.mean()
        passed = actual >= threshold
        icon   = "PASS" if passed else "FAIL"
        print(f"  [{icon}]  {metric}: {actual:.2f}  (min: {threshold:.2f})")
        if not passed:
            all_passed = False

    return all_passed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("ERROR: LANGSMITH_API_KEY not set in .env")
        sys.exit(1)
    if not os.environ.get("XAI_API_KEY"):
        print("ERROR: XAI_API_KEY not set in .env")
        sys.exit(1)
    if not os.path.exists(RESUME_PDF_PATH):
        print(f"ERROR: Resume PDF not found at {RESUME_PDF_PATH}")
        sys.exit(1)

    xai_api_key = os.environ["XAI_API_KEY"]

    client     = Client()
    dataset_name = create_or_get_dataset(client)
    evaluators   = make_evaluators(xai_api_key)

    print(f"\nRunning {len(QA_EXAMPLES)} questions through RAG chain "
          f"({len(FACTUAL_EXAMPLES)} factual, {len(ADVERSARIAL_EXAMPLES)} adversarial)...")

    results = evaluate(
        resume_qa_target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix="hao-resume-qa",
        metadata={
            "resume":        "LyHienHao_Resume.pdf",
            "model":         "grok-3",
            "chunk_size":    2000,
            "chunk_overlap": 200,
            "retrieval_k":   4,
            "eval_version":  "v2-retrieval-aware",
        },
    )

    df = results.to_pandas()

    # ── Per-question summary ──────────────────────────────────────────────────
    print("\n── Per-question Results ─────────────────────────────────────────")

    recall_col   = next((c for c in df.columns if "context_recall" in c.lower()), None)
    faithful_col = next((c for c in df.columns if "faithfulness"   in c.lower()), None)
    correct_col  = next((c for c in df.columns if "correctness"    in c.lower()), None)

    import pandas as pd

    for _, row in df.iterrows():
        question = row.get("inputs.question", "?")
        recall   = f"{row[recall_col]:.2f}"   if recall_col   and pd.notna(row[recall_col])   else "skip"
        faithful = f"{row[faithful_col]:.2f}" if faithful_col and pd.notna(row[faithful_col]) else "skip"
        correct  = f"{int(row[correct_col])}" if correct_col  and pd.notna(row[correct_col])  else "skip"

        # Flag: recall is low but correctness is high — LLM used memorized knowledge
        warning = ""
        if (recall_col and pd.notna(row[recall_col]) and row[recall_col] < 0.5
                and correct_col and pd.notna(row[correct_col]) and row[correct_col] == 1):
            warning = "  <- LLM may be using memorized knowledge"

        print(f"  recall={recall}  faithful={faithful}  correct={correct}  {question[:60]}{warning}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print("\n── Aggregate Scores ─────────────────────────────────────────────")
    for col_name, label in [
        (recall_col,   "Context Recall  (retrieval quality)      "),
        (faithful_col, "Faithfulness    (hallucination resistance)"),
        (correct_col,  "Correctness     (end-to-end accuracy)     "),
    ]:
        if col_name:
            valid = df[col_name].dropna()
            print(f"  {label}: {valid.mean():.2f}  (over {len(valid)} questions)")

    # ── Step 4: Quality gates ─────────────────────────────────────────────────
    passed = check_quality_gates(df)

    project = os.environ.get("LANGSMITH_PROJECT", "default")
    print(f"\nFull details: https://smith.langchain.com  (project: {project})")

    if not passed:
        print("\nOne or more quality gates failed — see [FAIL] lines above.")
        sys.exit(1)
    else:
        print("\nAll quality gates passed.")


if __name__ == "__main__":
    main()
