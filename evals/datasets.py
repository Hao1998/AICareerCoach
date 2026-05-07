"""
Eval Shared Helpers

Shared utilities used by all eval modules:
  - LangSmith dataset create-or-get
  - RAGAS-compatible async LLM factory (XAI Grok via OpenAI-compat endpoint)
  - Reusable asyncio event loop
"""

import asyncio
import os

from openai import AsyncOpenAI
from ragas.llms import llm_factory


def get_ragas_llm():
    """Return an async RAGAS-compatible LLM using XAI Grok-3 (OpenAI-compat endpoint).

    max_tokens=8192: RAGAS faithfulness decomposes the LLM response into individual
    statements and asks the model to produce a JSON verdict for ALL of them in one
    structured-output call (via instructor). Verbose answers can produce 15-20 statements
    whose verdict JSON needs 4000+ tokens to complete. The instructor default of 1024
    hits the limit and retries with 2048/3072 — all still too small. 8192 is enough
    headroom for any realistic resume QA or chat response.
    """
    async_client = AsyncOpenAI(
        api_key=os.environ["XAI_API_KEY"],
        base_url="https://api.x.ai/v1",
    )
    return llm_factory("grok-3", provider="openai", client=async_client, max_tokens=8192)


def get_event_loop():
    """Return a reusable event loop for running async RAGAS evaluators synchronously."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def create_or_get_dataset(client, name: str, description: str,
                           inputs: list, outputs: list) -> str:
    """
    Create a LangSmith dataset if it doesn't exist yet; return its name.
    If the dataset already exists it is reused — no duplicate examples are added.
    """
    existing = [d for d in client.list_datasets() if d.name == name]
    if existing:
        print(f"Reusing existing dataset: '{name}'")
        return name

    print(f"Creating dataset '{name}' with {len(inputs)} examples...")
    dataset = client.create_dataset(dataset_name=name, description=description)
    client.create_examples(
        inputs=inputs,
        outputs=outputs,
        dataset_id=dataset.id,
    )
    print(f"Dataset '{name}' created ({len(inputs)} examples).")
    return name
