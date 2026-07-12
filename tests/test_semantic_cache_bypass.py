"""
tests/test_semantic_cache_bypass.py
=====================================
Feature tested: semantic LLM cache bypass for the conversational chat prompt
(services/semantic_cache.py — SemanticCache, DEFAULT_BYPASS_PREFIXES).

Background
----------
SemanticCache stores LLM responses keyed by the embedding of the prompt.  If
two different prompts embed with cosine similarity >= 0.90, the second one gets
the first one's cached answer.

Chat prompts are dangerous here: every turn shares a large, identical system
prompt (build_system_prompt) which dominates the embedding vector.  Two
completely different user messages can embed nearly identically because the
system prompt is so long relative to the user message.  The result:

  Turn 1: "Hi"  -> LLM answers, stored in cache
  Turn 2: "Find me Python jobs"  -> embed similarity >= 0.90  -> cache hit
          -> user gets Turn 1's greeting answer
          -> and because cache hits skip on_llm_new_token, the streaming UI
             shows "(no response)"

Fix: the chat system prompt starts with "<trusted_instructions>", which is in
DEFAULT_BYPASS_PREFIXES.  SemanticCache._should_bypass() checks the first
message's content for any bypass prefix and skips both lookup and update.

What each test covers
---------------------
test_build_system_prompt_starts_with_trusted_instructions
    build_system_prompt() output starts with "<trusted_instructions>" — this is
    the tag that the cache uses as the bypass signal.  If the prompt format
    changes, this test catches the breakage before the cache bypass stops working.

test_chat_prompt_is_bypassed_by_default_config
    A serialised chat prompt (LangChain JSON format) is neither stored nor
    returned by SemanticCache when DEFAULT_BYPASS_PREFIXES is active.
    lookup() returns None before and after update() — the cache is a no-op for
    chat prompts.

test_non_chat_prompt_is_still_cached
    A generic prompt that does not start with any bypass prefix IS cached
    normally.  Guards against over-bypassing: job/resume agent prompts that are
    NOT on the bypass list must still benefit from the cache.

No API key or embedding model needed — FakeEmb returns a fixed vector, and the
bypass test uses object() as the embedding model (it is never called because
the bypass check short-circuits before embed_query).
"""

import json
from types import SimpleNamespace

from services.semantic_cache import SemanticCache, DEFAULT_BYPASS_PREFIXES
from chatbot.agent import build_system_prompt


def _serialize_chat_prompt(system_prompt_text):
    # LangChain serializes chat messages to JSON; SemanticCache reads the first
    # message's content. The system prompt is the first message.
    return json.dumps([{"kwargs": {"content": system_prompt_text}}])


def test_build_system_prompt_starts_with_trusted_instructions():
    user = SimpleNamespace(full_name="Jane Doe", username="jane")
    prompt = build_system_prompt(user, resume=None, agent_config=None)
    assert prompt.startswith("<trusted_instructions>")


def test_chat_prompt_is_bypassed_by_default_config():
    user = SimpleNamespace(full_name="Jane Doe", username="jane")
    system_prompt = build_system_prompt(user, resume=None, agent_config=None)
    chat_prompt = _serialize_chat_prompt(system_prompt)

    cache = SemanticCache(embedding_model=object(), bypass_prefixes=DEFAULT_BYPASS_PREFIXES)
    # bypassed -> never stored, never returns a cached value
    assert cache.lookup(chat_prompt, "llm") is None
    cache.update(chat_prompt, "llm", "CACHED ANSWER")
    assert cache.lookup(chat_prompt, "llm") is None


def test_non_chat_prompt_is_still_cached():
    """Guard against over-bypassing: an ordinary prompt must still cache."""
    class FakeEmb:
        def embed_query(self, text):
            return [1.0, 0.0, 0.0]

    cache = SemanticCache(embedding_model=FakeEmb(), bypass_prefixes=DEFAULT_BYPASS_PREFIXES)
    prompt = json.dumps([{"kwargs": {"content": "Summarize this generic text."}}])

    assert cache.lookup(prompt, "llm") is None
    cache.update(prompt, "llm", "RESP")
    assert cache.lookup(prompt, "llm") == "RESP"


def test_resume_analysis_prompt_is_bypassed():
    """Resume-analysis prompts must bypass the cache.

    The resume text dominates the embedding, so two distinct resumes can collide
    above the 0.90 threshold and return one user's summary to another (privacy
    leak). The resume-summary prompt prefix is on the bypass list to prevent this.
    """
    from services.llm_service import _resume_summary_template

    # An embedder that returns the SAME vector for everything — worst case, forces
    # a cache hit unless the prompt is bypassed.
    class ConstEmb:
        def embed_query(self, text):
            return [1.0, 0.0, 0.0]

    cache = SemanticCache(embedding_model=ConstEmb(), bypass_prefixes=DEFAULT_BYPASS_PREFIXES)

    prompt_a = json.dumps([{"kwargs": {"content": _resume_summary_template.format(
        resume="Alice — senior data scientist, PhD, 10 years experience.")}}])
    prompt_b = json.dumps([{"kwargs": {"content": _resume_summary_template.format(
        resume="Bob — junior frontend developer, bootcamp grad, 1 year.")}}])

    cache.update(prompt_a, "llm", "ALICE_SUMMARY")
    # Even with an identical embedding, Bob must NOT receive Alice's summary.
    assert cache.lookup(prompt_b, "llm") is None
