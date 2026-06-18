"""
tests/test_llm_chains.py
========================
Feature tested: resume analysis chain and interview-prep roadmap chain
(services/llm_service.py).

Background
----------
Both chains were migrated from LangChain's deprecated LLMChain + .run() API to
LCEL pipelines (prompt | llm | StrOutputParser()).  The user-visible behaviour
must be identical: pass a dict of template variables in, get a plain string out.

The tests use GenericFakeChatModel — a LangChain-provided test double that
returns pre-scripted AIMessages without touching any API.  monkeypatch swaps
get_llm() so the chains build with the fake model instead of ChatXAI.

What each test covers
---------------------
test_resume_analysis_chain_returns_plain_text
    invoke_chain_with_retry(chain, resume=...) returns the LLM's text directly
    as a string (regression: LLMChain.run() used to return a string; LCEL
    chain.invoke() must do the same via invoke_chain_with_retry).

test_resume_analysis_chain_invoke_returns_string_directly
    chain.invoke({"resume": ...}) itself returns a str, not a dict.  This pins
    the LCEL contract: StrOutputParser() is wired correctly.

test_roadmap_chain_returns_plain_text
    The interview-prep roadmap chain accepts all its template variables
    (resume, job_title, company, job_description, job_requirements, skill_gaps,
    timeline_months) and returns the LLM output as a plain string.

test_roadmap_chain_invoke_returns_string_directly
    Same LCEL contract check as above but for the roadmap chain.

No API key needed — ChatXAI is never instantiated.
"""

import itertools

import pytest
from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

import services.llm_service as llm_service


@pytest.fixture
def fake_llm(monkeypatch):
    """Patch get_llm to return a fake chat model and reset cached chains."""
    def _make(text):
        model = GenericFakeChatModel(messages=itertools.repeat(AIMessage(content=text)))
        monkeypatch.setattr(llm_service, "get_llm", lambda *a, **k: model)
        # Reset module-level chain caches so they rebuild with the fake model.
        monkeypatch.setattr(llm_service, "_resume_analysis_chain", None, raising=False)
        monkeypatch.setattr(llm_service, "_preparation_roadmap_chain", None, raising=False)
        return model
    return _make


def test_resume_analysis_chain_returns_plain_text(fake_llm):
    fake_llm("CANDIDATE SUMMARY TEXT")
    chain = llm_service.get_resume_analysis_chain()
    result = llm_service.invoke_chain_with_retry(chain, resume="Jane Doe, Python dev")
    assert result == "CANDIDATE SUMMARY TEXT"


def test_resume_analysis_chain_invoke_returns_string_directly(fake_llm):
    """LCEL contract: chain.invoke(dict) yields a str, not a dict."""
    fake_llm("DIRECT INVOKE TEXT")
    chain = llm_service.get_resume_analysis_chain()
    out = chain.invoke({"resume": "anything"})
    assert isinstance(out, str)
    assert out == "DIRECT INVOKE TEXT"


def test_roadmap_chain_returns_plain_text(fake_llm):
    fake_llm("ROADMAP JSON HERE")
    chain = llm_service.get_preparation_roadmap_chain()
    result = llm_service.invoke_chain_with_retry(
        chain,
        resume="resume text",
        job_title="ML Engineer",
        company="ACME",
        job_description="desc",
        job_requirements="reqs",
        skill_gaps="gap1, gap2",
        timeline_months=6,
    )
    assert result == "ROADMAP JSON HERE"


def test_roadmap_chain_invoke_returns_string_directly(fake_llm):
    fake_llm("ROADMAP DIRECT")
    chain = llm_service.get_preparation_roadmap_chain()
    out = chain.invoke({
        "resume": "r", "job_title": "t", "company": "c",
        "job_description": "d", "job_requirements": "rq",
        "skill_gaps": "g", "timeline_months": 3,
    })
    assert isinstance(out, str)
    assert out == "ROADMAP DIRECT"
