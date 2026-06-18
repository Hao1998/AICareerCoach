"""
tests/test_resume_qa.py
=======================
Feature tested: resume Q&A — the ability to ask questions about an uploaded
resume and get answers grounded in its content (services/resume_service.perform_qa).

Background
----------
perform_qa was migrated from LangChain's RetrievalQA (a black-box chain that
bundles retrieval + LLM) to a transparent LCEL pipeline:

  {"context": retriever | _format_docs, "question": RunnablePassthrough()}
  | prompt | llm | StrOutputParser()

The contract is unchanged: pass (question, user_id) -> get a plain answer string.

Two fakes make this fully offline:

  DeterministicFakeEmbedding  — produces fixed-size vectors without a model file,
      used to build a real FAISS index in memory from two sample sentences.

  GenericFakeChatModel        — returns a scripted answer string without any API
      call, wired via monkeypatching get_llm().

  _get_cached_resume_index    — monkeypatched to return the fake FAISS index
      instead of loading from disk (which would require an actual uploaded resume).

What each test covers
---------------------
test_perform_qa_returns_answer_string
    Happy path: with a FAISS index present and the LLM returning a scripted
    answer, perform_qa returns that answer string verbatim.  Confirms the
    retrieval → prompt → LLM → parse pipeline is wired end-to-end.

test_perform_qa_without_index_prompts_upload
    When no resume has been uploaded (index is None), perform_qa returns the
    user-friendly "Please upload a resume first" message instead of crashing.
    Confirms the guard branch still works after the LCEL migration.

No API key or PDF file needed.
"""

import itertools

import pytest
from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_community.vectorstores import FAISS

import services.resume_service as resume_service
import services.llm_service as llm_service


@pytest.fixture
def fake_index(monkeypatch):
    emb = DeterministicFakeEmbedding(size=64)
    db = FAISS.from_texts(
        ["Jane Doe has 5 years of Python experience.",
         "Jane is AWS certified and knows Docker."],
        emb,
    )
    monkeypatch.setattr(resume_service, "_get_cached_resume_index", lambda uid: db)
    return db


@pytest.fixture
def fake_llm(monkeypatch):
    def _make(text):
        model = GenericFakeChatModel(messages=itertools.repeat(AIMessage(content=text)))
        monkeypatch.setattr(llm_service, "get_llm", lambda *a, **k: model)
        return model
    return _make


def test_perform_qa_returns_answer_string(fake_index, fake_llm):
    fake_llm("Jane has 5 years of Python experience.")
    answer = resume_service.perform_qa("How many years of Python?", user_id=1)
    assert isinstance(answer, str)
    assert answer == "Jane has 5 years of Python experience."


def test_perform_qa_without_index_prompts_upload(monkeypatch):
    monkeypatch.setattr(resume_service, "_get_cached_resume_index", lambda uid: None)
    answer = resume_service.perform_qa("anything", user_id=1)
    assert answer == "Please upload a resume first before asking questions."
