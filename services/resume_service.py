"""
Resume Service

Handles PDF text extraction and resume Q&A using vector indices.
No Flask routes here — pure business logic.
"""

import os
import threading
from collections import OrderedDict

from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable

from jobs.utils import get_embeddings
from services.input_guard import scan_resume_text

_resume_index_cache = OrderedDict()
_resume_cache_lock = threading.Lock()
_MAX_RESUME_CACHE = 50

text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
)


@traceable(run_type="tool", name="pdf-extraction")
def extract_text_from_pdf(pdf_path, user_id=None):
    """Extract all text from a PDF file, stripping any embedded injection payloads."""
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    return scan_resume_text(text, user_id=user_id)


def get_resume_text(resume) -> str:
    """Return resume plain text, using the DB cache when available to skip PDF I/O."""
    if resume.text_content:
        return resume.text_content
    return extract_text_from_pdf(resume.file_path, user_id=resume.user_id)


def _get_cached_resume_index(user_id):
    """Load user's resume FAISS index, caching in-memory with LRU eviction."""
    user_vector_dir = os.path.join('vector_index', str(user_id))

    if not os.path.exists(user_vector_dir):
        return None

    index_file = os.path.join(user_vector_dir, 'index.faiss')
    if not os.path.exists(index_file):
        return None

    current_mtime = os.path.getmtime(index_file)

    with _resume_cache_lock:
        if user_id in _resume_index_cache:
            cached_db, cached_mtime = _resume_index_cache[user_id]
            if cached_mtime == current_mtime:
                _resume_index_cache.move_to_end(user_id)
                return cached_db

        vector_db = FAISS.load_local(
            user_vector_dir, get_embeddings(),
            allow_dangerous_deserialization=True
        )
        _resume_index_cache[user_id] = (vector_db, current_mtime)

        while len(_resume_index_cache) > _MAX_RESUME_CACHE:
            _resume_index_cache.popitem(last=False)

        return vector_db


def invalidate_resume_cache(user_id):
    """Remove a user's resume index from the in-memory cache."""
    with _resume_cache_lock:
        _resume_index_cache.pop(user_id, None)


_QA_PROMPT = ChatPromptTemplate.from_template(
    "Use the following pieces of context to answer the question at the end. "
    "If you don't know the answer, say so — don't make one up.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nHelpful Answer:"
)


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


@traceable(run_type="retriever", name="resume-qa")
def perform_qa(query, user_id):
    """Perform Q&A on the user's resume vector index (LCEL retrieval chain)."""
    from services.llm_service import get_llm

    vector_db = _get_cached_resume_index(user_id)
    if vector_db is None:
        return "Please upload a resume first before asking questions."

    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    # LCEL retrieval chain: fetch docs -> stuff into prompt -> model -> str.
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | _QA_PROMPT
        | get_llm()
        | StrOutputParser()
    )
    return chain.invoke(query)
