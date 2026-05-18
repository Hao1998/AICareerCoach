"""
Resume Service

Handles PDF text extraction and resume Q&A using vector indices.
No Flask routes here — pure business logic.
"""

import os
import threading
from collections import OrderedDict

import PyPDF2
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langsmith import traceable

from job_utils import get_embeddings
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
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
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


@traceable(run_type="retriever", name="resume-qa")
def perform_qa(query, user_id):
    """Perform Q&A on the user's resume vector index"""
    from services.llm_service import get_llm

    vector_db = _get_cached_resume_index(user_id)
    if vector_db is None:
        return "Please upload a resume first before asking questions."

    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    rqa = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    result = rqa.invoke(query)
    return result['result']
