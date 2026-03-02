"""
Resume Service

Handles PDF text extraction and resume Q&A using vector indices.
No Flask routes here — pure business logic.
"""

import os
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

from job_utils import embeddings

text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
)


def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file"""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text


def perform_qa(query, user_id):
    """Perform Q&A on the user's resume vector index"""
    from services.llm_service import get_llm

    user_vector_dir = os.path.join('vector_index', str(user_id))

    if not os.path.exists(user_vector_dir):
        return "Please upload a resume first before asking questions."

    vector_db = FAISS.load_local(user_vector_dir, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    rqa = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    result = rqa.invoke(query)
    return result['result']
