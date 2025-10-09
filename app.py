from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
import PyPDF2
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# from langchain_openai  import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
)

embeddings = HuggingFaceEmbeddings()


def perform_qa(query):
    db = FAISS.load_local("vector_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    rqa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    result = rqa.invoke(query)
    return result['result']


app = Flask(__name__)

# File upload configuration
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text


llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,

    api_key="KEY",  # if you prefer to pass api key in directly instaed of using env vars

)

resume_summary_template = """
Role: You are an AI Career Coach.

Task: Given the candidate's resume, provide a comprehensive summary that includes the following key aspects:

- Career Objective
- Skills and Expertise
- Professional Experience
- Educational Background
- Notable Achievements

Instructions:
Provide a concise summary of the resume, focusing on the candidate's skills, experience, and career trajectory. Ensure the summary is well-structured, clear, and highlights the candidate's strengths in alignment with industry standards.

Requirements:
{resume}

"""

resume_prompt = PromptTemplate(
    input_variables=["resume"],
    template=resume_summary_template,
)

resume_analysis_chain = LLMChain(
    llm=llm,
    prompt=resume_prompt,
)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Extracted   the  text from the PDF
        resume_text = extract_text_from_pdf(file_path)
        splitted_text = text_splitter.split_text(resume_text)
        vectorstore = FAISS.from_texts(splitted_text, embeddings)
        vectorstore.save_local("vector_index")

        # print(proposal_text)
        # Run SWOT analysis using the LLM chain
        resume_analysis = resume_analysis_chain.run(resume=resume_text)

        return render_template('results.html', resume_analysis=resume_analysis)


@app.route('/ask', methods=['GET', 'POST'])
def ask_query():
    if request.method == 'POST':
        query = request.form['query']
        result = perform_qa(query)
        return render_template('qa_results.html', query=query, result=result)
    return render_template('ask.html')


if __name__ == "__main__":
    app.run(debug=True)
