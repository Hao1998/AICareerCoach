# AICareerCoach

An intelligent AI-powered career coaching platform that analyzes resumes and matches them with perfect job opportunities using advanced machine learning and natural language processing.

## Features

### 1. Resume Analysis
- Upload PDF resumes for comprehensive AI analysis
- Extracts career objectives, skills, experience, education, and achievements
- Uses Grok for intelligent resume evaluation
- Provides actionable insights and recommendations

### 2. CV-Job Matching (NEW!)
- **Intelligent Job Matching**: Automatically finds the best matching jobs for uploaded resumes
- **Match Scoring**: AI-powered scoring system (0-100%) based on skills, experience, and requirements
- **Skill Analysis**: Identifies matched skills and skill gaps for each job
- **Personalized Recommendations**: Get AI-generated advice for improving job match scores
- **Vector Similarity Search**: Uses FAISS embeddings for semantic matching between resumes and jobs

### 3. Job Management
- Add and manage job postings
- View detailed job descriptions and requirements
- Track job matches across different resumes
- Filter and search through available positions

### 4. Interactive Q&A
- Ask questions about uploaded resumes
- Get intelligent answers using retrieval-augmented generation (RAG)
- Semantic search through resume content

## Technology Stack

- **Backend**: Flask (Python)
- **AI/ML**:
  - Grok for analysis and recommendations
  - LangChain for prompt management
  - HuggingFace Embeddings for vector representations
  - FAISS for similarity search
- **Database**: SQLite with SQLAlchemy ORM
- **PDF Processing**: PyPDF2, pdfplumber
- **Frontend**: HTML, Tailwind CSS, Framer Motion

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AICareerCoach
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Grok API key:
   - Edit `app.py` and replace the API key on line 57
   - Or set it as an environment variable:
```bash
export XAI_API_KEY='your-api-key-here'
```

4. Run the application:
```bash
python app.py
```

5. (Optional) Add sample job data:
```bash
python add_sample_jobs.py
```

6. Open your browser and navigate to:
```
http://localhost:5000
```

## How the CV-Job Matching Works

### 1. Resume Upload
When you upload a resume:
- Text is extracted from the PDF
- Content is split into chunks for efficient processing
- Embeddings are generated using HuggingFace models
- Resume is analyzed by Grok for comprehensive evaluation

### 2. Job Matching Algorithm
The matching process involves:

**Step 1: Vector Similarity**
- Both resume and job descriptions are converted to embeddings
- Cosine similarity is calculated between vectors
- Higher similarity indicates better semantic match

**Step 2: AI Analysis**
- Grok analyzes the resume against each job posting
- Identifies specific matching skills
- Highlights skill gaps and areas for improvement
- Generates a detailed match score (0-100%)
- Provides personalized recommendations

**Step 3: Results Ranking**
- Jobs are ranked by match score
- Top 5 matches are displayed by default
- Each match includes:
  - Match percentage
  - Matched skills (green badges)
  - Skill gaps (red badges)
  - AI recommendation

### 3. Match Storage
- All matches are stored in the database
- Historical match data can be retrieved via API
- Track improvements over time

## API Endpoints

### Resume Operations
- `GET /` - Home page with resume upload
- `POST /upload` - Upload and analyze resume
- `GET /ask` - Q&A interface
- `POST /ask` - Submit question about resume

### Job Management
- `GET /jobs` - List all active jobs
- `GET /jobs/add` - Add job form
- `POST /jobs/add` - Create new job posting
- `GET /jobs/<id>` - View job details
- `POST /jobs/<id>/delete` - Deactivate job

### API Routes
- `GET /api/jobs` - Get all jobs as JSON
- `GET /api/matches/<filename>` - Get matches for a specific resume

## Database Schema

### JobPosting
- `id`: Primary key
- `title`: Job title
- `company`: Company name
- `location`: Job location
- `job_type`: Full-time, Part-time, Contract, etc.
- `description`: Detailed job description
- `requirements`: Job requirements and qualifications
- `salary_range`: Salary information
- `posted_date`: Date posted
- `is_active`: Active status

### JobMatch
- `id`: Primary key
- `resume_filename`: Name of uploaded resume
- `job_id`: Foreign key to JobPosting
- `match_score`: AI-generated match percentage
- `matched_skills`: JSON array of matching skills
- `gaps`: JSON array of skill gaps
- `recommendation`: AI-generated advice
- `created_at`: Match timestamp

## Usage Examples

### Example 1: Upload Resume and Find Jobs
1. Go to home page
2. Upload your resume PDF
3. View analysis results
4. Scroll down to see top matching jobs
5. Click "View Full Details" to see complete job description

### Example 2: Add a Job Posting
1. Click "Add Job" from home page or jobs list
2. Fill in job details:
   - Title, Company, Location
   - Job Type and Salary Range
   - Description and Requirements
3. Submit the form
4. Job is now available for matching

### Example 3: Use Q&A Feature
1. Upload a resume
2. Click "Ask a Question"
3. Type your question (e.g., "What are my strongest skills?")
4. Get AI-powered answer based on resume content

## Project Structure

```
AICareerCoach/
├── app.py                 # Main Flask application
├── models.py              # Database models
├── requirements.txt       # Python dependencies
├── add_sample_jobs.py     # Sample data script
├── templates/            # HTML templates
│   ├── index.html        # Home page
│   ├── results.html      # Resume analysis + job matches
│   ├── jobs.html         # Job listings
│   ├── add_job.html      # Add job form
│   ├── view_job.html     # Job details
│   ├── ask.html          # Q&A interface
│   └── qa_results.html   # Q&A results
├── uploads/              # Uploaded resumes
├── vector_index/         # FAISS index for resume embeddings
├── job_vector_index/     # FAISS index for job embeddings
└── career_coach.db       # SQLite database
```

## Key Features Explained

### Embedding-Based Matching
The system uses semantic embeddings to understand the meaning behind resume content and job requirements, not just keyword matching. This results in:
- More accurate matches based on context
- Better understanding of transferable skills
- Identification of relevant experience even with different terminology

### AI-Powered Recommendations
Each match includes personalized advice such as:
- Which skills to highlight in your application
- Areas where you should gain more experience
- How to position yourself for the role
- Suggestions for professional development

### Real-Time Analysis
All processing happens in real-time:
- Resume upload and analysis: ~5-10 seconds
- Job matching: ~2-3 seconds per job
- Results displayed immediately

## Future Enhancements

Potential improvements:
- User authentication and profile management
- Resume optimization suggestions
- Cover letter generation
- Interview preparation tips
- Job application tracking
- Email notifications for new matching jobs
- Batch upload for multiple resumes
- Export matches to PDF/Excel
- Integration with job boards (LinkedIn, Indeed, etc.)

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

[Add your license information here]
