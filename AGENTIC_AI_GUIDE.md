# Job Scout Agent - Agentic AI Feature Guide

## Overview

The **Job Scout Agent** is an autonomous AI-powered assistant that continuously searches for job opportunities matching your resume and skills. Unlike traditional job search tools that require manual effort, the Job Scout Agent works autonomously in the background, making intelligent decisions about what jobs to fetch, analyze, and recommend.

## What is Agentic AI?

**Agentic AI** refers to AI systems that can:
- **Autonomously plan and decide** what actions to take to achieve goals
- **Use tools** (APIs, databases, search engines) to accomplish tasks
- **Learn and adapt** from feedback and past experiences
- **Work independently** without constant human supervision

The Job Scout Agent demonstrates these capabilities by:
1. Automatically fetching new jobs from external sources (Adzuna API)
2. Analyzing your resume to determine what types of jobs to search for
3. Using AI to match jobs against your skills and experience
4. Making autonomous decisions about which matches are worth showing you
5. Learning from your feedback to improve future recommendations

## Features

### 1. Autonomous Job Discovery
- **What it does**: Automatically fetches new job postings daily from Adzuna API
- **How it works**: The agent analyzes your resume, extracts key skills and roles, then searches for relevant jobs without you having to specify search terms
- **Benefit**: You never miss new opportunities - the agent works 24/7 (when your app is running)

### 2. Intelligent Job Matching
- **What it does**: Uses AI (Grok LLM) to deeply analyze each job against your resume
- **How it works**: Two-stage process:
  1. Fast FAISS vector search to find top 20 candidates
  2. Detailed LLM analysis on best matches to generate match scores, identify skills, and provide recommendations
- **Benefit**: Get personalized match scores (0-100%) and specific skill gap analysis

### 3. Smart Filtering
- **What it does**: Only shows you jobs above your configured threshold (default 75%)
- **How it works**: Agent autonomously decides which matches are "good enough" based on your preferences
- **Benefit**: No clutter - only see high-quality matches

### 4. Learning from Feedback
- **What it does**: Tracks your "Interested/Not Interested" feedback
- **How it works**: Stores feedback in database for future analysis
- **Future capability**: Can be extended to learn your preferences and adjust search strategies

### 5. Hybrid Scheduling
- **What it does**: Runs automatically on schedule AND on-demand via "Check Now" button
- **How it works**: APScheduler for automatic daily runs, manual trigger for immediate checks
- **Benefit**: Best of both worlds - automation + control

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Job Scout Agent System                   │
└─────────────────────────────────────────────────────────────┘

┌──────────────────┐    ┌──────────────────┐    ┌─────────────┐
│  APScheduler     │───▶│  Job Scout Agent │───▶│  Database   │
│  (Scheduler)     │    │  (Core Logic)    │    │  (SQLite)   │
└──────────────────┘    └──────────────────┘    └─────────────┘
        │                        │
        │                        ├──▶ Adzuna API (Fetch Jobs)
        │                        ├──▶ FAISS (Vector Search)
        │                        └──▶ Grok LLM (Job Analysis)
        │
        ▼
┌──────────────────┐
│   Flask Routes   │
│   (User Interface)│
└──────────────────┘
```

### Database Schema

**New Tables:**

1. **agent_configs** - User agent configuration
   - `user_id`, `is_enabled`, `schedule_time`, `match_threshold`, `max_results_per_run`

2. **agent_run_history** - Tracks each agent execution
   - `user_id`, `run_type`, `status`, `jobs_fetched`, `jobs_analyzed`, `matches_found`

3. **job_matches** (extended) - Added agent-related fields
   - `agent_generated`, `agent_run_id`, `user_feedback`, `feedback_at`

### Key Files

- `job_scout_agent.py` - Core autonomous agent logic
- `agent_scheduler.py` - APScheduler integration
- `models.py` - Database models (updated)
- `app.py` - Flask routes for agent UI
- `templates/agent_*.html` - Agent UI templates

## How the Agent Works (Step-by-Step)

### Autonomous Decision-Making Process

```
User enables agent → Scheduler triggers at 9 AM daily
                    ↓
┌─────────────────────────────────────────────────────┐
│ STEP 1: ANALYZE RESUME                              │
│ Decision: "What skills does this person have?"      │
│ Action: Extract text from latest resume PDF         │
│ Tool: PyPDF2                                         │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ STEP 2: DETERMINE SEARCH KEYWORDS                   │
│ Decision: "What jobs should I search for?"          │
│ Action: Use LLM to extract job titles from resume   │
│ Tool: Grok LLM                                       │
│ Example: "Python Developer, Data Scientist"         │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ STEP 3: FETCH NEW JOBS                              │
│ Decision: "What jobs are available?"                │
│ Action: Query Adzuna API for recent postings        │
│ Tool: AdzunaJobFetcher                               │
│ Result: 20 new jobs fetched                          │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ STEP 4: STAGE 1 MATCHING (Fast Filter)              │
│ Decision: "Which jobs are potentially relevant?"    │
│ Action: FAISS vector similarity search               │
│ Tool: HuggingFace embeddings + FAISS                 │
│ Result: Top 20 candidate jobs                        │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ STEP 5: STAGE 2 MATCHING (Deep Analysis)            │
│ Decision: "How well does each job match?"           │
│ Action: LLM analyzes resume vs job description       │
│ Tool: Grok LLM with structured prompt                │
│ Output: Match score, skills, gaps, recommendation    │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ STEP 6: FILTER BY THRESHOLD                         │
│ Decision: "Is this match good enough to show user?" │
│ Action: If match_score >= threshold (75%), save it  │
│ Tool: Database query + logic                         │
│ Result: Only 3 matches above 75% saved               │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ STEP 7: STORE RESULTS                               │
│ Action: Save matches and run history to database    │
│ User sees: "Found 3 new matches!" on dashboard      │
└─────────────────────────────────────────────────────┘
```

## Configuration

### Agent Settings

Access via: **Profile → Agent Dashboard → Configure**

| Setting | Description | Default | Range |
|---------|-------------|---------|-------|
| **Enabled** | Whether agent runs automatically | True | On/Off |
| **Schedule Time** | Daily run time | 09:00 | Any time |
| **Match Threshold** | Minimum match score to notify | 75% | 50-95% |
| **Max Results** | Max matches per run | 10 | 5-20 |

### Customization Tips

**High Standards** (Few, excellent matches):
- Threshold: 85%+
- Max Results: 5

**Broad Search** (More opportunities):
- Threshold: 60%
- Max Results: 20

## Usage

### First-Time Setup

1. **Upload Your Resume**
   - Go to Home → Upload your latest resume PDF
   - The agent needs this to understand your skills

2. **Configure the Agent** (Optional)
   - Visit Agent Dashboard → Configure
   - Adjust threshold and schedule time
   - Enable the agent

3. **Manual First Run**
   - Click "Check for New Jobs Now"
   - See immediate results
   - Agent will run automatically daily thereafter

### Daily Workflow

**Automated:**
- Agent runs daily at your scheduled time
- New matches appear on Agent Dashboard
- No action needed from you

**Manual Check:**
- Visit Agent Dashboard anytime
- Click "Check for New Jobs Now"
- Get instant results

**Reviewing Matches:**
- View matches on Agent Dashboard
- Click "View Full Job Details" to see complete posting
- Mark as "Interested" or "Not Interested"
- Feedback helps agent learn your preferences

### Viewing History

- **Recent Runs**: See last 10 runs on Agent Dashboard
- **Full History**: Agent Dashboard → "View All" → Complete run log
- **Run Details**: Click any run to see specific matches found

## Local vs. Production Deployment

### Running Locally

**How it works:**
- Agent runs only when Flask app is running
- Scheduler operates in background thread
- Perfect for development and personal use

**Limitations:**
- Computer must be on and app running
- No runs when laptop is closed/sleeping

**Solution:**
- Keep Flask running in background
- Use manual "Check Now" when you start app

### Production Deployment (Recommended for 24/7 Operation)

**Deploy to cloud:**
1. **Render/Railway** (Free tier available)
   ```bash
   # Push to GitHub
   git push origin main

   # Connect to Render/Railway
   # They auto-deploy from GitHub
   ```

2. **Set environment variables:**
   ```
   XAI_API_KEY=your_key
   ADZUNA_APP_ID=your_id
   ADZUNA_APP_KEY=your_key
   ```

3. **Benefits:**
   - Runs 24/7 automatically
   - Never miss new jobs
   - Can add email notifications

## Advanced Features (Future Enhancements)

### Currently Implemented
✅ Autonomous job fetching
✅ AI-powered matching
✅ Smart filtering by threshold
✅ Feedback collection
✅ Run history tracking
✅ Manual + automatic triggers

### Potential Enhancements
- **Email Notifications**: Get emailed when high-quality matches are found
- **Multi-Source Search**: Fetch from LinkedIn, Indeed, Glassdoor
- **Learning Algorithm**: Adjust search based on feedback patterns
- **Resume Optimization**: Suggest resume improvements for better matches
- **Application Tracking**: Track applications submitted
- **Interview Scheduling**: Auto-coordinate interview times

## Troubleshooting

### Agent Not Running

**Problem**: "Next Run: Not scheduled"
**Solution**:
1. Check if agent is enabled in Configuration
2. Verify Flask app is running
3. Check logs for errors

### No Matches Found

**Problem**: Agent runs but finds 0 matches
**Solution**:
1. Lower match threshold (try 65%)
2. Check if resume is uploaded
3. Verify Adzuna API credentials are set
4. Try manual "Check Now" to see error messages

### Jobs Not Fetching

**Problem**: "Jobs Fetched: 0"
**Solution**:
1. Verify Adzuna API credentials:
   ```bash
   echo $ADZUNA_APP_ID
   echo $ADZUNA_APP_KEY
   ```
2. Check API rate limits (500 calls/month on free tier)
3. Try broader search terms

### Scheduler Not Starting

**Problem**: Agent never runs automatically
**Solution**:
1. Check app startup logs for scheduler errors
2. Verify APScheduler is installed: `pip list | grep APScheduler`
3. Restart Flask app

## API Endpoints

### Agent Routes

```
GET  /agent/dashboard          - Agent dashboard UI
GET  /agent/config            - Configuration page
POST /agent/config            - Update configuration
POST /agent/trigger           - Manual agent run
GET  /agent/history           - Full run history
GET  /agent/matches/<run_id>  - Matches from specific run
POST /agent/feedback/<match_id> - Submit user feedback
GET  /agent/status            - Agent status API (JSON)
```

### Example: Manual Trigger via API

```python
import requests

response = requests.post('http://localhost:5001/agent/trigger')
result = response.json()

print(f"Matches found: {result['matches_found']}")
print(f"Jobs analyzed: {result['jobs_analyzed']}")
```

## Performance

### Metrics

- **Job Fetching**: ~2-3 seconds (Adzuna API)
- **FAISS Search**: ~0.5 seconds (Stage 1)
- **LLM Analysis**: ~5-10 seconds per job (Stage 2)
- **Total Run Time**: ~30-60 seconds for 10 matches

### Optimization

The two-stage retrieval architecture provides **60-100x speedup** compared to analyzing all jobs:
- Old method: Analyze 1000 jobs = 1000 LLM calls = ~3 hours
- New method: FAISS filters to 20, analyze 5 = 5 LLM calls = ~30 seconds

## Security & Privacy

- All data stored locally in SQLite database
- API keys stored in environment variables
- User-specific data isolation (multi-user support)
- No data shared with third parties except API providers

## Cost Considerations

### API Usage

**Adzuna API** (Free Tier):
- 500 calls/month
- Agent uses ~1 call per day
- Monthly cost: $0

**XAI API** (Grok):
- Pay per token
- ~10 matches/day = ~$0.10/day
- Monthly cost: ~$3

**Total**: ~$3/month for autonomous job searching

## Contributing

Want to enhance the agent? Ideas:
1. Add more job sources (LinkedIn API, Indeed scraper)
2. Implement email notifications
3. Add cover letter generation
4. Build learning algorithm for personalization
5. Create mobile app for notifications

## Support

Having issues? Check:
1. This documentation
2. Application logs: `tail -f app.log`
3. GitHub Issues: [Report a bug]
4. Flask debug mode: Set `app.debug = True` for detailed errors

---

**Built with:** Flask, APScheduler, LangChain, Grok, FAISS, HuggingFace
**License:** MIT
**Version:** 1.0.0
