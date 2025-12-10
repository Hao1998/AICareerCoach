# Adzuna Job Fetching Setup Guide

This guide explains how to set up and use the Adzuna API integration to fetch real job postings from online sources.

## What is Adzuna?

Adzuna is a job search aggregator that collects job listings from thousands of websites. Their API allows you to programmatically search and retrieve job postings. The API aggregates jobs from sources including:

- Company career pages
- Job boards
- Recruitment agencies
- Government job sites
- And many more

## Getting Your API Credentials

### Step 1: Register for Adzuna API

1. Visit [Adzuna Developer Portal](https://developer.adzuna.com/)
2. Click "Register" or "Sign Up"
3. Fill in the registration form with your details
4. Verify your email address

### Step 2: Create an Application

1. Log in to the Adzuna Developer Portal
2. Navigate to "Applications" or "My Apps"
3. Click "Create New Application"
4. Enter your application details:
   - **Application Name**: e.g., "AI Career Coach"
   - **Description**: Brief description of your project
   - **Website URL**: Your website or localhost URL
5. Submit the form

### Step 3: Get Your Credentials

After creating your application, you'll receive:
- **App ID**: A unique identifier for your application
- **App Key**: A secret key for authentication

Keep these credentials secure and never commit them to version control.

## Configuration

### Option 1: Environment Variables (Recommended)

Set environment variables in your shell:

```bash
# Linux/Mac
export ADZUNA_APP_ID="your_app_id_here"
export ADZUNA_APP_KEY="your_app_key_here"

# Windows (Command Prompt)
set ADZUNA_APP_ID=your_app_id_here
set ADZUNA_APP_KEY=your_app_key_here

# Windows (PowerShell)
$env:ADZUNA_APP_ID="your_app_id_here"
$env:ADZUNA_APP_KEY="your_app_key_here"
```

### Option 2: .env File

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your credentials:
   ```
   ADZUNA_APP_ID=your_app_id_here
   ADZUNA_APP_KEY=your_app_key_here
   ```

3. Load the .env file (if using python-dotenv):
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```

## Using the Job Fetcher

### Through the Web Interface

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Navigate to the Jobs page: `http://localhost:5001/jobs`

3. Click "Fetch from Adzuna" button

4. Fill in the search criteria:
   - **Keywords**: Job titles, skills, or keywords (e.g., "python developer")
   - **Location**: City, state, or "Remote"
   - **Max Jobs**: How many jobs to fetch (1-200)
   - **Max Age**: Only jobs posted within this many days

5. Click "Fetch Jobs"

The system will:
- Search Adzuna's database
- Parse and store matching jobs
- Compute embeddings for each job
- Rebuild the FAISS search index
- Show you statistics about the fetch operation

### Through the API

You can also fetch jobs programmatically using the API endpoint:

```bash
curl -X POST http://localhost:5001/api/jobs/fetch \
  -H "Content-Type: application/json" \
  -d '{
    "keywords": "python developer",
    "location": "San Francisco",
    "max_jobs": 50,
    "max_days_old": 30
  }'
```

Response:
```json
{
  "success": true,
  "stats": {
    "fetched": 50,
    "stored": 45,
    "duplicates": 5,
    "errors": 0,
    "error_messages": []
  }
}
```

### Through Python Code

```python
from job_fetcher import fetch_jobs_from_adzuna

# Fetch jobs
stats = fetch_jobs_from_adzuna(
    keywords="machine learning engineer",
    location="New York",
    max_jobs=100,
    max_days_old=14
)

print(f"Fetched: {stats['fetched']}")
print(f"Stored: {stats['stored']}")
print(f"Duplicates: {stats['duplicates']}")
```

## Search Tips

### Keywords

- Be specific: "python backend developer" vs "developer"
- Use job titles: "data scientist", "devops engineer"
- Include technologies: "react developer", "aws architect"
- Leave empty to fetch all available jobs

### Location

- City names: "San Francisco", "New York", "London"
- State/Province: "California", "Texas"
- Remote work: "Remote", "Work from home"
- Leave empty for all locations

### Max Jobs

- Start with 50 jobs to test
- Increase to 100-200 for comprehensive results
- API limit is typically 50 results per page
- The fetcher automatically handles pagination

### Max Days Old

- `7` - Only this week's jobs
- `14` - Last two weeks
- `30` - Last month (recommended)
- `60` - Last two months

## Supported Countries

Adzuna operates in multiple countries. Update the country code in `job_fetcher.py`:

```python
fetcher = AdzunaJobFetcher(country='us')  # Default is 'us'
```

Available countries:
- `us` - United States
- `gb` - United Kingdom
- `ca` - Canada
- `au` - Australia
- `de` - Germany
- `fr` - France
- And many more...

## Troubleshooting

### "API credentials required" Error

**Solution**: Ensure you've set the environment variables or created a `.env` file with your credentials.

```bash
# Check if variables are set
echo $ADZUNA_APP_ID
echo $ADZUNA_APP_KEY
```

### "Failed to fetch jobs from Adzuna" Error

**Possible causes**:
1. Invalid API credentials
2. Network connection issues
3. API rate limits exceeded
4. Invalid search parameters

**Solution**:
- Verify your credentials are correct
- Check your internet connection
- Wait a few minutes if rate limited
- Try simpler search parameters

### No Jobs Found

**Possible causes**:
1. Search criteria too specific
2. Location not available in Adzuna
3. No jobs matching criteria

**Solution**:
- Broaden your search (remove location or keywords)
- Try different keywords
- Check if your location is supported

### Duplicates Skipped

This is normal! The system checks for duplicate jobs (same title and company) to avoid storing the same job multiple times. This protects your database from redundant data.

## API Rate Limits

Adzuna has rate limits to prevent abuse:
- Free tier: Typically 500-1000 requests/month
- Requests are counted per API call, not per job
- Fetching 50 jobs = 1 API request

**Tips to stay under limits**:
- Fetch jobs in batches (50-100 at a time)
- Don't fetch the same jobs repeatedly
- Use the `max_days_old` parameter to limit results

## Data Privacy & Usage

- Jobs are stored locally in your SQLite database
- No data is sent to Adzuna except search queries
- Comply with Adzuna's Terms of Service
- Don't republish job listings without permission
- This tool is for personal career coaching use

## Advanced Usage

### Custom Country Selection

```python
from job_fetcher import AdzunaJobFetcher

fetcher = AdzunaJobFetcher(country='gb')  # UK jobs
stats = fetcher.fetch_and_store_jobs(
    keywords="software engineer",
    max_jobs=50
)
```

### Skip Duplicate Checking

```python
stats = fetch_jobs_from_adzuna(
    keywords="python developer",
    skip_duplicates=False  # Allow duplicates
)
```

### Integration with Existing Workflow

The fetched jobs are automatically:
1. ✅ Stored in the database
2. ✅ Embedded using HuggingFace models
3. ✅ Indexed in FAISS for fast retrieval
4. ✅ Available for CV matching

No additional steps needed!

## Support

For issues with:
- **Adzuna API**: Visit [Adzuna Support](https://www.adzuna.com/contact)
- **This Integration**: Check the code in `job_fetcher.py` or open an issue

## Resources

- [Adzuna Developer Documentation](https://developer.adzuna.com/)
- [Adzuna API Search Docs](https://developer.adzuna.com/docs/search)
- [Interactive API Explorer](https://developer.adzuna.com/activedocs)

---

**Note**: While this guide mentions fetching from LinkedIn, LinkedIn data is actually accessed through Adzuna's aggregation service, which complies with all applicable terms of service. We don't scrape LinkedIn directly.
