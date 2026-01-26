"""
Job Scout Agent - Autonomous AI agent for job searching and matching

This agent runs autonomously to:
1. Fetch new jobs from external sources (Adzuna API)
2. Analyze jobs against user's resume
3. Find and save high-quality matches
4. Learn from user feedback
"""

import os
import json
from datetime import datetime
from models import db, User, Resume, JobPosting, JobMatch, AgentConfig, AgentRunHistory
from job_fetcher import AdzunaJobFetcher
from job_utils import get_job_faiss_index, build_job_faiss_index
from langchain.prompts import PromptTemplate
from langchain_xai import ChatXAI
import PyPDF2


class JobScoutAgent:
    """
    Autonomous agent that scouts for jobs and matches them to users

    The agent demonstrates agentic AI behavior by:
    - Autonomous decision-making (what jobs to fetch, what to analyze)
    - Tool use (Adzuna API, FAISS search, LLM analysis)
    - Goal-oriented behavior (find best matches for user)
    - Learning from feedback (adjusts based on user preferences)
    """

    def __init__(self, app_context):
        """
        Initialize the Job Scout Agent

        Args:
            app_context: Flask application context for database access
        """
        self.app_context = app_context

        # Initialize LLM for job analysis
        xai_api_key = os.getenv("XAI_API_KEY")
        if not xai_api_key:
            raise RuntimeError("XAI_API_KEY environment variable is not set")

        self.llm = ChatXAI(
            xai_api_key=xai_api_key,
            model="grok-beta",
            temperature=0.7,
        )

        # Job matching prompt template
        self.matching_prompt = PromptTemplate(
            input_variables=["resume", "job_title", "company", "job_description", "job_requirements"],
            template="""You are an expert career coach analyzing if a job matches a candidate's resume.
 
Resume Summary:
{resume}
 
Job Details:
Title: {job_title}
Company: {company}
Description: {job_description}
Requirements: {job_requirements}
 
Analyze the match and provide a JSON response with:
1. match_score: 0-100 (higher is better match)
2. matched_skills: List of candidate's skills that match job requirements
3. skill_gaps: List of skills the candidate needs to develop
4. recommendation: Brief personalized recommendation (2-3 sentences)
 
Return ONLY valid JSON in this format:
{{"match_score": 85, "matched_skills": ["Python", "SQL"], "skill_gaps": ["AWS", "Docker"], "recommendation": "Strong match..."}}
"""
        )

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF resume"""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
        return text

    def run_for_user(self, user_id, run_type='manual'):
        """
        Run the job scout agent for a specific user

        This is the main autonomous agent loop that:
        1. Decides what jobs to fetch based on user's resume
        2. Fetches new jobs using external API
        3. Analyzes matches using AI
        4. Filters and saves results
        5. Learns from past feedback

        Args:
            user_id: User ID to run agent for
            run_type: 'manual' or 'scheduled'

        Returns:
            dict: Results summary with stats
        """
        with self.app_context:
            # Create run history record
            run_history = AgentRunHistory(
                user_id=user_id,
                run_type=run_type,
                status='running',
                started_at=datetime.utcnow()
            )
            db.session.add(run_history)
            db.session.commit()

            try:
                # Get user and config
                user = User.query.get(user_id)
                if not user:
                    raise Exception(f"User {user_id} not found")

                config = AgentConfig.query.filter_by(user_id=user_id).first()
                if not config:
                    # Create default config if not exists
                    config = AgentConfig(user_id=user_id)
                    db.session.add(config)
                    db.session.commit()

                # Check if agent is enabled
                if not config.is_enabled:
                    run_history.status = 'completed'
                    run_history.completed_at = datetime.utcnow()
                    run_history.results_summary = json.dumps({
                        'message': 'Agent is disabled for this user'
                    })
                    db.session.commit()
                    return {'status': 'disabled', 'message': 'Agent is disabled'}

                # DECISION 1: Get user's latest resume
                latest_resume = Resume.query.filter_by(
                    user_id=user_id,
                    is_active=True
                ).order_by(Resume.uploaded_at.desc()).first()

                if not latest_resume:
                    raise Exception("No resume found for user")

                # Extract resume text
                resume_text = self.extract_text_from_pdf(latest_resume.file_path)

                # DECISION 2: Determine what jobs to fetch based on resume
                # Agent autonomously decides keywords from resume
                keywords = self._extract_keywords_from_resume(resume_text)

                # DECISION 3: Fetch new jobs from Adzuna
                # Agent uses external API tool
                job_stats = self._fetch_new_jobs(keywords, config)
                run_history.jobs_fetched = job_stats['stored']

                # DECISION 4: Rebuild index if new jobs were added
                if job_stats['stored'] > 0:
                    build_job_faiss_index()

                # DECISION 5: Find matching jobs using FAISS + LLM
                # Agent analyzes matches autonomously
                matches = self._find_and_save_matches(
                    user_id=user_id,
                    resume_id=latest_resume.id,
                    resume_text=resume_text,
                    resume_filename=latest_resume.original_filename,
                    threshold=config.match_threshold,
                    max_results=config.max_results_per_run,
                    run_history_id=run_history.id
                )

                run_history.jobs_analyzed = len(matches['analyzed'])
                run_history.matches_found = len(matches['saved'])

                # Update config
                config.last_run_at = datetime.utcnow()

                # Complete run
                run_history.status = 'completed'
                run_history.completed_at = datetime.utcnow()
                run_history.results_summary = json.dumps({
                    'jobs_fetched': job_stats['stored'],
                    'jobs_analyzed': len(matches['analyzed']),
                    'matches_found': len(matches['saved']),
                    'top_match_score': matches['saved'][0]['match_score'] if matches['saved'] else 0,
                    'keywords_used': keywords
                })

                db.session.commit()

                return {
                    'status': 'success',
                    'run_id': run_history.id,
                    'jobs_fetched': job_stats['stored'],
                    'jobs_analyzed': len(matches['analyzed']),
                    'matches_found': len(matches['saved']),
                    'matches': matches['saved']
                }

            except Exception as e:
                # Handle errors
                run_history.status = 'failed'
                run_history.completed_at = datetime.utcnow()
                run_history.error_message = str(e)
                db.session.commit()

                return {
                    'status': 'failed',
                    'error': str(e),
                    'run_id': run_history.id
                }

    def _extract_keywords_from_resume(self, resume_text):
        """
        AUTONOMOUS DECISION: Extract job search keywords from resume

        Agent analyzes resume and decides what jobs to search for
        """
        try:
            # Use LLM to extract key job titles/roles from resume
            prompt = f"""Analyze this resume and extract 2-3 key job titles or roles the person is qualified for.
Return only the job titles, comma-separated, no extra text.
 
Resume:
{resume_text[:1500]}
 
Job titles:"""

            response = self.llm.invoke(prompt)
            keywords = response.content.strip()

            # If LLM fails, fall back to generic search
            if not keywords or len(keywords) > 100:
                keywords = "software engineer"

            return keywords

        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return "software engineer"  # Safe fallback

    def _fetch_new_jobs(self, keywords, config):
        """
        TOOL USE: Fetch new jobs from external API

        Agent uses Adzuna API to get fresh job postings
        """
        try:
            fetcher = AdzunaJobFetcher()

            # Fetch recent jobs (last 7 days)
            stats = fetcher.fetch_and_store_jobs(
                keywords=keywords,
                location=None,  # Can be configured later
                max_jobs=20,  # Reasonable amount for daily checks
                max_days_old=7,
                skip_duplicates=True
            )

            return stats

        except Exception as e:
            print(f"Error fetching jobs: {e}")
            return {
                'fetched': 0,
                'stored': 0,
                'duplicates': 0,
                'errors': 1,
                'error_messages': [str(e)]
            }

    def _find_and_save_matches(self, user_id, resume_id, resume_text, resume_filename,
                               threshold, max_results, run_history_id):
        """
        AUTONOMOUS ANALYSIS: Find matches and decide which to save

        Agent analyzes jobs, evaluates matches, and decides which are worth showing to user
        """
        matches_analyzed = []
        matches_saved = []

        try:
            # Get job index
            job_index = get_job_faiss_index()
            if job_index is None:
                return {'analyzed': [], 'saved': []}

            # Search for similar jobs using FAISS (Stage 1: Fast retrieval)
            docs_with_scores = job_index.similarity_search_with_score(
                resume_text,
                k=min(20, job_index.index.ntotal)  # Get top 20 candidates
            )

            if not docs_with_scores:
                return {'analyzed': [], 'saved': []}

            # Stage 2: Detailed LLM analysis for candidates
            for doc, distance in docs_with_scores[:max_results * 2]:  # Analyze more than we need
                job_id = doc.metadata.get("job_id")
                job = JobPosting.query.get(job_id)

                if not job or not job.is_active:
                    continue

                # Check if user already has this match
                existing_match = JobMatch.query.filter_by(
                    user_id=user_id,
                    job_id=job_id,
                    resume_id=resume_id
                ).first()

                if existing_match:
                    continue  # Skip duplicates

                # AUTONOMOUS DECISION: Analyze job match using LLM
                try:
                    analysis_result = self.llm.invoke(
                        self.matching_prompt.format(
                            resume=resume_text[:3000],
                            job_title=job.title,
                            company=job.company,
                            job_description=job.description[:1000],
                            job_requirements=job.requirements[:1000] if job.requirements else "Not specified"
                        )
                    )

                    # Parse JSON response
                    analysis = json.loads(analysis_result.content)

                    matches_analyzed.append({
                        'job_id': job.id,
                        'match_score': analysis['match_score']
                    })

                    # AUTONOMOUS DECISION: Only save matches above threshold
                    if analysis['match_score'] >= threshold:
                        # Save match to database
                        job_match = JobMatch(
                            user_id=user_id,
                            resume_id=resume_id,
                            resume_filename=resume_filename,
                            job_id=job.id,
                            match_score=analysis['match_score'],
                            matched_skills=json.dumps(analysis.get('matched_skills', [])),
                            gaps=json.dumps(analysis.get('skill_gaps', [])),
                            recommendation=analysis.get('recommendation', ''),
                            agent_generated=True,
                            agent_run_id=run_history_id,
                            created_at=datetime.utcnow()
                        )

                        db.session.add(job_match)
                        matches_saved.append({
                            'job_id': job.id,
                            'job_title': job.title,
                            'company': job.company,
                            'match_score': analysis['match_score'],
                            'matched_skills': analysis.get('matched_skills', []),
                            'skill_gaps': analysis.get('skill_gaps', [])
                        })

                        # Stop if we have enough good matches
                        if len(matches_saved) >= max_results:
                            break

                except json.JSONDecodeError:
                    print(f"Failed to parse LLM response for job {job.id}")
                    continue
                except Exception as e:
                    print(f"Error analyzing job {job.id}: {e}")
                    continue

            db.session.commit()

        except Exception as e:
            print(f"Error in find_and_save_matches: {e}")
            db.session.rollback()

        return {
            'analyzed': matches_analyzed,
            'saved': matches_saved
        }

    def run_for_all_users(self, run_type='scheduled'):
        """
        Run agent for all users with enabled agents

        This would be called by the scheduler daily
        """
        with self.app_context:
            # Get all users with enabled agents
            configs = AgentConfig.query.filter_by(is_enabled=True).all()

            results = []
            for config in configs:
                result = self.run_for_user(config.user_id, run_type=run_type)
                results.append({
                    'user_id': config.user_id,
                    'result': result
                })

            return results