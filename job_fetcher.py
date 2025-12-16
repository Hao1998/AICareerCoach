"""
Job Fetcher Service - Fetches jobs from external APIs (Adzuna)
"""
import requests
import os
from datetime import datetime
from models import db, JobPosting
from job_utils import compute_job_embedding, build_job_faiss_index

class AdzunaJobFetcher:
    """Fetch jobs from Adzuna API"""

    BASE_URL = "https://api.adzuna.com/v1/api/jobs"

    def __init__(self, app_id=None, app_key=None, country='us'):
        """
        Initialize the Adzuna job fetcher

        Args:
            app_id: Adzuna API app ID (get from https://developer.adzuna.com/)
            app_key: Adzuna API app key
            country: Country code (us, gb, ca, au, etc.)
        """
        self.app_id = app_id or os.getenv('ADZUNA_APP_ID')
        self.app_key = app_key or os.getenv('ADZUNA_APP_KEY')
        self.country = country

        if not self.app_id or not self.app_key:
            raise ValueError(
                "Adzuna API credentials required. "
                "Set ADZUNA_APP_ID and ADZUNA_APP_KEY environment variables "
                "or pass them to the constructor. "
                "Get credentials at https://developer.adzuna.com/"
            )

    def search_jobs(self, keywords=None, location=None, results_per_page=50,
                   page=1, sort_by='relevance', max_days_old=30):
        """
        Search for jobs on Adzuna

        Args:
            keywords: Job search keywords (e.g., "python developer")
            location: Location (e.g., "San Francisco", "Remote")
            results_per_page: Number of results per page (max 50)
            page: Page number
            sort_by: Sort order (relevance, date, salary)
            max_days_old: Only return jobs posted within this many days

        Returns:
            dict: API response with job listings
        """
        url = f"{self.BASE_URL}/{self.country}/search/{page}"

        params = {
            'app_id': self.app_id,
            'app_key': self.app_key,
            'results_per_page': min(results_per_page, 50),  # Max 50
            'sort_by': sort_by
        }

        if keywords:
            params['what'] = keywords

        if location:
            params['where'] = location

        if max_days_old:
            params['max_days_old'] = max_days_old

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch jobs from Adzuna: {str(e)}")

    def parse_job(self, job_data):
        """
        Parse Adzuna job data to our JobPosting format

        Args:
            job_data: Job data from Adzuna API

        Returns:
            dict: Parsed job data ready for JobPosting model
        """
        # Extract salary range
        salary_min = job_data.get('salary_min')
        salary_max = job_data.get('salary_max')
        salary_range = None

        if salary_min and salary_max:
            salary_range = f"${salary_min:,.0f} - ${salary_max:,.0f}"
        elif salary_min:
            salary_range = f"From ${salary_min:,.0f}"
        elif salary_max:
            salary_range = f"Up to ${salary_max:,.0f}"

        # Determine job type from contract time
        contract_time = job_data.get('contract_time', '').lower()
        job_type = 'Full-time'  # Default
        if 'part' in contract_time:
            job_type = 'Part-time'
        elif 'contract' in contract_time:
            job_type = 'Contract'

        # Build comprehensive description
        description_parts = []

        if job_data.get('description'):
            description_parts.append(job_data['description'])

        # Add additional metadata as context
        if job_data.get('category', {}).get('label'):
            description_parts.append(f"\n\nCategory: {job_data['category']['label']}")

        if job_data.get('contract_type'):
            description_parts.append(f"Contract Type: {job_data['contract_type']}")

        description = '\n'.join(description_parts)

        # Location
        location_parts = []
        if job_data.get('location', {}).get('display_name'):
            location_parts.append(job_data['location']['display_name'])
        if job_data.get('location', {}).get('area'):
            for area in job_data['location']['area']:
                location_parts.append(area)

        location = ', '.join(location_parts[:3]) if location_parts else 'Not specified'

        return {
            'title': job_data.get('title', 'Untitled Position'),
            'company': job_data.get('company', {}).get('display_name', 'Unknown Company'),
            'location': location,
            'job_type': job_type,
            'description': description,
            'requirements': None,  # Adzuna doesn't separate requirements
            'salary_range': salary_range,
            'source_url': job_data.get('redirect_url'),
            'source_id': job_data.get('id'),
            'source': 'adzuna'
        }

    def fetch_and_store_jobs(self, keywords=None, location=None,
                            max_jobs=50, max_days_old=30,
                            skip_duplicates=True):
        """
        Fetch jobs from Adzuna and store them in the database

        Args:
            keywords: Job search keywords
            location: Location filter
            max_jobs: Maximum number of jobs to fetch
            max_days_old: Only fetch jobs posted within this many days
            skip_duplicates: Skip jobs that already exist in database

        Returns:
            dict: Statistics about the fetch operation
        """
        stats = {
            'fetched': 0,
            'stored': 0,
            'duplicates': 0,
            'errors': 0,
            'error_messages': []
        }

        try:
            # Calculate number of pages needed
            results_per_page = min(max_jobs, 50)
            pages_needed = (max_jobs + results_per_page - 1) // results_per_page

            jobs_stored = 0

            for page in range(1, pages_needed + 1):
                if jobs_stored >= max_jobs:
                    break

                # Fetch jobs from Adzuna
                response = self.search_jobs(
                    keywords=keywords,
                    location=location,
                    results_per_page=results_per_page,
                    page=page,
                    max_days_old=max_days_old
                )

                jobs = response.get('results', [])
                stats['fetched'] += len(jobs)

                for job_data in jobs:
                    if jobs_stored >= max_jobs:
                        break

                    try:
                        parsed_job = self.parse_job(job_data)

                        # Check for duplicates by title and company
                        if skip_duplicates:
                            existing = JobPosting.query.filter_by(
                                title=parsed_job['title'],
                                company=parsed_job['company']
                            ).first()

                            if existing:
                                stats['duplicates'] += 1
                                continue

                        # Create and store job
                        job = JobPosting(
                            title=parsed_job['title'],
                            company=parsed_job['company'],
                            location=parsed_job['location'],
                            job_type=parsed_job['job_type'],
                            description=parsed_job['description'],
                            requirements=parsed_job['requirements'],
                            salary_range=parsed_job['salary_range']
                        )

                        # Compute embedding
                        compute_job_embedding(job)

                        db.session.add(job)
                        stats['stored'] += 1
                        jobs_stored += 1

                    except Exception as e:
                        stats['errors'] += 1
                        stats['error_messages'].append(f"Error parsing job: {str(e)}")

                # Commit after each page
                if stats['stored'] > 0:
                    db.session.commit()

            # Rebuild FAISS index after all jobs are stored
            if stats['stored'] > 0:
                try:
                    build_job_faiss_index()
                except Exception as e:
                    stats['error_messages'].append(f"Error rebuilding FAISS index: {str(e)}")

        except Exception as e:
            stats['errors'] += 1
            stats['error_messages'].append(f"Fetch error: {str(e)}")
            db.session.rollback()

        return stats


def fetch_jobs_from_adzuna(keywords=None, location=None, max_jobs=50, max_days_old=30):
    """
    Convenience function to fetch jobs from Adzuna

    Args:
        keywords: Job search keywords (e.g., "python developer")
        location: Location (e.g., "San Francisco", "Remote")
        max_jobs: Maximum number of jobs to fetch
        max_days_old: Only return jobs posted within this many days

    Returns:
        dict: Statistics about the fetch operation
    """
    try:
        fetcher = AdzunaJobFetcher()
        return fetcher.fetch_and_store_jobs(
            keywords=keywords,
            location=location,
            max_jobs=max_jobs,
            max_days_old=max_days_old
        )
    except Exception as e:
        return {
            'fetched': 0,
            'stored': 0,
            'duplicates': 0,
            'errors': 1,
            'error_messages': [str(e)]
        }
