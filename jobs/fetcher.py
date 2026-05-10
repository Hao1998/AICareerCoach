"""
Job Fetcher Service — backward-compatible shim.

The real implementations now live in jobs.fetchers.*
This file re-exports AdzunaJobFetcher and the convenience function
so existing imports (scout_agent, job_service, etc.) keep working.
"""

from jobs.fetchers.adzuna import AdzunaJobFetcher


def fetch_jobs_from_adzuna(keywords=None, location=None, max_jobs=50, max_days_old=30):
    try:
        fetcher = AdzunaJobFetcher()
        return fetcher.fetch_and_store_jobs(
            keywords=keywords,
            location=location,
            max_jobs=max_jobs,
            max_days_old=max_days_old,
        )
    except Exception as e:
        return {
            'fetched': 0,
            'stored': 0,
            'duplicates': 0,
            'errors': 1,
            'error_messages': [str(e)],
        }
