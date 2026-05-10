"""
Abstract base class for all job fetchers.

Subclasses implement fetch_jobs() and parse_job(); the shared
store/dedup/embed/index logic lives here so every source gets it for free.
"""

import time
import logging
from abc import ABC, abstractmethod

import requests

from models import db, JobPosting
from services.db_lock import safe_commit
from jobs.utils import compute_job_embedding, build_job_faiss_index

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 15
MAX_RETRIES = 2
RETRY_BACKOFF = 1.5


def _request_with_retry(method, url, **kwargs):
    """HTTP request with retry on 429/5xx, respecting timeouts."""
    kwargs.setdefault('timeout', REQUEST_TIMEOUT)
    last_exc = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.request(method, url, **kwargs)

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "%s %s returned %s, retrying in %.1fs (attempt %d/%d)",
                        method.upper(), url, resp.status_code, wait,
                        attempt + 1, MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()

            resp.raise_for_status()
            return resp

        except requests.exceptions.Timeout as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * (2 ** attempt)
                logger.warning(
                    "%s %s timed out, retrying in %.1fs",
                    method.upper(), url, wait,
                )
                time.sleep(wait)
                continue
            raise

        except requests.exceptions.RequestException:
            raise

    raise last_exc


class BaseJobFetcher(ABC):
    source_name: str = ""

    @abstractmethod
    def fetch_jobs(self, keywords=None, location=None, max_jobs=50, **kwargs) -> list[dict]:
        """Hit the external API and return a list of raw response dicts."""

    @abstractmethod
    def parse_job(self, raw: dict) -> dict:
        """Normalise one raw API result to the standard schema:
        {title, company, location, job_type, description,
         requirements, salary_range, source, source_id, source_url}
        """

    def fetch_and_store_jobs(self, keywords=None, location=None,
                             max_jobs=50, skip_duplicates=True, **kwargs) -> dict:
        stats = {
            'fetched': 0,
            'stored': 0,
            'duplicates': 0,
            'errors': 0,
            'error_messages': [],
        }

        try:
            raw_jobs = self.fetch_jobs(
                keywords=keywords, location=location,
                max_jobs=max_jobs, **kwargs,
            )
            stats['fetched'] = len(raw_jobs)

            for raw in raw_jobs:
                if stats['stored'] >= max_jobs:
                    break
                try:
                    parsed = self.parse_job(raw)

                    if skip_duplicates and parsed.get('source_id'):
                        existing = JobPosting.query.filter_by(
                            source=parsed['source'],
                            source_id=str(parsed['source_id']),
                        ).first()
                        if existing:
                            stats['duplicates'] += 1
                            continue
                    elif skip_duplicates:
                        existing = JobPosting.query.filter_by(
                            title=parsed['title'],
                            company=parsed['company'],
                        ).first()
                        if existing:
                            stats['duplicates'] += 1
                            continue

                    job = JobPosting(
                        title=parsed['title'],
                        company=parsed['company'],
                        location=parsed.get('location'),
                        job_type=parsed.get('job_type'),
                        description=parsed['description'],
                        requirements=parsed.get('requirements'),
                        salary_range=parsed.get('salary_range'),
                        source=parsed.get('source', self.source_name),
                        source_id=str(parsed['source_id']) if parsed.get('source_id') else None,
                        source_url=parsed.get('source_url'),
                    )

                    compute_job_embedding(job)
                    db.session.add(job)
                    stats['stored'] += 1

                except Exception as e:
                    stats['errors'] += 1
                    stats['error_messages'].append(f"Error parsing job: {e}")

            if stats['stored'] > 0:
                safe_commit()
                try:
                    build_job_faiss_index()
                except Exception as e:
                    stats['error_messages'].append(f"Error rebuilding FAISS index: {e}")

        except Exception as e:
            stats['errors'] += 1
            stats['error_messages'].append(f"Fetch error ({self.source_name}): {e}")
            db.session.rollback()

        return stats
