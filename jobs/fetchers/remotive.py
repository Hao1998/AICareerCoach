"""Remotive fetcher — free, no API key. Must credit Remotive as source."""

from jobs.fetchers.base import BaseJobFetcher, _request_with_retry


class RemotiveFetcher(BaseJobFetcher):
    source_name = "remotive"

    BASE_URL = "https://remotive.com/api/remote-jobs"

    def fetch_jobs(self, keywords=None, location=None, max_jobs=50, **kwargs) -> list[dict]:
        params = {'limit': max_jobs}
        if keywords:
            params['search'] = keywords
        category = kwargs.get('category')
        if category:
            params['category'] = category

        response = _request_with_retry('GET', self.BASE_URL, params=params)
        response.raise_for_status()
        return response.json().get('jobs', [])[:max_jobs]

    def parse_job(self, raw: dict) -> dict:
        salary = raw.get('salary') or ''
        salary_range = salary.strip() if salary.strip() else None

        job_type_raw = (raw.get('job_type') or '').lower()
        if 'part' in job_type_raw:
            job_type = 'Part-time'
        elif 'contract' in job_type_raw or 'freelance' in job_type_raw:
            job_type = 'Contract'
        else:
            job_type = 'Full-time'

        description = raw.get('description', '')
        description += "\n\n[Powered by Remotive]"

        return {
            'title': raw.get('title', 'Untitled Position'),
            'company': raw.get('company_name', 'Unknown Company'),
            'location': raw.get('candidate_required_location') or 'Remote',
            'job_type': job_type,
            'description': description,
            'requirements': None,
            'salary_range': salary_range,
            'source': 'remotive',
            'source_id': raw.get('id'),
            'source_url': raw.get('url'),
        }
