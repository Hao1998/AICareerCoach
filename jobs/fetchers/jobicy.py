"""Jobicy fetcher — free, no API key. Remote jobs with salary and geo filters."""

from jobs.fetchers.base import BaseJobFetcher, _request_with_retry


class JobicyFetcher(BaseJobFetcher):
    source_name = "jobicy"

    BASE_URL = "https://jobicy.com/api/v2/remote-jobs"

    def fetch_jobs(self, keywords=None, location=None, max_jobs=50, **kwargs) -> list[dict]:
        params = {'count': min(max_jobs, 50)}
        if kwargs.get('geo'):
            params['geo'] = kwargs['geo']
        if kwargs.get('industry'):
            params['industry'] = kwargs['industry']
        if keywords:
            params['tag'] = keywords

        response = _request_with_retry('GET', self.BASE_URL, params=params)
        response.raise_for_status()
        return response.json().get('jobs', [])[:max_jobs]

    def parse_job(self, raw: dict) -> dict:
        try:
            salary_min = int(raw['annualSalaryMin']) if raw.get('annualSalaryMin') else None
        except (ValueError, TypeError):
            salary_min = None
        try:
            salary_max = int(raw['annualSalaryMax']) if raw.get('annualSalaryMax') else None
        except (ValueError, TypeError):
            salary_max = None
        salary_range = None
        if salary_min and salary_max:
            salary_range = f"${salary_min:,} - ${salary_max:,}"
        elif salary_min:
            salary_range = f"From ${salary_min:,}"
        elif salary_max:
            salary_range = f"Up to ${salary_max:,}"

        jt = raw.get('jobType') or ''
        job_type_raw = (', '.join(jt) if isinstance(jt, list) else str(jt)).lower()
        if 'part' in job_type_raw:
            job_type = 'Part-time'
        elif 'contract' in job_type_raw or 'freelance' in job_type_raw:
            job_type = 'Contract'
        else:
            job_type = 'Full-time'

        return {
            'title': raw.get('jobTitle', 'Untitled Position'),
            'company': raw.get('companyName', 'Unknown Company'),
            'location': raw.get('jobGeo') or 'Remote',
            'job_type': job_type,
            'description': raw.get('jobDescription', ''),
            'requirements': None,
            'salary_range': salary_range,
            'source': 'jobicy',
            'source_id': raw.get('id'),
            'source_url': raw.get('url'),
        }
