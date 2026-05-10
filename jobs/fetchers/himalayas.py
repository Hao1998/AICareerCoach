"""Himalayas fetcher — free, no API key. Remote jobs with salary and tech stack."""

from jobs.fetchers.base import BaseJobFetcher, _request_with_retry


class HimalayasFetcher(BaseJobFetcher):
    source_name = "himalayas"

    BASE_URL = "https://himalayas.app/jobs/api"

    def fetch_jobs(self, keywords=None, location=None, max_jobs=50, **kwargs) -> list[dict]:
        params = {'limit': max_jobs}
        if keywords:
            params['q'] = keywords

        response = _request_with_retry('GET', self.BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get('jobs', data if isinstance(data, list) else [])[:max_jobs]

    def parse_job(self, raw: dict) -> dict:
        salary_min = raw.get('salaryCurrencyMin') or raw.get('salaryMin')
        salary_max = raw.get('salaryCurrencyMax') or raw.get('salaryMax')
        salary_range = None
        if salary_min and salary_max:
            salary_range = f"${salary_min:,} - ${salary_max:,}"
        elif salary_min:
            salary_range = f"From ${salary_min:,}"
        elif salary_max:
            salary_range = f"Up to ${salary_max:,}"

        skills = raw.get('skills') or raw.get('techStack') or []
        description = raw.get('description', '')
        if skills:
            skill_names = [s if isinstance(s, str) else s.get('name', '') for s in skills]
            description += f"\n\nTech Stack: {', '.join(skill_names)}"

        return {
            'title': raw.get('title', 'Untitled Position'),
            'company': raw.get('companyName') or raw.get('company', 'Unknown Company'),
            'location': raw.get('location') or 'Remote',
            'job_type': raw.get('type') or 'Full-time',
            'description': description,
            'requirements': None,
            'salary_range': salary_range,
            'source': 'himalayas',
            'source_id': raw.get('id') or raw.get('slug'),
            'source_url': raw.get('applicationUrl') or raw.get('url'),
        }
