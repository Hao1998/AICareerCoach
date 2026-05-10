"""RemoteOK fetcher — free, no API key. Must set User-Agent header."""

from jobs.fetchers.base import BaseJobFetcher, _request_with_retry


class RemoteOKFetcher(BaseJobFetcher):
    source_name = "remoteok"

    BASE_URL = "https://remoteok.com/api"

    def fetch_jobs(self, keywords=None, location=None, max_jobs=50, **kwargs) -> list[dict]:
        headers = {
            'User-Agent': 'AiCareerCoach/1.0 (job aggregator)',
        }
        params = {}
        if keywords:
            params['tags'] = keywords

        response = _request_with_retry('GET', self.BASE_URL, params=params,
                                       headers=headers)
        response.raise_for_status()
        data = response.json()

        # First element is metadata/legal notice, skip it
        jobs = [item for item in data if isinstance(item, dict) and item.get('position')]
        return jobs[:max_jobs]

    def parse_job(self, raw: dict) -> dict:
        salary = raw.get('salary') or ''
        salary_range = salary.strip() if salary.strip() else None

        tags = raw.get('tags') or []
        description = raw.get('description', '')
        if tags:
            description += f"\n\nTags: {', '.join(tags)}"

        return {
            'title': raw.get('position', 'Untitled Position'),
            'company': raw.get('company', 'Unknown Company'),
            'location': raw.get('location') or 'Remote',
            'job_type': 'Full-time',
            'description': description,
            'requirements': None,
            'salary_range': salary_range,
            'source': 'remoteok',
            'source_id': raw.get('id'),
            'source_url': raw.get('url') or raw.get('apply_url'),
        }
