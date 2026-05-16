"""Adzuna job fetcher — requires ADZUNA_APP_ID and ADZUNA_APP_KEY env vars."""

import os

from jobs.fetchers.base import BaseJobFetcher, _request_with_retry


class AdzunaJobFetcher(BaseJobFetcher):
    source_name = "adzuna"

    BASE_URL = "https://api.adzuna.com/v1/api/jobs"

    def __init__(self, app_id=None, app_key=None, country='us'):
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

    def fetch_jobs(self, keywords=None, location=None, max_jobs=50, **kwargs) -> list[dict]:
        max_days_old = kwargs.get('max_days_old', 30)
        results_per_page = min(max_jobs, 50)
        pages_needed = (max_jobs + results_per_page - 1) // results_per_page
        all_jobs: list[dict] = []

        for page in range(1, pages_needed + 1):
            if len(all_jobs) >= max_jobs:
                break

            url = f"{self.BASE_URL}/{self.country}/search/{page}"
            params = {
                'app_id': self.app_id,
                'app_key': self.app_key,
                'results_per_page': results_per_page,
                'sort_by': 'relevance',
            }
            if keywords:
                params['what'] = keywords
            if location:
                params['where'] = location
            if max_days_old:
                params['max_days_old'] = max_days_old

            response = _request_with_retry('GET', url, params=params)
            response.raise_for_status()
            data = response.json()
            all_jobs.extend(data.get('results', []))

        return all_jobs[:max_jobs]

    def parse_job(self, raw: dict) -> dict:
        salary_min = raw.get('salary_min')
        salary_max = raw.get('salary_max')
        salary_range = None
        if salary_min and salary_max:
            salary_range = f"${salary_min:,.0f} - ${salary_max:,.0f}"
        elif salary_min:
            salary_range = f"From ${salary_min:,.0f}"
        elif salary_max:
            salary_range = f"Up to ${salary_max:,.0f}"

        contract_time = raw.get('contract_time', '').lower()
        job_type = 'Full-time'
        if 'part' in contract_time:
            job_type = 'Part-time'
        elif 'contract' in contract_time:
            job_type = 'Contract'

        description_parts = []
        if raw.get('description'):
            description_parts.append(raw['description'])
        if raw.get('category', {}).get('label'):
            description_parts.append(f"\n\nCategory: {raw['category']['label']}")
        if raw.get('contract_type'):
            description_parts.append(f"Contract Type: {raw['contract_type']}")

        location_parts = []
        if raw.get('location', {}).get('display_name'):
            location_parts.append(raw['location']['display_name'])
        if raw.get('location', {}).get('area'):
            for area in raw['location']['area']:
                location_parts.append(area)

        return {
            'title': raw.get('title', 'Untitled Position'),
            'company': raw.get('company', {}).get('display_name', 'Unknown Company'),
            'location': ', '.join(location_parts[:3]) if location_parts else 'Not specified',
            'job_type': job_type,
            'description': '\n'.join(description_parts),
            'requirements': None,
            'salary_range': salary_range,
            'source': 'adzuna',
            'source_id': raw.get('id'),
            'source_url': raw.get('redirect_url'),
        }
