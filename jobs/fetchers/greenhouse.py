"""
Greenhouse fetcher — free, no API key for GET endpoints.

Unlike other fetchers, Greenhouse is per-company (requires board_token).
Not exposed in the UI — available for programmatic use only.
"""

import re

from jobs.fetchers.base import BaseJobFetcher, _request_with_retry


def _strip_html(html: str) -> str:
    text = re.sub(r'<br\s*/?>', '\n', html)
    text = re.sub(r'<li>', '\n- ', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&#?\w+;', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


class GreenhouseFetcher(BaseJobFetcher):
    source_name = "greenhouse"

    BASE_URL = "https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs"

    def __init__(self, board_tokens=None):
        self.board_tokens = board_tokens or []

    def fetch_jobs(self, keywords=None, location=None, max_jobs=50, **kwargs) -> list[dict]:
        board_tokens = kwargs.get('board_tokens') or self.board_tokens
        if not board_tokens:
            return []

        all_jobs = []
        for token in board_tokens:
            if len(all_jobs) >= max_jobs:
                break
            try:
                url = self.BASE_URL.format(board_token=token)
                params = {'content': 'true'}
                response = _request_with_retry('GET', url, params=params)
                response.raise_for_status()
                data = response.json()

                jobs = data.get('jobs', [])
                for job in jobs:
                    job['_board_token'] = token
                all_jobs.extend(jobs)

            except Exception:
                continue

        if keywords:
            kw_lower = keywords.lower()
            all_jobs = [
                j for j in all_jobs
                if kw_lower in (j.get('title', '') + ' ' + _strip_html(j.get('content', ''))).lower()
            ]

        return all_jobs[:max_jobs]

    def parse_job(self, raw: dict) -> dict:
        location_data = raw.get('location', {})
        location = location_data.get('name') if isinstance(location_data, dict) else str(location_data or '')

        description = _strip_html(raw.get('content', ''))

        departments = raw.get('departments', [])
        if departments:
            dept_names = [d.get('name', '') for d in departments if d.get('name')]
            if dept_names:
                description += f"\n\nDepartment: {', '.join(dept_names)}"

        board_token = raw.get('_board_token', '')

        return {
            'title': raw.get('title', 'Untitled Position'),
            'company': board_token.replace('-', ' ').title() if board_token else 'Unknown Company',
            'location': location or 'Not specified',
            'job_type': 'Full-time',
            'description': description,
            'requirements': None,
            'salary_range': None,
            'source': 'greenhouse',
            'source_id': raw.get('id'),
            'source_url': raw.get('absolute_url'),
        }
