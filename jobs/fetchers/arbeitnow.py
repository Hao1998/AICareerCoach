"""Arbeitnow fetcher — free, no API key. ATS-sourced, Europe-focused with remote flag."""

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


class ArbeitnowFetcher(BaseJobFetcher):
    source_name = "arbeitnow"

    BASE_URL = "https://arbeitnow.com/api/job-board-api"

    def fetch_jobs(self, keywords=None, location=None, max_jobs=50, **kwargs) -> list[dict]:
        all_jobs = []
        page = 1

        while len(all_jobs) < max_jobs:
            params = {'page': page}
            response = _request_with_retry('GET', self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            jobs = data.get('data', [])
            if not jobs:
                break

            if keywords:
                kw_lower = keywords.lower()
                jobs = [
                    j for j in jobs
                    if kw_lower in (j.get('title', '') + ' ' + j.get('description', '')).lower()
                ]

            all_jobs.extend(jobs)
            page += 1

            if not data.get('links', {}).get('next'):
                break

        return all_jobs[:max_jobs]

    def parse_job(self, raw: dict) -> dict:
        tags = raw.get('tags') or []
        is_remote = raw.get('remote', False)
        location = raw.get('location') or ''
        if is_remote and 'remote' not in location.lower():
            location = f"{location} (Remote)".strip(' ()')

        description = _strip_html(raw.get('description', ''))
        if tags:
            description += f"\n\nTags: {', '.join(tags)}"

        return {
            'title': raw.get('title', 'Untitled Position'),
            'company': raw.get('company_name', 'Unknown Company'),
            'location': location or 'Not specified',
            'job_type': 'Full-time',
            'description': description,
            'requirements': None,
            'salary_range': None,
            'source': 'arbeitnow',
            'source_id': raw.get('slug'),
            'source_url': raw.get('url'),
        }
