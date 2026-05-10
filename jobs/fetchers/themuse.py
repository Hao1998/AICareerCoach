"""The Muse fetcher — free, no API key for basic access. Full job descriptions + company info."""

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


class TheMuseFetcher(BaseJobFetcher):
    source_name = "themuse"

    BASE_URL = "https://www.themuse.com/api/public/jobs"

    def fetch_jobs(self, keywords=None, location=None, max_jobs=50, **kwargs) -> list[dict]:
        all_jobs = []
        page = 0

        while len(all_jobs) < max_jobs:
            params = {'page': page}
            if kwargs.get('category'):
                params['category'] = kwargs['category']
            if kwargs.get('level'):
                params['level'] = kwargs['level']
            if location:
                params['location'] = location

            response = _request_with_retry('GET', self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            results = data.get('results', [])
            if not results:
                break

            all_jobs.extend(results)
            page += 1

            if page >= data.get('page_count', 1):
                break

        return all_jobs[:max_jobs]

    def parse_job(self, raw: dict) -> dict:
        locations = raw.get('locations') or []
        location_names = [loc.get('name', '') for loc in locations if loc.get('name')]
        location = ', '.join(location_names) if location_names else 'Not specified'

        company = raw.get('company', {}).get('name', 'Unknown Company')

        levels = raw.get('levels') or []
        level_names = [lv.get('name', '') for lv in levels]

        categories = raw.get('categories') or []
        category_names = [cat.get('name', '') for cat in categories]

        description = _strip_html(raw.get('contents', ''))
        if level_names:
            description += f"\n\nLevel: {', '.join(level_names)}"
        if category_names:
            description += f"\nCategory: {', '.join(category_names)}"

        return {
            'title': raw.get('name', 'Untitled Position'),
            'company': company,
            'location': location,
            'job_type': 'Full-time',
            'description': description,
            'requirements': None,
            'salary_range': None,
            'source': 'themuse',
            'source_id': raw.get('id'),
            'source_url': raw.get('refs', {}).get('landing_page'),
        }
