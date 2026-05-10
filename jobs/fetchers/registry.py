"""
Fetcher registry — maps source names to fetcher classes and provides
a single entry point for multi-source job fetching.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from jobs.fetchers.adzuna import AdzunaJobFetcher
from jobs.fetchers.remotive import RemotiveFetcher
from jobs.fetchers.jobicy import JobicyFetcher
from jobs.fetchers.remoteok import RemoteOKFetcher
from jobs.fetchers.himalayas import HimalayasFetcher
from jobs.fetchers.themuse import TheMuseFetcher
from jobs.fetchers.arbeitnow import ArbeitnowFetcher
from jobs.fetchers.greenhouse import GreenhouseFetcher

logger = logging.getLogger(__name__)

FETCHER_REGISTRY = {
    'adzuna': AdzunaJobFetcher,
    'remotive': RemotiveFetcher,
    'jobicy': JobicyFetcher,
    'remoteok': RemoteOKFetcher,
    'himalayas': HimalayasFetcher,
    'themuse': TheMuseFetcher,
    'arbeitnow': ArbeitnowFetcher,
    'greenhouse': GreenhouseFetcher,
}

# Sources shown in UI checkboxes (greenhouse excluded — per-company, needs special config)
USER_VISIBLE_SOURCES = ['adzuna', 'remotive', 'jobicy', 'remoteok', 'himalayas', 'themuse', 'arbeitnow']
ALL_SOURCES = list(FETCHER_REGISTRY.keys())


def _run_one_fetcher(source_name, fetcher_cls, keywords, location, max_jobs, extra_kwargs):
    """Run a single fetcher, return (source_name, stats_dict)."""
    try:
        if source_name == 'adzuna':
            fetcher = fetcher_cls()
        elif source_name == 'greenhouse':
            fetcher = fetcher_cls(board_tokens=extra_kwargs.get('board_tokens', []))
        else:
            fetcher = fetcher_cls()

        kwargs = dict(extra_kwargs)
        kwargs.pop('board_tokens', None)

        stats = fetcher.fetch_and_store_jobs(
            keywords=keywords,
            location=location,
            max_jobs=max_jobs,
            **kwargs,
        )
        return source_name, stats

    except Exception as e:
        logger.error("Fetcher %s failed: %s", source_name, e)
        return source_name, {
            'fetched': 0,
            'stored': 0,
            'duplicates': 0,
            'errors': 1,
            'error_messages': [str(e)],
        }


def fetch_from_sources(sources, keywords=None, location=None,
                       max_jobs_per_source=50, **kwargs):
    """
    Run multiple fetchers and aggregate stats.

    Each fetcher stores its own jobs. FAISS index is rebuilt once in
    each fetcher's base class (after its batch), so the index stays
    current even if one source fails.

    Returns:
        dict: {
            'by_source': {source_name: stats_dict, ...},
            'totals': {fetched, stored, duplicates, errors},
        }
    """
    results = {}

    for source in sources:
        if source not in FETCHER_REGISTRY:
            logger.warning("Unknown source %s, skipping", source)
            continue

        fetcher_cls = FETCHER_REGISTRY[source]
        source_name, stats = _run_one_fetcher(
            source, fetcher_cls, keywords, location,
            max_jobs_per_source, kwargs,
        )
        results[source_name] = stats

    totals = {'fetched': 0, 'stored': 0, 'duplicates': 0, 'errors': 0, 'error_messages': []}
    for stats in results.values():
        totals['fetched'] += stats.get('fetched', 0)
        totals['stored'] += stats.get('stored', 0)
        totals['duplicates'] += stats.get('duplicates', 0)
        totals['errors'] += stats.get('errors', 0)
        totals['error_messages'].extend(stats.get('error_messages', []))

    return {'by_source': results, 'totals': totals}
