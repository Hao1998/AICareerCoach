"""
`flask --app wsgi redis-vectors ...` — drive the Redis vector search experiment.

    rebuild   drop idx:jobs and reload every active, embedded job from the DB
    relay     apply vector_outbox rows to Redis (--watch to keep running)
    search    embed a query and print the nearest jobs straight from Redis

All commands require VECTOR_BACKEND=redis.
"""
import sys
import time

import click
from flask.cli import AppGroup
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from services.vector_backend import is_redis_vectors

redis_vectors_cli = AppGroup("redis-vectors", help="Redis vector search experiment.")


def _require_enabled():
    if not is_redis_vectors():
        click.echo("Redis vector search is off — set VECTOR_BACKEND=redis", err=True)
        sys.exit(1)


@redis_vectors_cli.command("rebuild")
def rebuild():
    """Wipe the Redis index and reload it from the DB."""
    _require_enabled()
    from jobs.redis_vector_store import get_store
    from models import JobPosting

    store = get_store()
    jobs = (
        JobPosting.query
        .filter(JobPosting.is_active.is_(True), JobPosting.embedding.isnot(None))
        .yield_per(500)
    )
    count = store.rebuild(jobs)
    click.echo(f"indexed {count} jobs into {store.index_name}")


@redis_vectors_cli.command("relay")
@click.option("--watch", is_flag=True, help="Keep running, polling the outbox.")
@click.option("--interval", default=2.0, show_default=True, help="Seconds between polls with --watch.")
@click.option("--batch-size", default=100, show_default=True)
def relay(watch, interval, batch_size):
    """Apply pending vector_outbox rows to Redis."""
    _require_enabled()
    from jobs.redis_vector_store import get_store
    from services.vector_outbox import relay_once

    store = get_store()
    if watch:
        click.echo(f"watching vector_outbox every {interval}s → {store.index_name} (Ctrl-C to stop)")
    try:
        while True:
            try:
                actions = relay_once(store, batch_size=batch_size)
            except (RedisConnectionError, RedisTimeoutError) as e:
                if not watch:
                    raise
                click.echo(f"redis unavailable, retrying ({e})", err=True)
                actions = []
            for action, job_id in actions:
                click.echo(f"{action} {store.key(job_id)}")
            if not watch:
                if not actions:
                    click.echo("outbox empty")
                return
            if len(actions) < batch_size:
                time.sleep(interval)
    except KeyboardInterrupt:
        click.echo("stopped")


@redis_vectors_cli.command("search")
@click.argument("text")
@click.option("-k", default=5, show_default=True)
def search(text, k):
    """Embed TEXT and print the k nearest jobs from Redis."""
    _require_enabled()
    from jobs import vector_store
    from jobs.redis_vector_store import get_store
    from models import JobPosting

    started = time.perf_counter()
    hits = get_store().search(vector_store._embed_query(text), k)
    elapsed_ms = (time.perf_counter() - started) * 1000
    titles = {
        j.id: j.title
        for j in JobPosting.query.filter(JobPosting.id.in_([jid for jid, _ in hits])).all()
    }
    click.echo(f"{len(hits)} hits in {elapsed_ms:.0f} ms (includes embedding the query)")
    for rank, (job_id, sim) in enumerate(hits, 1):
        click.echo(f"{rank:>3}  {job_id:>6}  {sim:.3f}  {titles.get(job_id, '?')}")
