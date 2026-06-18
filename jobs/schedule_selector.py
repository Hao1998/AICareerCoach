"""
Schedule selection for the Job Scout dispatcher.

Pure logic, intentionally free of Celery and DB imports so it can be unit-tested
in isolation and reused by any scheduler driver. The Celery Beat dispatcher runs
once per minute and uses due_user_ids() to decide which enabled users should have
a scout run enqueued *now*, evaluated in each user's own timezone.
"""

import pytz


def due_user_ids(configs, now_utc):
    """Return user_ids whose local HH:MM (in their timezone) equals schedule_time.

    configs: iterable of objects with .user_id, .schedule_time ('HH:MM' or falsy),
             .timezone (IANA string or None), .is_enabled (bool).
    now_utc: a UTC datetime (naive is treated as UTC).
    """
    if now_utc.tzinfo is None:
        now_utc = pytz.UTC.localize(now_utc)

    due = []
    for cfg in configs:
        if not getattr(cfg, "is_enabled", False):
            continue
        if not cfg.schedule_time:
            continue
        try:
            tz = pytz.timezone(cfg.timezone) if cfg.timezone else pytz.UTC
        except pytz.exceptions.UnknownTimeZoneError:
            tz = pytz.UTC
        local_hhmm = now_utc.astimezone(tz).strftime("%H:%M")
        if local_hhmm == cfg.schedule_time:
            due.append(cfg.user_id)
    return due
