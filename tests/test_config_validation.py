"""
tests/test_config_validation.py
=================================
Feature tested: production configuration guardrails in factory._validate_config.

A production deployment must never run with the default development SECRET_KEY —
that key is public (it lives in config.py), so anyone could forge session cookies
and impersonate users. The production gunicorn entry point (wsgi.py) passes
skip_api_check=True, so the guard must fire independently of that flag.

Test config (TESTING=True) and development (DEBUG=True) are exempt so the local
workflow and the test suite keep working without extra env setup.
"""

import pytest

from factory import create_app


def test_production_rejects_default_secret_key():
    # No SECRET_KEY env var set in the test environment -> falls back to the
    # public dev default -> must refuse to boot in production.
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        create_app("production", skip_api_check=True)


def test_test_config_boots_without_secret_key():
    # TESTING=True is exempt from the production SECRET_KEY guard.
    app = create_app("test", skip_api_check=True)
    assert app is not None


def test_production_accepts_a_real_secret_key():
    # Exercise the guard directly with a production-like config so we don't have
    # to reload modules (which would mutate factory's shared globals). A valid
    # SECRET_KEY plus skip_api_check=True must pass without raising.
    from types import SimpleNamespace
    from factory import _validate_config

    fake_app = SimpleNamespace(config={
        "DEBUG": False,
        "TESTING": False,
        "SECRET_KEY": "a-real-random-production-secret",
    })
    _validate_config(fake_app, skip_api_check=True)  # should not raise


def test_production_rejects_missing_adzuna_keys():
    # With API checks enabled and a valid secret, production must still refuse to
    # boot when Adzuna credentials are absent.
    from types import SimpleNamespace
    from factory import _validate_config

    fake_app = SimpleNamespace(config={
        "DEBUG": False,
        "TESTING": False,
        "SECRET_KEY": "a-real-random-production-secret",
        "XAI_API_KEY": "xai-test-key",
        "ADZUNA_APP_ID": None,
        "ADZUNA_APP_KEY": None,
    })
    with pytest.raises(RuntimeError, match="Adzuna"):
        _validate_config(fake_app, skip_api_check=False)
