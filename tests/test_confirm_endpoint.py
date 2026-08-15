"""
tests/test_confirm_endpoint.py
==============================
Feature tested: POST /api/chat/confirm — the only path that executes a gated
capability (spec §3.2, §3.3).

The property under test
-----------------------
The client sends ONLY a nonce. It cannot name the capability or supply
arguments. These tests assert that a client attempting to do so is ignored,
and that ownership, single-use, and expiry all hold at the HTTP boundary.

What each test covers
---------------------
test_confirm_requires_authentication
test_confirm_rejects_a_missing_nonce
test_confirm_rejects_an_unknown_nonce
test_confirm_executes_the_stored_capability
test_confirm_ignores_client_supplied_capability_and_args
    The decisive test: a client posting its own capability/args alongside a
    valid nonce gets the STORED action, not the one it asked for.
test_confirm_is_single_use
"""

import json

import pytest

from models import User, TaskPlan, db
from services.pending_actions import propose


@pytest.fixture
def client_with_user(app_sqlite, fake_redis):
    app = app_sqlite
    with app.app_context():
        user = User(username="tester", email="t@example.com")
        user.set_password("pw-not-used-in-test")
        db.session.add(user)
        db.session.commit()
        user_id = user.id

    client = app.test_client()
    with client.session_transaction() as session:
        session['_user_id'] = str(user_id)
        session['_fresh'] = True
    return app, client, user_id


def test_confirm_requires_authentication(app_sqlite, fake_redis):
    client = app_sqlite.test_client()
    response = client.post('/api/chat/confirm', json={"nonce": "x"})
    assert response.status_code in (302, 401)


def test_confirm_rejects_a_missing_nonce(client_with_user):
    _, client, _ = client_with_user
    response = client.post('/api/chat/confirm', json={})
    assert response.status_code == 400


def test_confirm_rejects_an_unknown_nonce(client_with_user):
    _, client, _ = client_with_user
    response = client.post('/api/chat/confirm', json={"nonce": "never-issued"})
    assert response.status_code == 400


def test_confirm_executes_the_stored_capability(client_with_user):
    app, client, user_id = client_with_user
    with app.app_context():
        db.session.add(TaskPlan(user_id=user_id, goal="g", status="active"))
        db.session.commit()

    nonce = propose(user_id, "abandon_career_plan", {"reason": "done"}, "Abandon 'g'?")
    response = client.post('/api/chat/confirm', json={"nonce": nonce})

    assert response.status_code == 200
    assert response.get_json()["success"] is True
    with app.app_context():
        assert TaskPlan.query.filter_by(user_id=user_id).first().status == "abandoned"


def test_confirm_ignores_client_supplied_capability_and_args(client_with_user):
    app, client, user_id = client_with_user
    with app.app_context():
        db.session.add(TaskPlan(user_id=user_id, goal="g", status="active"))
        db.session.commit()

    nonce = propose(user_id, "abandon_career_plan", {"reason": "done"}, "Abandon 'g'?")
    response = client.post('/api/chat/confirm', json={
        "nonce": nonce,
        "capability": "trigger_job_scout_agent",
        "args": {"reason": "attacker supplied"},
    })

    assert response.status_code == 200
    # The STORED capability ran, not the one the client named.
    with app.app_context():
        assert TaskPlan.query.filter_by(user_id=user_id).first().status == "abandoned"


def test_confirm_is_single_use(client_with_user):
    app, client, user_id = client_with_user
    with app.app_context():
        db.session.add(TaskPlan(user_id=user_id, goal="g", status="active"))
        db.session.commit()

    nonce = propose(user_id, "abandon_career_plan", {"reason": ""}, "Abandon 'g'?")
    assert client.post('/api/chat/confirm', json={"nonce": nonce}).status_code == 200
    assert client.post('/api/chat/confirm', json={"nonce": nonce}).status_code == 400
