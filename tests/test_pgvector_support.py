"""
tests/test_pgvector_support.py
==============================
Dialect detection and vector literal formatting.

These run on SQLite (the default test DB), so is_postgres() must report False
and the literal formatter must work with no database at all.
"""
from factory import create_app
from services.pgvector_support import (
    EMBEDDING_DIM,
    is_postgres,
    to_vector_literal,
)


def test_embedding_dim_is_768():
    assert EMBEDDING_DIM == 768


def test_to_vector_literal_formats_pgvector_syntax():
    assert to_vector_literal([1.0, 2.5, -3.0]) == "[1.0,2.5,-3.0]"


def test_to_vector_literal_coerces_ints_and_numpy_scalars():
    import numpy as np
    assert to_vector_literal([1, 2]) == "[1.0,2.0]"
    assert to_vector_literal(np.array([0.5, 0.25])) == "[0.5,0.25]"


def test_is_postgres_false_on_sqlite():
    app = create_app('test', skip_api_check=True)
    with app.app_context():
        assert is_postgres() is False


def test_is_postgres_false_outside_app_context():
    # Celery beat imports these modules before an app context exists.
    assert is_postgres() is False
