"""add vec_user_memories virtual table

Revision ID: e9e97b70ce4c
Revises: 0ac7c7de65f0
Create Date: 2026-06-25 15:41:43.712951

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e9e97b70ce4c'
down_revision = '0ac7c7de65f0'
branch_labels = None
depends_on = None


def upgrade():
    # Virtual tables are SQLite-only; skip on Postgres.
    bind = op.get_bind()
    if bind.dialect.name != "sqlite":
        return
    op.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_user_memories "
        "USING vec0(embedding float[768])"
    )


def downgrade():
    bind = op.get_bind()
    if bind.dialect.name != "sqlite":
        return
    op.execute("DROP TABLE IF EXISTS vec_user_memories")
