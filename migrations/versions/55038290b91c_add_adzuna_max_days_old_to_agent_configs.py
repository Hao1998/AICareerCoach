"""add adzuna_max_days_old to agent_configs

Revision ID: 55038290b91c
Revises: 2c5e6f9c180
Create Date: 2026-02-04 17:17:45.258791

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '55038290b91c'
down_revision = '2c5e6f9c180'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('agent_configs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('adzuna_max_days_old', sa.Integer(), nullable=True))


def downgrade():
    with op.batch_alter_table('agent_configs', schema=None) as batch_op:
        batch_op.drop_column('adzuna_max_days_old')
