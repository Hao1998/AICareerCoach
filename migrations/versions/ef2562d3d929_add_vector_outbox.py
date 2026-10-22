"""add vector_outbox

Revision ID: ef2562d3d929
Revises: 3c81b22b9730
Create Date: 2026-10-20 16:19:37.768924

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ef2562d3d929'
down_revision = '3c81b22b9730'
branch_labels = None
depends_on = None


# Autogenerate also emitted drops for the pgvector columns/index
# (job_postings.embedding_vec, embedding_vec_updated_at,
# ix_job_postings_embedding_vec, user_memory_chunks.embedding_vec). Those are
# deliberately absent from models.py (raw-DDL only, see CLAUDE.md), so the
# drops were removed — the known Alembic limitation CLAUDE.md allows editing for.

def upgrade():
    op.create_table('vector_outbox',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('job_id', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('processed_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('vector_outbox', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_vector_outbox_job_id'), ['job_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_vector_outbox_processed_at'), ['processed_at'], unique=False)


def downgrade():
    with op.batch_alter_table('vector_outbox', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_vector_outbox_processed_at'))
        batch_op.drop_index(batch_op.f('ix_vector_outbox_job_id'))

    op.drop_table('vector_outbox')
