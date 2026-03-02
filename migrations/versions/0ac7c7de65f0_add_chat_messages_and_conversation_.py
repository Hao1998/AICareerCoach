"""add chat messages and conversation summary

Revision ID: 0ac7c7de65f0
Revises: 55038290b91c
Create Date: 2026-02-21 14:40:01.518901

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0ac7c7de65f0'
down_revision = '55038290b91c'
branch_labels = None
depends_on = None


def upgrade():
    # Create chat_messages table (if not already created by db.create_all)
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    if 'chat_messages' not in inspector.get_table_names():
        op.create_table('chat_messages',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.Column('role', sa.String(length=20), nullable=False),
            sa.Column('content', sa.Text(), nullable=False),
            sa.Column('timestamp', sa.DateTime(), nullable=True),
            sa.Column('intent', sa.String(length=50), nullable=True),
            sa.Column('action_data', sa.Text(), nullable=True),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_chat_messages_user_id'), 'chat_messages', ['user_id'], unique=False)
        op.create_index(op.f('ix_chat_messages_timestamp'), 'chat_messages', ['timestamp'], unique=False)

    # Add conversation_summary to agent_configs (if not already present)
    existing_cols = [c['name'] for c in inspector.get_columns('agent_configs')]
    if 'conversation_summary' not in existing_cols:
        with op.batch_alter_table('agent_configs', schema=None) as batch_op:
            batch_op.add_column(sa.Column('conversation_summary', sa.Text(), nullable=True))


def downgrade():
    with op.batch_alter_table('agent_configs', schema=None) as batch_op:
        batch_op.drop_column('conversation_summary')

    op.drop_index(op.f('ix_chat_messages_timestamp'), table_name='chat_messages')
    op.drop_index(op.f('ix_chat_messages_user_id'), table_name='chat_messages')
    op.drop_table('chat_messages')