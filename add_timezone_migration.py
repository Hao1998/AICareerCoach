"""
Migration script to add timezone column to agent_configs table

Run this script to update the database schema:
    python add_timezone_migration.py
"""

from app import app, db
from models import AgentConfig
from sqlalchemy import inspect

def migrate():
    with app.app_context():
        # Check if timezone column exists
        inspector = inspect(db.engine)
        columns = [col['name'] for col in inspector.get_columns('agent_configs')]

        if 'timezone' not in columns:
            print("Adding timezone column to agent_configs table...")
            # Add the timezone column with default value 'UTC'
            db.session.execute(db.text("ALTER TABLE agent_configs ADD COLUMN timezone VARCHAR(50) DEFAULT 'UTC'"))
            db.session.commit()
            print("✓ Timezone column added successfully")
        else:
            print("✓ Timezone column already exists")

if __name__ == '__main__':
    migrate()