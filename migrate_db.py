"""
Database migration script to add embedding columns to JobPosting table

This script adds:
- embedding: PickleType column to store pre-computed job embeddings
- embedding_updated_at: DateTime column to track when embedding was computed

Run this once after updating the models.py file.
"""

import sqlite3
import os

DB_PATH = 'career_coach.db'

def migrate_database():
    """Add new columns to job_postings table"""

    if not os.path.exists(DB_PATH):
        print(f"Database {DB_PATH} not found. It will be created when the app runs.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if columns already exist
    cursor.execute("PRAGMA table_info(job_postings)")
    columns = [column[1] for column in cursor.fetchall()]

    changes_made = False

    # Add embedding column if it doesn't exist
    if 'embedding' not in columns:
        print("Adding 'embedding' column...")
        cursor.execute("""
            ALTER TABLE job_postings
            ADD COLUMN embedding BLOB
        """)
        changes_made = True
        print("✓ Added 'embedding' column")
    else:
        print("✓ 'embedding' column already exists")

    # Add embedding_updated_at column if it doesn't exist
    if 'embedding_updated_at' not in columns:
        print("Adding 'embedding_updated_at' column...")
        cursor.execute("""
            ALTER TABLE job_postings
            ADD COLUMN embedding_updated_at DATETIME
        """)
        changes_made = True
        print("✓ Added 'embedding_updated_at' column")
    else:
        print("✓ 'embedding_updated_at' column already exists")

    if changes_made:
        conn.commit()
        print("\n✅ Database migration completed successfully!")
        print("\nNext steps:")
        print("1. Run the Flask app: python app.py")
        print("2. The app will automatically compute embeddings for all jobs on startup")
        print("3. Or manually rebuild index via: POST /jobs/rebuild-index")
    else:
        print("\n✅ Database is already up to date!")

    conn.close()

if __name__ == '__main__':
    print("="*60)
    print("Database Migration: Adding Embedding Columns")
    print("="*60)
    migrate_database()
