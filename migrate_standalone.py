"""
Standalone Migration script for multi-user support
This doesn't require importing app.py (avoiding langchain dependencies)
"""

import os
import sys
import shutil
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask

# Import db from models first
from models import db, User, Resume, JobMatch, JobPosting

# Create minimal Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///career_coach.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'dev-secret-key'

# Initialize database with the app
db.init_app(app)

def migrate_database():
    """Migrate the database to support multi-user functionality"""

    print("=" * 60)
    print("AI Career Coach - Multi-User Migration Script")
    print("=" * 60)

    with app.app_context():
        print("\n1. Creating new database tables...")
        db.create_all()
        print("   ✓ Tables created successfully")

        # Check if there's already a user
        existing_users = User.query.count()
        if existing_users > 0:
            print(f"\n   ℹ Users already exist ({existing_users} found). Skipping default user creation.")
            print("\n   Migration appears to have been run before.")
            print("   If you want to start fresh, delete career_coach.db and run again.")
            return

        print("\n2. Creating default admin user...")
        # Create a default admin user
        admin = User(
            username='admin',
            email='admin@aicareercoach.local',
            full_name='Admin User'
        )
        admin.set_password('admin123')  # Uses pbkdf2:sha256 for compatibility
        db.session.add(admin)
        db.session.commit()
        print(f"   ✓ Default user created:")
        print(f"     Email: admin@aicareercoach.local")
        print(f"     Password: admin123")
        print(f"     ⚠ PLEASE CHANGE THIS PASSWORD AFTER FIRST LOGIN!")

        print("\n3. Migrating existing data...")

        # Check if there are any existing job matches without user_id
        old_matches = JobMatch.query.filter_by(user_id=None).all()

        if old_matches:
            print(f"   Found {len(old_matches)} existing job matches to migrate...")

            # Migrate old uploads to user-specific directory
            upload_folder = 'uploads'
            if os.path.exists(upload_folder):
                admin_upload_dir = os.path.join(upload_folder, str(admin.id))
                os.makedirs(admin_upload_dir, exist_ok=True)

                # Get all PDF files in uploads directory (not in subdirectories)
                pdf_files = []
                for f in os.listdir(upload_folder):
                    file_path = os.path.join(upload_folder, f)
                    if os.path.isfile(file_path) and f.endswith('.pdf'):
                        pdf_files.append(f)

                print(f"   Found {len(pdf_files)} PDF files to migrate...")

                for pdf_file in pdf_files:
                    old_path = os.path.join(upload_folder, pdf_file)

                    # Create Resume record
                    resume = Resume(
                        user_id=admin.id,
                        filename=pdf_file,
                        original_filename=pdf_file,
                        file_path=os.path.join(admin_upload_dir, pdf_file)
                    )
                    db.session.add(resume)
                    db.session.flush()

                    # Move the file
                    new_path = os.path.join(admin_upload_dir, pdf_file)
                    try:
                        shutil.move(old_path, new_path)
                        print(f"   ✓ Migrated resume: {pdf_file}")
                    except Exception as e:
                        print(f"   ⚠ Warning: Could not move {pdf_file}: {e}")

                    # Update job matches for this resume
                    matches = JobMatch.query.filter_by(resume_filename=pdf_file, user_id=None).all()
                    for match in matches:
                        match.user_id = admin.id
                        match.resume_id = resume.id

                    if matches:
                        print(f"     Updated {len(matches)} job matches for this resume")

                db.session.commit()

            # Migrate vector index
            if os.path.exists('vector_index') and os.path.isdir('vector_index'):
                # Check if it's the old format (files directly in vector_index/)
                try:
                    vector_files = os.listdir('vector_index')
                    # Check if there are FAISS files directly in vector_index/
                    has_faiss_files = any(f.endswith('.faiss') or f.endswith('.pkl') for f in vector_files)

                    if has_faiss_files:
                        print("\n   Migrating vector index to user-specific directory...")
                        admin_vector_dir = os.path.join('vector_index', str(admin.id))
                        os.makedirs(admin_vector_dir, exist_ok=True)

                        # Move vector index files
                        for file in vector_files:
                            old_path = os.path.join('vector_index', file)
                            if os.path.isfile(old_path):
                                new_path = os.path.join(admin_vector_dir, file)
                                try:
                                    shutil.move(old_path, new_path)
                                except Exception as e:
                                    print(f"   ⚠ Warning: Could not move {file}: {e}")

                        print(f"   ✓ Vector index migrated to user-specific directory")
                except Exception as e:
                    print(f"   ⚠ Warning: Could not migrate vector index: {e}")

            print("\n   ✓ Data migration completed successfully")
        else:
            print("   No existing data to migrate")

        print("\n" + "=" * 60)
        print("Migration completed successfully!")
        print("=" * 60)
        print("\nYou can now:")
        print("1. Login with: admin@aicareercoach.local / admin123")
        print("2. Create new user accounts via the registration page")
        print("3. Each user will have their own isolated data")
        print("\n⚠ Remember to change the admin password!")
        print("=" * 60)


if __name__ == '__main__':
    try:
        migrate_database()
    except Exception as e:
        print(f"\n❌ Error during migration: {e}")
        print("\nPlease check the error and try again.")
        import traceback
        traceback.print_exc()
