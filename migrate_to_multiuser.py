"""
Migration script to add multi-user support to AI Career Coach

This script:
1. Creates new tables (users, resumes)
2. Adds user_id and resume_id columns to job_matches
3. Creates a default admin user
4. Migrates existing data to the new schema
"""

from app import app, db
from models import User, Resume, JobMatch
from datetime import datetime
import os
import shutil


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
        admin.set_password('admin123')  # Change this in production!
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
            upload_folder = app.config.get('UPLOAD_FOLDER', 'uploads')
            if os.path.exists(upload_folder):
                admin_upload_dir = os.path.join(upload_folder, str(admin.id))
                os.makedirs(admin_upload_dir, exist_ok=True)

                # Get all PDF files in uploads directory
                pdf_files = [f for f in os.listdir(upload_folder) if f.endswith('.pdf')]

                for pdf_file in pdf_files:
                    old_path = os.path.join(upload_folder, pdf_file)
                    if os.path.isfile(old_path):  # Only move files, not directories
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
                        shutil.move(old_path, new_path)
                        print(f"   ✓ Migrated resume: {pdf_file}")

                        # Update job matches for this resume
                        matches = JobMatch.query.filter_by(resume_filename=pdf_file).all()
                        for match in matches:
                            match.user_id = admin.id
                            match.resume_id = resume.id
                        print(f"     Updated {len(matches)} job matches for this resume")

                db.session.commit()

            # Migrate vector index
            if os.path.exists('vector_index') and os.path.isdir('vector_index'):
                # Check if it's the old format (files directly in vector_index/)
                vector_files = os.listdir('vector_index')
                if any(f.endswith('.faiss') or f.endswith('.pkl') for f in vector_files):
                    print("\n   Migrating vector index to user-specific directory...")
                    admin_vector_dir = os.path.join('vector_index', str(admin.id))
                    os.makedirs(admin_vector_dir, exist_ok=True)

                    # Move vector index files
                    for file in vector_files:
                        if not os.path.isdir(os.path.join('vector_index', file)):
                            old_path = os.path.join('vector_index', file)
                            new_path = os.path.join(admin_vector_dir, file)
                            shutil.move(old_path, new_path)
                    print(f"   ✓ Vector index migrated to user-specific directory")

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
