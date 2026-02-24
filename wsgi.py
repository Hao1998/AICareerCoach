"""
WSGI Entry Point

This file creates the Flask application using the factory pattern.
Use this for:
- Running the application in production (gunicorn wsgi:app)
- Running Flask-Migrate commands (flask db ...)
- Running tests

The factory pattern allows migrations to work without requiring API keys.
"""

import os
from factory import create_app
from models import db

# Determine environment
env = os.getenv('FLASK_ENV', 'development')

# Create application instance
# When running migrations, Flask-Migrate will use this app
app = create_app(config_name=env)

# Register routes (import after app creation to avoid circular imports)
with app.app_context():
    # Import routes here to register them
    # This needs to be done after app creation but before running
    import routes  # We'll create this next

if __name__ == '__main__':
    # Build job index on startup if it doesn't exist
    from job_utils import build_job_faiss_index, JOB_VECTOR_INDEX
    import os.path

    try:
        if not os.path.exists(os.path.join(JOB_VECTOR_INDEX, "index.faiss")):
            print("Job index not found, building initial index...")
            with app.app_context():
                build_job_faiss_index()
            print("Job index built successfully")
    except Exception as e:
        print(f"Warning: Could not build job index on startup: {e}")

    # Initialize scheduler
    from agent_scheduler import init_scheduler
    from job_scout_agent import JobScoutAgent

    agent_scheduler = init_scheduler(app, JobScoutAgent)

    # Run the app
    app.run(host='0.0.0.0', port=5001, debug=(env == 'development'))