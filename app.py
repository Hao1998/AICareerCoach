"""
Entry Point

Start the application with:  python app.py
Run migrations with:         flask db migrate  (uses wsgi.py / factory pattern)
"""

import os
from factory import create_app
from job_utils import build_job_faiss_index, JOB_VECTOR_INDEX

env = os.getenv('FLASK_ENV', 'development')
app = create_app(config_name=env)

if __name__ == '__main__':
    with app.app_context():
        try:
            index_path = os.path.join(JOB_VECTOR_INDEX, "index.faiss")
            if not os.path.exists(index_path):
                print("Job index not found, building initial index...")
                build_job_faiss_index()
                print("Job index built successfully")
        except Exception as e:
            print(f"Warning: Could not build job index on startup: {e}")

    app.run(host='0.0.0.0', port=5001, debug=(env == 'development'))
