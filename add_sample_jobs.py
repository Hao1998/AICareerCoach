"""
Script to add sample job postings for testing the CV-job matching feature.
Run this script to populate the database with sample jobs.
"""

from app import app, db
from models import JobPosting
from datetime import datetime

def add_sample_jobs():
    with app.app_context():
        # Clear existing jobs
        JobPosting.query.delete()

        # Sample jobs
        jobs = [
            {
                'title': 'Senior Python Developer',
                'company': 'Tech Innovations Inc.',
                'location': 'San Francisco, CA (Remote)',
                'job_type': 'Full-time',
                'salary_range': '$120k-$180k',
                'description': '''We are seeking an experienced Senior Python Developer to join our growing engineering team.
You will be responsible for designing and implementing scalable backend services, working with modern frameworks like Flask and Django,
and collaborating with cross-functional teams to deliver high-quality software solutions.

Key Responsibilities:
- Design and develop RESTful APIs and microservices
- Write clean, maintainable, and well-tested code
- Mentor junior developers and conduct code reviews
- Optimize application performance and scalability
- Participate in architectural decisions''',
                'requirements': '''Required Skills:
- 5+ years of professional Python development experience
- Strong knowledge of Flask, Django, or FastAPI
- Experience with SQL and NoSQL databases (PostgreSQL, MongoDB)
- Proficiency in Git and CI/CD pipelines
- Understanding of software design patterns and best practices
- Experience with cloud platforms (AWS, GCP, or Azure)

Preferred Qualifications:
- Experience with Docker and Kubernetes
- Knowledge of machine learning frameworks
- Contributions to open-source projects
- Bachelor's degree in Computer Science or related field'''
            },
            {
                'title': 'Machine Learning Engineer',
                'company': 'AI Solutions Corp',
                'location': 'New York, NY (Hybrid)',
                'job_type': 'Full-time',
                'salary_range': '$150k-$200k',
                'description': '''Join our AI team to build cutting-edge machine learning models that power our products.
You will work on natural language processing, computer vision, and recommendation systems.

Responsibilities:
- Develop and deploy ML models to production
- Conduct experiments and analyze results
- Optimize model performance and accuracy
- Collaborate with data engineers and product managers
- Stay updated with latest ML research and techniques''',
                'requirements': '''Requirements:
- Master's degree in Computer Science, Statistics, or related field
- 3+ years of ML engineering experience
- Strong Python programming skills
- Experience with TensorFlow, PyTorch, or scikit-learn
- Knowledge of deep learning architectures
- Experience with large-scale data processing
- Proficiency in SQL and data manipulation

Nice to Have:
- PhD in Machine Learning or AI
- Published research papers
- Experience with LLMs and transformers
- Knowledge of MLOps practices'''
            },
            {
                'title': 'Full Stack JavaScript Developer',
                'company': 'WebStart Technologies',
                'location': 'Austin, TX (Remote)',
                'job_type': 'Full-time',
                'salary_range': '$100k-$140k',
                'description': '''We're looking for a talented Full Stack Developer to build modern web applications.
You'll work with React, Node.js, and cloud technologies to create amazing user experiences.

What You'll Do:
- Build responsive web applications with React
- Develop backend services with Node.js and Express
- Design and implement RESTful APIs
- Work with databases and ensure data integrity
- Collaborate with designers and product team''',
                'requirements': '''Must Have:
- 4+ years of JavaScript development experience
- Expertise in React and modern frontend frameworks
- Strong Node.js and Express.js skills
- Experience with MongoDB or PostgreSQL
- Knowledge of HTML5, CSS3, and responsive design
- Understanding of authentication and security best practices

Bonus Points:
- TypeScript experience
- Experience with Next.js or React Native
- Knowledge of GraphQL
- Familiarity with AWS or other cloud platforms'''
            },
            {
                'title': 'Data Scientist',
                'company': 'Analytics Pro',
                'location': 'Boston, MA',
                'job_type': 'Full-time',
                'salary_range': '$130k-$170k',
                'description': '''Looking for a Data Scientist to extract insights from complex datasets and build predictive models.
You'll work on projects ranging from customer analytics to business intelligence.

Your Role:
- Analyze large datasets to identify trends and patterns
- Build statistical models and machine learning algorithms
- Create data visualizations and dashboards
- Present findings to stakeholders
- Collaborate with engineering teams to productionize models''',
                'requirements': '''Qualifications:
- Master's or PhD in Data Science, Statistics, or related field
- 3+ years of data science experience
- Strong programming skills in Python or R
- Experience with pandas, numpy, scikit-learn
- Proficiency in SQL and data manipulation
- Knowledge of statistical analysis and hypothesis testing
- Experience with data visualization tools (Tableau, PowerBI, or matplotlib)

Preferred:
- Experience with big data tools (Spark, Hadoop)
- Knowledge of A/B testing methodologies
- Business acumen and domain expertise'''
            },
            {
                'title': 'DevOps Engineer',
                'company': 'Cloud Systems Inc.',
                'location': 'Seattle, WA (Remote)',
                'job_type': 'Full-time',
                'salary_range': '$110k-$160k',
                'description': '''Join our infrastructure team to build and maintain scalable cloud systems.
You'll work with Kubernetes, Docker, and CI/CD pipelines to ensure reliable deployments.

Responsibilities:
- Design and implement CI/CD pipelines
- Manage cloud infrastructure on AWS or GCP
- Monitor system performance and reliability
- Automate deployment and scaling processes
- Ensure security and compliance standards''',
                'requirements': '''Requirements:
- 4+ years of DevOps or SRE experience
- Strong knowledge of Kubernetes and Docker
- Experience with AWS, GCP, or Azure
- Proficiency in infrastructure as code (Terraform, CloudFormation)
- Scripting skills in Python, Bash, or similar
- Understanding of networking and security
- Experience with monitoring tools (Prometheus, Grafana, DataDog)

Nice to Have:
- Certifications in AWS or Kubernetes
- Experience with service mesh (Istio, Linkerd)
- Knowledge of GitOps practices'''
            }
        ]

        # Add jobs to database
        for job_data in jobs:
            job = JobPosting(**job_data)
            db.session.add(job)

        db.session.commit()
        print(f"Successfully added {len(jobs)} sample job postings!")

        # Print summary
        print("\nSample Jobs Added:")
        for job in JobPosting.query.all():
            print(f"- {job.title} at {job.company}")

if __name__ == '__main__':
    add_sample_jobs()
