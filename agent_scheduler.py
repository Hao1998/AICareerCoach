"""
Agent Scheduler - APScheduler integration for Job Scout Agent

This module manages the scheduling of the autonomous Job Scout Agent using APScheduler.
Supports both automatic scheduled runs and manual triggers.
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentScheduler:
    """
    Manages scheduling for the Job Scout Agent

    Uses APScheduler BackgroundScheduler to run the agent:
    - Automatically at scheduled times (daily, configurable per user)
    - On-demand via manual trigger
    """

    def __init__(self, app, agent_class):
        """
        Initialize the agent scheduler

        Args:
            app: Flask application instance
            agent_class: JobScoutAgent class (not instance)
        """
        self.app = app
        self.agent_class = agent_class
        self.scheduler = None
        self._is_running = False

    def start(self):
        """
        Start the background scheduler

        This starts APScheduler which will run in a background thread
        """
        if self._is_running:
            logger.warning("Scheduler already running")
            return

        try:
            # Create BackgroundScheduler
            # This runs in a separate thread and doesn't block Flask
            self.scheduler = BackgroundScheduler(
                daemon=True,  # Daemon thread exits when main program exits
                job_defaults={
                    'coalesce': True,  # Combine missed runs
                    'max_instances': 1  # Only one instance of each job at a time
                }
            )

            # Add daily job to run agent for all users
            # Default: Run every day at 9:00 AM
            self.scheduler.add_job(
                func=self._run_scheduled_agent,
                trigger=CronTrigger(hour=9, minute=0),  # Every day at 9:00 AM
                id='job_scout_agent_daily',
                name='Job Scout Agent - Daily Run',
                replace_existing=True
            )

            self.scheduler.start()
            self._is_running = True

            logger.info("Agent Scheduler started successfully")
            logger.info("Job Scout Agent scheduled to run daily at 9:00 AM")

        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            raise

    def stop(self):
        """Stop the background scheduler"""
        if self.scheduler and self._is_running:
            self.scheduler.shutdown(wait=False)
            self._is_running = False
            logger.info("Agent Scheduler stopped")

    def is_running(self):
        """Check if scheduler is running"""
        return self._is_running

    def _run_scheduled_agent(self):
        """
        Internal method called by scheduler

        Runs the Job Scout Agent for all enabled users
        """
        logger.info("Starting scheduled Job Scout Agent run")

        try:
            with self.app.app_context():
                # Create agent instance with app context
                agent = self.agent_class(self.app.app_context())

                # Run for all users with enabled agents
                results = agent.run_for_all_users(run_type='scheduled')

                # Log results
                successful = len([r for r in results if r['result']['status'] == 'success'])
                failed = len([r for r in results if r['result']['status'] == 'failed'])

                logger.info(f"Scheduled agent run completed: {successful} successful, {failed} failed")

                return results

        except Exception as e:
            logger.error(f"Error in scheduled agent run: {e}")
            return []

    def trigger_manual_run(self, user_id):
        """
        Trigger manual agent run for a specific user

        This is called when user clicks "Check Now" button

        Args:
            user_id: User ID to run agent for

        Returns:
            dict: Results of the agent run
        """
        logger.info(f"Manual agent run triggered for user {user_id}")

        try:
            with self.app.app_context():
                # Create agent instance
                agent = self.agent_class(self.app.app_context())

                # Run for specific user
                result = agent.run_for_user(user_id, run_type='manual')

                logger.info(f"Manual agent run completed for user {user_id}: {result['status']}")

                return result

        except Exception as e:
            logger.error(f"Error in manual agent run: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def get_next_run_time(self):
        """
        Get next scheduled run time

        Returns:
            datetime: Next scheduled run time, or None if scheduler not running
        """
        if not self.scheduler or not self._is_running:
            return None

        job = self.scheduler.get_job('job_scout_agent_daily')
        if job:
            return job.next_run_time

        return None

    def get_job_info(self):
        """
        Get information about scheduled jobs

        Returns:
            list: List of job information dictionaries
        """
        if not self.scheduler or not self._is_running:
            return []

        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger)
            })

        return jobs

    def update_schedule(self, hour=9, minute=0):
        """
        Update the schedule time for the agent

        Args:
            hour: Hour to run (0-23)
            minute: Minute to run (0-59)
        """
        if not self.scheduler or not self._is_running:
            logger.warning("Cannot update schedule: scheduler not running")
            return False

        try:
            # Remove old job
            self.scheduler.remove_job('job_scout_agent_daily')

            # Add new job with updated time
            self.scheduler.add_job(
                func=self._run_scheduled_agent,
                trigger=CronTrigger(hour=hour, minute=minute),
                id='job_scout_agent_daily',
                name='Job Scout Agent - Daily Run',
                replace_existing=True
            )

            logger.info(f"Schedule updated: Agent will run daily at {hour:02d}:{minute:02d}")
            return True

        except Exception as e:
            logger.error(f"Failed to update schedule: {e}")
            return False


def init_scheduler(app, agent_class):
    """
    Initialize and start the agent scheduler

    Call this from app.py after Flask app is created

    Args:
        app: Flask application instance
        agent_class: JobScoutAgent class

    Returns:
        AgentScheduler: Scheduler instance
    """
    scheduler = AgentScheduler(app, agent_class)

    # Start scheduler automatically
    # Only start if not in debug mode to avoid duplicate runs during auto-reload
    if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        scheduler.start()

    return scheduler