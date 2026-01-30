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
        Start the background scheduler with bucketed scheduling

        Creates one APScheduler job per unique schedule_time from enabled users.
        This allows different users to have different schedule times efficiently.
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

            self.scheduler.start()
            self._is_running = True

            # Build schedule from database (creates jobs per unique time)
            with self.app.app_context():
                self.rebuild_schedule()

            logger.info("Agent Scheduler started successfully with bucketed scheduling")

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

    def _run_users_at_time(self, schedule_time):
        """
        Internal method called by scheduler for a specific time bucket

        Runs the Job Scout Agent for all users scheduled at this time

        Args:
            schedule_time: Time string in HH:MM format (e.g., "09:00")
        """
        logger.info(f"Starting scheduled Job Scout Agent run for time: {schedule_time}")

        try:
            with self.app.app_context():
                # Import here to avoid circular imports
                from models import AgentConfig

                # Get all enabled users with this schedule time
                user_configs = AgentConfig.query.filter_by(
                    is_enabled=True,
                    schedule_time=schedule_time
                ).all()

                if not user_configs:
                    logger.info(f"No enabled users found for schedule time {schedule_time}")
                    return []

                logger.info(f"Running agent for {len(user_configs)} users at {schedule_time}")

                # Create agent instance
                agent = self.agent_class(self.app.app_context())

                # Run for each user at this time
                results = []
                for user_config in user_configs:
                    try:
                        result = agent.run_for_user(user_config.user_id, run_type='scheduled')
                        results.append({
                            'user_id': user_config.user_id,
                            'result': result
                        })
                    except Exception as user_error:
                        logger.error(f"Error running agent for user {user_config.user_id}: {user_error}")
                        results.append({
                            'user_id': user_config.user_id,
                            'result': {
                                'status': 'failed',
                                'error': str(user_error)
                            }
                        })

                # Log summary
                successful = len([r for r in results if r['result']['status'] == 'success'])
                failed = len([r for r in results if r['result']['status'] == 'failed'])

                logger.info(f"Scheduled run at {schedule_time} completed: {successful} successful, {failed} failed")

                return results

        except Exception as e:
            logger.error(f"Error in scheduled agent run at {schedule_time}: {e}")
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

    def rebuild_schedule(self):
        """
        Rebuild scheduler jobs based on current database configuration

        This method:
        1. Removes all existing schedule jobs
        2. Queries database for unique schedule times from enabled users
        3. Creates one APScheduler job per unique time bucket

        Should be called when:
        - Scheduler starts
        - User enables/disables agent
        - User changes schedule time
        """
        if not self.scheduler or not self._is_running:
            logger.warning("Cannot rebuild schedule: scheduler not running")
            return False

        try:
            # Import here to avoid circular imports
            from models import AgentConfig

            # Remove all existing schedule jobs (keep manual trigger jobs)
            existing_jobs = self.scheduler.get_jobs()
            for job in existing_jobs:
                if job.id.startswith('schedule_'):
                    self.scheduler.remove_job(job.id)
                    logger.debug(f"Removed old schedule job: {job.id}")

            # Get unique schedule times from enabled users
            unique_times = AgentConfig.query.with_entities(AgentConfig.schedule_time) \
                .filter(AgentConfig.is_enabled == True) \
                .distinct() \
                .all()

            if not unique_times:
                logger.info("No enabled users found, no schedule jobs created")
                return True

            # Create one job per unique time bucket
            job_count = 0
            for (schedule_time,) in unique_times:
                try:
                    # Parse time string (HH:MM)
                    hour, minute = map(int, schedule_time.split(':'))

                    # Validate time
                    if not (0 <= hour <= 23 and 0 <= minute <= 59):
                        logger.warning(f"Invalid schedule time: {schedule_time}, skipping")
                        continue

                    # Create job for this time bucket
                    job_id = f'schedule_{schedule_time.replace(":", "")}'
                    self.scheduler.add_job(
                        func=self._run_users_at_time,
                        trigger=CronTrigger(hour=hour, minute=minute),
                        id=job_id,
                        name=f'Job Scout Agent - {schedule_time}',
                        args=[schedule_time],
                        replace_existing=True
                    )

                    job_count += 1
                    logger.info(f"Scheduled job for time bucket: {schedule_time}")

                except ValueError as e:
                    logger.error(f"Invalid time format '{schedule_time}': {e}")
                    continue

            logger.info(f"Schedule rebuilt: {job_count} time buckets created")
            return True

        except Exception as e:
            logger.error(f"Failed to rebuild schedule: {e}")
            return False

    def get_next_run_time(self):
        """
        Get next scheduled run time across all schedule buckets

        Returns:
            datetime: Next scheduled run time, or None if scheduler not running
        """
        if not self.scheduler or not self._is_running:
            return None

        # Get all schedule jobs and find earliest next run time
        next_times = []
        for job in self.scheduler.get_jobs():
            if job.id.startswith('schedule_') and job.next_run_time:
                next_times.append(job.next_run_time)

        return min(next_times) if next_times else None

    def get_job_info(self):
        """""
        Get information about scheduled jobs with user counts per time bucket

         Returns:
             list: List of job information dictionaries including user counts
         """
        if not self.scheduler or not self._is_running:
            return []

        try:
            with self.app.app_context():
                from models import AgentConfig

                jobs = []
                for job in self.scheduler.get_jobs():
                    if job.id.startswith('schedule_'):
                        # Extract schedule time from job args
                        schedule_time = job.args[0] if job.args else None

                        # Count users for this time bucket
                        user_count = 0
                        if schedule_time:
                            user_count = AgentConfig.query.filter_by(
                                is_enabled=True,
                                schedule_time=schedule_time
                            ).count()

                        jobs.append({
                            'id': job.id,
                            'name': job.name,
                            'schedule_time': schedule_time,
                            'user_count': user_count,
                            'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
                            'trigger': str(job.trigger)
                        })

                return jobs

        except Exception as e:
            logger.error(f"Error getting job info: {e}")
            return []


def update_user_schedule(self, user_id, schedule_time):
    """
    Update schedule time for a specific user and rebuild scheduler

    Args:
        user_id: User ID to update
        schedule_time: New schedule time in HH:MM format

    Returns:
        bool: True if successful, False otherwise
    """
    if not self.scheduler or not self._is_running:
        logger.warning("Cannot update schedule: scheduler not running")
        return False

    try:
        with self.app.app_context():
            from models import AgentConfig, db

            # Validate time format
            hour, minute = map(int, schedule_time.split(':'))
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                logger.error(f"Invalid schedule time: {schedule_time}")
                return False

            # Update user's schedule time in database
            user_config = AgentConfig.query.filter_by(user_id=user_id).first()
            if not user_config:
                logger.error(f"AgentConfig not found for user {user_id}")
                return False

            user_config.schedule_time = schedule_time
            db.session.commit()

            logger.info(f"Updated schedule for user {user_id} to {schedule_time}")

            # Rebuild entire schedule to reflect changes
            self.rebuild_schedule()

            return True

    except ValueError as e:
        logger.error(f"Invalid time format '{schedule_time}': {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to update schedule: {e}")
        return False


def toggle_user_agent(self, user_id, is_enabled):
    """
    Enable or disable agent for a user and rebuild scheduler

    Args:
        user_id: User ID to update
        is_enabled: True to enable, False to disable

    Returns:
        bool: True if successful, False otherwise
    """
    if not self.scheduler or not self._is_running:
        logger.warning("Cannot toggle agent: scheduler not running")
        return False

    try:
        with self.app.app_context():
            from models import AgentConfig, db

            user_config = AgentConfig.query.filter_by(user_id=user_id).first()
            if not user_config:
                logger.error(f"AgentConfig not found for user {user_id}")
                return False

            user_config.is_enabled = is_enabled
            db.session.commit()

            logger.info(f"{'Enabled' if is_enabled else 'Disabled'} agent for user {user_id}")

            # Rebuild schedule to add/remove user from time buckets
            self.rebuild_schedule()

            return True

    except Exception as e:
        logger.error(f"Failed to toggle agent: {e}")
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
