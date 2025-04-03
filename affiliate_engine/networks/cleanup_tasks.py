"""
Background tasks for postback handler cleanup.
"""

import logging
from datetime import UTC, datetime

from fastapi import BackgroundTasks

from ..analytics.signature_replay import SignatureReplay

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def cleanup_expired_signatures(
    background_tasks: BackgroundTasks, replay_protection: SignatureReplay
) -> None:
    """Clean up expired signatures and counters.

    Args:
        background_tasks: FastAPI background tasks
        replay_protection: SignatureReplay instance
    """
    try:
        cleared = replay_protection.clear_expired()
        logger.info(f"Cleaned up {cleared} expired signatures")

        # Schedule next cleanup in 1 hour
        background_tasks.add_task(cleanup_expired_signatures, background_tasks, replay_protection)
    except Exception as e:
        logger.error(f"Error in cleanup task: {e}")
        # Still schedule next cleanup even if this one failed
        background_tasks.add_task(cleanup_expired_signatures, background_tasks, replay_protection)


def start_cleanup_task(
    background_tasks: BackgroundTasks, replay_protection: SignatureReplay
) -> None:
    """Start the cleanup task on application startup.

    Args:
        background_tasks: FastAPI background tasks
        replay_protection: SignatureReplay instance
    """
    logger.info("Starting signature cleanup task")
    background_tasks.add_task(cleanup_expired_signatures, background_tasks, replay_protection)
