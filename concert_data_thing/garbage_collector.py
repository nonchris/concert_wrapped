"""Garbage collector for cleaning up old artifact folders."""

import os
import shutil
import time
from datetime import timedelta
from pathlib import Path
from threading import Event

from concert_data_thing.logger import LOGGING_PROVIDER

logger = LOGGING_PROVIDER.new_logger("concert_data_thing.garbage_collector")


def cleanup_old_folders(artifacts_folder: Path, max_age_hours: int | None = None) -> None:
    """
    Delete folders in artifacts_folder that are older than max_age_hours.

    Args:
        artifacts_folder: Path to the artifacts folder containing user_data_* folders.
        max_age_hours: Maximum age in hours before a folder is deleted. If None, reads from GC_MAX_AGE_HOURS env var (default: 24).
    """
    if max_age_hours is None:
        max_age_hours = int(os.environ.get("GC_MAX_AGE_HOURS", "24"))
    if not artifacts_folder.exists():
        logger.debug(f"Artifacts folder does not exist: {artifacts_folder}")
        return

    cutoff_time = time.time() - (max_age_hours * 60 * 60)
    deleted_count = 0
    total_size = 0

    try:
        for folder_path in artifacts_folder.iterdir():
            if not folder_path.is_dir():
                continue

            # Check if folder matches the user_data_* pattern
            if not folder_path.name.startswith("user_data_"):
                continue

            try:
                # Get folder modification time
                folder_mtime = folder_path.stat().st_mtime

                if folder_mtime < cutoff_time:
                    # Calculate folder size before deletion
                    folder_size = sum(f.stat().st_size for f in folder_path.rglob("*") if f.is_file())
                    total_size += folder_size

                    # Delete the folder and all its contents
                    shutil.rmtree(folder_path)
                    deleted_count += 1
                    logger.info(
                        f"Deleted old folder: {folder_path.name} "
                        f"(age: {timedelta(seconds=time.time() - folder_mtime)}, "
                        f"size: {folder_size / (1024 * 1024):.2f} MB)"
                    )
            except OSError as e:
                logger.warning(f"Error processing folder {folder_path.name}: {e}")

        if deleted_count > 0:
            logger.info(
                f"Cleanup completed: deleted {deleted_count} folder(s), " f"freed {total_size / (1024 * 1024):.2f} MB"
            )
        else:
            logger.debug("No old folders found to delete")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}", exc_info=True)


def run_garbage_collector_loop(stop_event: Event, interval_minutes: int | None = None) -> None:
    """
    Run garbage collector in a loop until stop_event is set.

    Args:
        stop_event: Event to signal when to stop the loop.
        interval_minutes: Interval in minutes between cleanup runs. If None, reads from GC_INTERVAL_MINUTES env var (default: 10).
    """
    logger.info(f"GC loop enabled, waiting one minute before first cleanup!")
    stop_event.wait(timeout=60)

    if interval_minutes is None:
        interval_minutes = int(os.environ.get("GC_INTERVAL_MINUTES", "10"))

    artifacts_path = os.environ.get("ARTIFACTS_PATH", "out")
    artifacts_folder = Path(artifacts_path)

    max_age_hours = int(os.environ.get("GC_MAX_AGE_HOURS", "24"))

    logger.info(f"GC_INTERVAL_MINUTES: {interval_minutes}")
    logger.info(f"GC_MAX_AGE_HOURS: {max_age_hours}")
    logger.info(f"ARTIFACTS_PATH: {artifacts_path}")

    logger.info(
        f"Starting garbage collector thread (interval: {interval_minutes} minutes, "
        f"max_age: {max_age_hours} hours, folder: {artifacts_folder})"
    )

    while not stop_event.is_set():
        try:
            cleanup_old_folders(artifacts_folder, max_age_hours=max_age_hours)
        except Exception as e:
            logger.error(f"Error in garbage collector loop: {e}", exc_info=True)

        # Wait for interval_minutes or until stop_event is set
        stop_event.wait(timeout=interval_minutes * 60)

    logger.info("Garbage collector thread stopped")
