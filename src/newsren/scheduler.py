"""Optional in-process scheduler (``NEWSREN_SCHEDULER=1``).

Runs the full pipeline periodically in a background thread. Production
deployments can instead call ``newsren pipeline`` from cron.
"""

from __future__ import annotations

import logging
import threading

from newsren.config import get_settings
from newsren.db import session_scope

log = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, interval_minutes: int):
        self.interval_seconds = max(60, interval_minutes * 60)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="newsren-scheduler", daemon=True)

    def start(self) -> Scheduler:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def run_once(self) -> None:
        from newsren.pipeline import run_pipeline

        try:
            with session_scope() as session:
                report = run_pipeline(session)
            log.info("Scheduled pipeline run: %s", report.as_dict())
        except Exception:
            log.exception("Scheduled pipeline run failed")

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.run_once()
            self._stop.wait(self.interval_seconds)


def start_scheduler() -> Scheduler:
    return Scheduler(get_settings().scheduler_interval_minutes).start()
