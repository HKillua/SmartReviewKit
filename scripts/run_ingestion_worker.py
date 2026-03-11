"""Run the separate-process ingestion worker."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.settings import load_settings
from src.ingestion.worker import IngestionWorker


async def _run(settings_path: str) -> None:
    worker = IngestionWorker(load_settings(settings_path))
    loop = asyncio.get_running_loop()

    def _handle_signal(sig_name: str) -> None:
        logging.getLogger(__name__).info("Received %s, stopping ingestion worker...", sig_name)
        worker.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal, sig.name)
        except NotImplementedError:  # pragma: no cover - platform specific
            signal.signal(sig, lambda *_args, _name=sig.name: _handle_signal(_name))

    await worker.run_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ingestion worker against the configured Postgres task store.")
    parser.add_argument("--settings", default="config/settings.storage_stack.yaml", help="Path to settings YAML")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(_run(args.settings))
    except KeyboardInterrupt:  # pragma: no cover - signal handlers should handle this
        logging.getLogger(__name__).info("Ingestion worker stopped.")


if __name__ == "__main__":
    main()
